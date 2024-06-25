import numpy as np
import tqdm.autonotebook
from pathlib import Path

import gmsh
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element
from ufl import (FacetNormal, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, dx, inner, lhs, grad, nabla_grad, rhs)

from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.io import (VTXWriter, gmshio, XDMFFile)
from dolfinx.mesh import locate_entities_boundary


####################################################
#                                                  #
#           MESH PARAMETERS                        #
#                                                  #
####################################################

L = 2.2
H = 0.41
c_x = c_y = 0.2
r = 0.05
gdim = 2
res_min = r / 3
mesh_order = 2

####################################################
#                                                  #
#           SET UP MESH                            #
#                                                  #
####################################################

gmsh.initialize()

mesh_comm = MPI.COMM_WORLD
model_rank = 0

# mesh markers
fluid_marker = 1
obstacle_marker = 2
inlet_marker, outlet_marker, wall_marker, edge_marker = 3, 4, 5, 6

if mesh_comm.rank == model_rank:

    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)], tag=3)[0][0][1]
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities(dim=gdim)
    assert (len(volumes) == 2)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")
    gmsh.model.addPhysicalGroup(volumes[1][0], [volumes[1][1]], obstacle_marker)
    gmsh.model.setPhysicalName(volumes[1][0], obstacle_marker, "Obstacle")

    # set sections
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    inflow, outflow, walls, edge = [], [], [], []
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, H / 2, 0]):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L, H / 2, 0]):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(center_of_mass, [L / 2, 0, 0]):
            walls.append(boundary[1])
        else:
            edge.append(boundary[1])
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, edge, edge_marker)
    gmsh.model.setPhysicalName(1, edge_marker, "Obstacle Edge")

    # set resolution
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", edge)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    # generate the mesh
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(mesh_order)
    gmsh.model.mesh.optimize("Netgen")

mesh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
ft.name = "Facet markers"


####################################################
#                                                  #
#           PROBLEM PARAMETERS                     #
#                                                  #
####################################################

t = 0
T = 2                      # Final time
dt = 1 / 250                 # Time step size
num_steps = int(T / dt)
k = Constant(mesh, PETSc.ScalarType(dt))
mu = Constant(mesh, PETSc.ScalarType(0.001))  # Dynamic viscosity
rho = Constant(mesh, PETSc.ScalarType(1))     # Density

class InletVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41**2)
        return values

class DiskDisplacement():
    def __init__(self, t, dt):
        self.t = t
        self.dt = dt

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        freq = [0.4, 0.6]  # Hz
        ampl = [0.05, 0.03]
        values[0] = ampl[0]  * np.sin(self.t * 2*np.pi * freq[0]) * 2*np.pi * freq[0] * self.dt          # x shift
        values[1] = ampl[1]  * np.cos(self.t * 2*np.pi * freq[1]) * 2*np.pi * freq[1] * self.dt          # y shift
        return values
    

####################################################
#                                                  #
#           SET UP PROBLEM                         #
#                                                  #
####################################################

def _all_interior_surfaces(x):
    eps = 1.0e-5
    x_min = 0
    x_max = L
    y_min = 0
    y_max = H

    x_mid = np.logical_and(
        x_min + eps < x[0], x[0] < x_max - eps
    )
    y_mid = np.logical_and(
        y_min + eps < x[1], x[1] < y_max - eps
    )
    return np.logical_and( x_mid, y_mid )

def _all_exterior_surfaces(x):
    eps = 1.0e-5
    x_min = 0
    x_max = L
    y_min = 0
    y_max = H

    x_edge = np.logical_or(
        x[0] < x_min + eps, x_max - eps < x[0]
    )
    y_edge = np.logical_or(
        x[1] < y_min + eps, y_max - eps < x[1]
    )
    return np.logical_or( x_edge, y_edge )

# boundary conditions
v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
v_cg_mesh = element("Lagrange", mesh.topology.cell_name(), mesh_order, shape=(mesh.geometry.dim, ))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
V_mesh = functionspace(mesh, v_cg_mesh)
Q = functionspace(mesh, s_cg1)

fdim = mesh.topology.dim - 1

# get interior points
facet_dim = 1
all_interior_facets = locate_entities_boundary(
    mesh, facet_dim, _all_interior_surfaces
)
all_interior_V_mesh_dofs = locate_dofs_topological(
    V_mesh, facet_dim, all_interior_facets
)
all_exterior_facets = locate_entities_boundary(
    mesh, facet_dim, _all_exterior_surfaces
)
all_exterior_V_mesh_dofs = locate_dofs_topological(
    V_mesh, facet_dim, all_exterior_facets
)

# Mesh
mesh_delta = DiskDisplacement(t, dt)
mesh_displacement_bc = Function(V_mesh)
mesh_displacement_bc.interpolate(mesh_delta)
bcx_in = dirichletbc(mesh_displacement_bc, all_interior_V_mesh_dofs)
bcx_out = dirichletbc(Constant(mesh, PETSc.ScalarType((0.0, 0.0))), all_exterior_V_mesh_dofs, V_mesh)
bcx = [bcx_in, bcx_out]
# Inlet
u_inlet = Function(V)
inlet_velocity = InletVelocity(t)
u_inlet.interpolate(inlet_velocity)
bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(inlet_marker)))
# Walls
u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu_walls = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(wall_marker)), V)
# Obstacle
bcu_obstacle = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(edge_marker)), V)
bcu = [bcu_inflow, bcu_obstacle, bcu_walls]
# Outlet
bcp_outlet = dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(Q, fdim, ft.find(outlet_marker)), Q)
bcp = [bcp_outlet]

####################################################
#                                                  #
#           SET UP EQUATIONS                       #
#                                                  #
####################################################

# Mesh movement setup
mesh_displacement = Function(V_mesh)
total_mesh_displacement = Function(V_mesh)
total_mesh_displacement.name = "Mesh Displacement"

# Variational form setup
u = TrialFunction(V)
v = TestFunction(V)
u_mesh = TrialFunction(V_mesh)
v_mesh = TestFunction(V_mesh)
u_ = Function(V)
u_.name = "u"
u_s = Function(V)
u_n = Function(V)
u_n1 = Function(V)
p = TrialFunction(Q)
q = TestFunction(Q)
p_ = Function(Q)
p_.name = "p"
phi = Function(Q)

# first step
f = Constant(mesh, PETSc.ScalarType((0, 0)))
F1 = rho / k * dot(u - u_n, v) * dx
F1 += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
F1 += 0.5 * mu * inner(grad(u + u_n), grad(v)) * dx - dot(p_, div(v)) * dx
F1 += dot(f, v) * dx
a1 = form(lhs(F1))
L1 = form(rhs(F1))
A1 = create_matrix(a1)
b1 = create_vector(L1)

# second step
a2 = form(dot(grad(p), grad(q)) * dx)
L2 = form(-rho / k * dot(div(u_s), q) * dx)
A2 = assemble_matrix(a2, bcs=bcp)
A2.assemble()
b2 = create_vector(L2)

# last step 
a3 = form(rho * dot(u, v) * dx)
L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx)
A3 = assemble_matrix(a3)
A3.assemble()
b3 = create_vector(L3)

# mesh movement
a4 = form(inner(grad(u_mesh), grad(v_mesh)) * dx)
zero_vec = Constant(mesh, PETSc.ScalarType((0.0, 0.0)))
L4 = form(inner(zero_vec, v_mesh) * dx)
A4 = assemble_matrix(a4, bcs=bcx)
A4.assemble()
b4 = create_vector(L4)

# setup solvers
solver1 = PETSc.KSP().create(mesh.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.JACOBI)

solver2 = PETSc.KSP().create(mesh.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.MINRES)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

solver3 = PETSc.KSP().create(mesh.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)

solver4 = PETSc.KSP().create(mesh.comm)
solver4.setOperators(A4)
solver4.setType(PETSc.KSP.Type.CG)
pc4 = solver4.getPC()
pc4.setType(PETSc.PC.Type.JACOBI)

# compute drag and lift coefficients
n = -FacetNormal(mesh)  # Normal pointing out of obstacle
dObs = Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=edge_marker)
u_t = inner(as_vector((n[1], -n[0])), u_)
drag = form(2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[1] - p_ * n[0]) * dObs)
lift = form(-2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[0] + p_ * n[1]) * dObs)
if mesh.comm.rank == 0:
    C_D = np.zeros(num_steps, dtype=PETSc.ScalarType)
    C_L = np.zeros(num_steps, dtype=PETSc.ScalarType)
    t_u = np.zeros(num_steps, dtype=np.float64)
    t_p = np.zeros(num_steps, dtype=np.float64)


####################################################
#                                                  #
#           SOLVE TIME-DEPENDENT PROBLEM           #
#                                                  #
####################################################

# output folders
folder = Path("results")
folder.mkdir(exist_ok=True, parents=True)
vtx_u = VTXWriter(mesh.comm, "results/dfg2D-3-u.bp", [u_], engine="BP4")
vtx_p = VTXWriter(mesh.comm, "results/dfg2D-3-p.bp", [p_], engine="BP4")
xdmf_m = XDMFFile(MPI.COMM_WORLD, f"results/displacement.xdmf", "w")
xdmf_m.write_mesh(mesh)
xdmf_m.write_function(total_mesh_displacement, t)
xdmf_u = XDMFFile(MPI.COMM_WORLD, f"results/velocity.xdmf", "w")
xdmf_u.write_mesh(mesh)
xdmf_u.write_function(u_, t)
vtx_u.write(t)
vtx_p.write(t)

# time step loop
progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=num_steps)
for i in range(num_steps):
    progress.update(1)
    # Update current time step
    t += dt
    # Update inlet velocity
    inlet_velocity.t = t
    u_inlet.interpolate(inlet_velocity)
    # Update mesh perturbation
    mesh_delta.t = t
    mesh_displacement_bc.interpolate(mesh_delta)

    # Step 1: Tentative velocity step
    A1.zeroEntries()
    assemble_matrix(A1, a1, bcs=bcu)
    A1.assemble()
    with b1.localForm() as loc:
        loc.set(0)
    assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u_s.vector)
    u_s.x.scatter_forward()

    # Step 2: Pressure corrrection step
    with b2.localForm() as loc:
        loc.set(0)
    assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, phi.vector)
    phi.x.scatter_forward()

    p_.vector.axpy(1, phi.vector)
    p_.x.scatter_forward()

    # Step 3: Velocity correction step
    with b3.localForm() as loc:
        loc.set(0)
    assemble_vector(b3, L3)
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, u_.vector)
    u_.x.scatter_forward()

    # Step 4: Solve for mesh movement
    A4.zeroEntries()
    assemble_matrix(A4, a4, bcs=bcx)
    A4.assemble()
    with b4.localForm() as loc:
        loc.set(0)
    assemble_vector(b4, L4)
    apply_lifting(b4, [a4], [bcx])
    b4.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b4, bcx)
    solver4.solve(b4, mesh_displacement.vector)
    mesh_displacement.x.scatter_forward()

    # Move mesh
    with mesh_displacement.vector.localForm() as vals_local:
        vals = vals_local.array
        vals = vals.reshape(-1, 2)
    mesh.geometry.x[:, :2] += vals[:, :]

    # Track total mesh movement for visualization
    total_mesh_displacement.vector.array[:] += mesh_displacement.vector.array
    total_mesh_displacement.x.scatter_forward()

    # Write solutions to file
    vtx_u.write(t)
    vtx_p.write(t)
    xdmf_m.write_function(total_mesh_displacement, t)
    xdmf_u.write_function(u_, t)
   
    # Update variable with solution form this time step
    with u_.vector.localForm() as loc_, u_n.vector.localForm() as loc_n, u_n1.vector.localForm() as loc_n1:
        loc_n.copy(loc_n1)
        loc_.copy(loc_n)

    # Compute physical quantities (just lift and drag now)
    # For this to work in paralell, we gather contributions from all processors
    # to processor zero and sum the contributions.
    drag_coeff = mesh.comm.gather(assemble_scalar(drag), root=0)
    lift_coeff = mesh.comm.gather(assemble_scalar(lift), root=0)
    if mesh.comm.rank == 0:
        t_u[i] = t
        t_p[i] = t - dt / 2
        C_D[i] = sum(drag_coeff)
        C_L[i] = sum(lift_coeff)

# close output folders
vtx_u.close()
vtx_p.close()
xdmf_m.close()
xdmf_u.close()
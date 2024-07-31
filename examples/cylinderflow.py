import numpy as np
import tqdm.autonotebook
from pathlib import Path
from random import random

import gmsh
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element
from ufl import (
    FacetNormal,
    Measure,
    TestFunction,
    TrialFunction,
    as_vector,
    div,
    dot,
    dx,
    inner,
    lhs,
    grad,
    nabla_grad,
    rhs,
    Identity,
)

from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_topological,
    set_bc,
)
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_vector,
    create_matrix,
    set_bc,
)
from dolfinx.io import VTXWriter, gmshio, XDMFFile
from dolfinx.mesh import locate_entities_boundary


####################################################
#                                                  #
#           MESH PARAMETERS                        #
#                                                  #
####################################################

d = 1
L = 32.5 * d
H = 20 * d
c_x = 12.5 * d
c_y = 10 * d
r = 0.5 * d
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
    assert len(volumes) == 2
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
        elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(
            center_of_mass, [L / 2, 0, 0]
        ):
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

strouhal_period = 1 / 0.1727

t = 0
T_noise = 5  # time that arbitrary noise ends
T1 = 20 * strouhal_period
T2 = T1  # Final time
dt1 = strouhal_period / 150  # Time step size
dt2 = strouhal_period / 150 / 4
dt = dt1
save_interval = 0.5

Re = 100
disk_freq = 1.1 / strouhal_period  # Hz
disk_ampl = 0.25


####################################################
#                                                  #
#           SET UP PROBLEM                         #
#                                                  #
####################################################

num_steps = int(T1 / dt1) + int((T2 - T1) / dt2)
save_every_n = int(save_interval / dt)
k = Constant(mesh, PETSc.ScalarType(dt))
mu = Constant(mesh, PETSc.ScalarType(1 / Re))  # Dynamic viscosity
rho = Constant(mesh, PETSc.ScalarType(1))  # Density
u_inf = 1
ampl_noise = 0.01


class InletVelocity:
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        if self.t < T_noise:

            def noise():
                return random() * 2 - 1  # between -1 and 1, noninclusive

            velocities = [u_inf + ampl_noise * noise() for _ in range(len(values[0]))]
            values[0] = velocities
        else:
            values[0] = u_inf
        return values


class DiskVelocity:
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[1] = self.y_shift()
        return values

    def y_shift(self):
        return (
            disk_ampl * np.cos(self.t * 2 * np.pi * disk_freq) * 2 * np.pi * disk_freq
        )


def _all_interior_surfaces(x):
    eps = 1.0e-5
    x_min = 0
    x_max = L
    y_min = 0
    y_max = H

    x_mid = np.logical_and(x_min + eps < x[0], x[0] < x_max - eps)
    y_mid = np.logical_and(y_min + eps < x[1], x[1] < y_max - eps)
    return np.logical_and(x_mid, y_mid)


def _all_exterior_surfaces(x):
    eps = 1.0e-5
    x_min = 0
    x_max = L
    y_min = 0
    y_max = H

    x_edge = np.logical_or(x[0] < x_min + eps, x_max - eps < x[0])
    y_edge = np.logical_or(x[1] < y_min + eps, y_max - eps < x[1])
    return np.logical_or(x_edge, y_edge)


# boundary conditions
v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,))
v_cg_mesh = element(
    "Lagrange", mesh.topology.cell_name(), mesh_order, shape=(mesh.geometry.dim,)
)
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
V_mesh = functionspace(mesh, v_cg_mesh)
Q = functionspace(mesh, s_cg1)

fdim = mesh.topology.dim - 1

# get interior points
facet_dim = 1
all_interior_facets = locate_entities_boundary(mesh, facet_dim, _all_interior_surfaces)
all_interior_V_mesh_dofs = locate_dofs_topological(
    V_mesh, facet_dim, all_interior_facets
)
all_exterior_facets = locate_entities_boundary(mesh, facet_dim, _all_exterior_surfaces)
all_exterior_V_mesh_dofs = locate_dofs_topological(
    V_mesh, facet_dim, all_exterior_facets
)

# Mesh
mesh_speed = DiskVelocity(t)
mesh_vel_bc = Function(V_mesh)
mesh_vel_bc.interpolate(mesh_speed)
bcx_in = dirichletbc(mesh_vel_bc, all_interior_V_mesh_dofs)
zero_vec = Constant(mesh, PETSc.ScalarType((0.0, 0.0)))
bcx_out = dirichletbc(zero_vec, all_exterior_V_mesh_dofs, V_mesh)
bcx = [bcx_in, bcx_out]
# Inlet
u_inlet = Function(V)
inlet_velocity = InletVelocity(t)
u_inlet.interpolate(inlet_velocity)
dofs_inflow = locate_dofs_topological(V, fdim, ft.find(inlet_marker))
bcu_inflow = dirichletbc(u_inlet, dofs_inflow)
# Walls
dofs_walls = locate_dofs_topological(V.sub(1), fdim, ft.find(wall_marker))
bcu_walls = dirichletbc(PETSc.ScalarType(0), dofs_walls, V.sub(1))
# Obstacle
nonslip_bc = Function(V)
nonslip_bc.interpolate(mesh_speed)
dofs_obstacle = locate_dofs_topological(V, fdim, ft.find(edge_marker))
bcu_obstacle = dirichletbc(nonslip_bc, dofs_obstacle)
bcu = [bcu_inflow, bcu_obstacle, bcu_walls]
# Outlet
dofs_outlet = locate_dofs_topological(Q, fdim, ft.find(outlet_marker))
bcp_outlet = dirichletbc(PETSc.ScalarType(0), dofs_outlet, Q)
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
# This needs to be on the "V" space since it will affect the
# fluid velocity calculations (not tied to mesh nodes):
mesh_vel = Function(V)
mesh_vel_old = Function(V)
mesh_vel_at_halfstep = 0.5 * (mesh_vel + mesh_vel_old)

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
F1 += (
    inner(
        dot(1.5 * u_n - 0.5 * u_n1 - mesh_vel_at_halfstep, 0.5 * nabla_grad(u + u_n)), v
    )
    * dx
)
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
# u_t = inner(as_vector((n[1], -n[0])), u_)
# drag = form(2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[1] - p_ * n[0]) * dObs)
# lift = form(-2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[0] + p_ * n[1]) * dObs)
if mesh.comm.rank == 0:
    y = np.zeros(num_steps, dtype=PETSc.ScalarType)
    C_D = np.zeros(num_steps, dtype=PETSc.ScalarType)
    C_L = np.zeros(num_steps, dtype=PETSc.ScalarType)
    t_u = np.zeros(num_steps, dtype=np.float64)
    t_p = np.zeros(num_steps, dtype=np.float64)


def sigma(u, p, nu, rho):
    """
    Convenience expression for fluid stress, sigma

    Args:
        u (dolfinx.fem.Function): Velocity
        p (dolfinx.fem.Function): Pressure
        nu (float, dolfinx.fem.Function): Viscosity

    Returns:
        ufl.dolfinx.fem.form: Stress in fluid, $2 nu epsilon (u)$
    """
    return (mu) * (grad(u) + grad(u).T) - p * Identity(len(u))
    # return 2 * nu * rho * epsilon(u) - p * ufl.Identity(len(u))


stress = sigma(u_, p_, mu, rho)

facet_normal = FacetNormal(mesh)

# Compute traction vector
traction = dot(stress, -facet_normal)

lift = form(traction[1] * dObs)
drag = form(traction[0] * dObs)

####################################################
#                                                  #
#           SOLVE TIME-DEPENDENT PROBLEM           #
#                                                  #
####################################################

# specifiy output folder
output_folder = "results"

# output folders
folder = Path(output_folder)
folder.mkdir(exist_ok=True, parents=True)
vtx_u = VTXWriter(mesh.comm, f"{output_folder}/dfg2D-3-u.bp", [u_], engine="BP4")
vtx_p = VTXWriter(mesh.comm, f"{output_folder}/dfg2D-3-p.bp", [p_], engine="BP4")
xdmf_m = XDMFFile(MPI.COMM_WORLD, f"{output_folder}/displacement.xdmf", "w")
xdmf_m.write_mesh(mesh)
xdmf_m.write_function(total_mesh_displacement, t)
xdmf_u = XDMFFile(MPI.COMM_WORLD, f"{output_folder}/velocity.xdmf", "w")
xdmf_u.write_mesh(mesh)
xdmf_u.write_function(u_, t)
vtx_u.write(t)
vtx_p.write(t)
y_cylinder_displacement = 0
# time step loop
progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=num_steps)
for i in range(num_steps):
    progress.update(1)
    # Update current time step
    if t >= T1:
        dt = dt2
        k.value = dt
    t += dt
    # Update inlet velocity
    inlet_velocity.t = t
    u_inlet.interpolate(inlet_velocity)
    # Update mesh perturbation
    if disk_ampl > 0.0:
        mesh_speed.t = t
        mesh_vel_bc.interpolate(mesh_speed)
        nonslip_bc.interpolate(mesh_speed)

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
    A2.zeroEntries()
    assemble_matrix(A2, a2, bcs=bcp)
    A2.assemble()
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
    A3.zeroEntries()
    assemble_matrix(A3, a3)
    A3.assemble()
    with b3.localForm() as loc:
        loc.set(0)
    assemble_vector(b3, L3)
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, u_.vector)
    u_.x.scatter_forward()

    # Step 4: Solve for mesh movement
    if disk_ampl > 0.0:
        # Update old mesh velocity to store the last timestep's mesh velocity
        mesh_vel_old.x.array[:] = mesh_vel.x.array[:]

        A4.zeroEntries()
        assemble_matrix(A4, a4, bcs=bcx)
        A4.assemble()
        with b4.localForm() as loc:
            loc.set(0)
        assemble_vector(b4, L4)
        apply_lifting(b4, [a4], [bcx])
        b4.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b4, bcx)
        solver4.solve(b4, mesh_vel.vector)
        mesh_vel.x.scatter_forward()

        mesh_displacement.interpolate(mesh_vel)
        mesh_displacement.x.array[:] *= dt
        y_cylinder_displacement += mesh_speed.y_shift() * dt

        # Move mesh
        with mesh_displacement.vector.localForm() as vals_local:
            vals = vals_local.array
            vals = vals.reshape(-1, 2)
        mesh.geometry.x[:, :2] += vals[:, :]

        # Track total mesh movement for visualization
        total_mesh_displacement.vector.array[:] += mesh_displacement.vector.array
        total_mesh_displacement.x.scatter_forward()

    # Write solutions to file
    if (i + 1) % save_every_n == 0:
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
        C_D[i] = sum(drag_coeff) / (0.5 * rho.value * u_inf**2 * d)
        C_L[i] = sum(lift_coeff) / (0.5 * rho.value * u_inf**2 * d)
        y[i] = y_cylinder_displacement

if mesh.comm.rank == 0:
    np.savetxt(
        f"{output_folder}/drag_over_time.csv",
        np.vstack((t_u, C_D)).T,
        delimiter=",",
        header="time,drag",
    )
    np.savetxt(
        f"{output_folder}/lift_over_time.csv",
        np.vstack((t_u, C_L)).T,
        delimiter=",",
        header="time,lift",
    )
    np.savetxt(
        f"{output_folder}/y_over_time.csv",
        np.vstack((t_u, y)).T,
        delimiter=",",
        header="time,y",
    )

# close output folders
vtx_u.close()
vtx_p.close()
xdmf_m.close()
xdmf_u.close()

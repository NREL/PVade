"""
Adding temperature effects for air
- by Brooke Stanislawski, Ethan Young, and Walid Arsalane
- via buoyancy term in momentum equation
- and adding a fourth solve for the advection-diffusion equation for temperature
"""

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

import gmsh
from dolfinx import io
from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    create_vector,
    set_bc,
)

# from dolfinx.io import VTXWriter
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary, locate_entities

# from dolfinx.plot import vtk_mesh
from ufl import (
    FacetNormal,
    FiniteElement,
    Identity,
    TestFunction,
    TrialFunction,
    VectorElement,
    div,
    dot,
    ds,
    dx,
    inner,
    lhs,
    nabla_grad,
    grad,
    rhs,
    sym,
    sqrt,
    CellDiameter,
)

# ================================================================
# Inputs
# ================================================================

x_min = 0.0
x_max = 1.2 # 0.4 #1.0

y_min = 0.0
y_max = 0.4 #1.0 #3.0

# h = 0.05
nx = 120 # 50 # 100 # 100 # 50  # 150 # int((x_max - x_min)/h)
ny = 40 # 50 # 100 # 150 #.5*.02/.1 30 # 10 # 50  # 50 # int((y_max - y_min)/h)

# flow over a flat plate
T_ambient = 300.0
T0_top_wall = T_ambient
T0_bottom_wall = T_ambient + 20.0
T0_pv_panel = T_ambient + 40.0 # only used if pv_panel_flag == True

#neutral
# T0_bottom_wall = T_ambient
# T0_pv_panel = T_ambient

t_bc_flag = 'stepchange'
# t_bc_flag = 'rampdown'

# # uniform inflow
# inflow = 'uniform'
# u0 = 0.0 # 1.0

# loglaw inflow
inflow = 'loglaw'
u_hub = 0.5 #1.0
z_hub = 0.12
z0 = 0.005
d0 = 0.0 # 0.65*z_hub

# stabilizing = False
# save_fn = 'temp_panel'
stabilizing = True
# save_fn = 'temp_panel_stab'
save_fn = 'temp_panel_neutral'

pv_panel_flag = True  # empty domain or with a pv panel in the center

t_final = 2.0 # 1.0 # 10.0 # 20.0 # 120.0
dt_num = 0.01 # 0.01 #0.001

# ================================================================
# Build Mesh
# ================================================================

if pv_panel_flag:
    # pass

    comm = MPI.COMM_WORLD

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    gmsh_model = gmsh.model()
    gmsh_model.add("domain")
    gmsh_model.setCurrent("domain")

    ndim = 2

    domain_width = x_max - x_min #3.0  # box width
    domain_height = y_max - y_min # 1.0  #  # box height

    domain_id = gmsh_model.occ.addRectangle(
        0, 0, 0, domain_width, domain_height
    )  # Notice this spans from [0, x_max], [0, y_max], your BCs may need adjustment
    domain_tag = (ndim, domain_id)

    panel_width = 0.1 # 0.5  # Chord length, or width
    panel_height = 0.03 # 0.05  # Sets the panel thickness, really
    panel_angle = np.radians(
        30
    )  # Sets the panel rotation (argument must be radians for gmsh)

    panel_id = gmsh_model.occ.addRectangle(
        -0.5 * panel_width, -0.5 * panel_height, 0, panel_width, panel_height
    )
    panel_tag = (ndim, panel_id)

    # Rotate the panel and shift it into its correct position
    gmsh_model.occ.rotate([panel_tag], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, panel_angle)
    gmsh_model.occ.translate([panel_tag], 0.3 * domain_width, 0.3 * domain_height, 0.0)

    # Cookie cutter step, domain = domain - panel, is how to read this
    gmsh_model.occ.cut([domain_tag], [panel_tag])

    gmsh_model.occ.synchronize()

    all_pts = gmsh_model.occ.getEntities(0)

    l_characteristic = 1.0/nx # 0.05  # Sets the characteristic size of the cells
    gmsh_model.mesh.setSize(all_pts, l_characteristic)

    vol_tag_list = gmsh_model.occ.getEntities(ndim)

    for vol_tag in vol_tag_list:
        vol_id = vol_tag[1]
        gmsh_model.add_physical_group(ndim, [vol_id], vol_id)

    # Generate the mesh
    gmsh_model.mesh.generate(ndim)

    mesh, mt, ft = io.gmshio.model_to_mesh(gmsh_model, comm, 0, gdim=2)

else:
    # create an empty domain mesh
    mesh = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([x_min, y_min]), np.array([x_max, y_max])],
        [nx, ny],
        CellType.triangle,
    )


# ================================================================
# Define Constants
# ================================================================

# calc alpha from Incropera for air at 300 K
g_f = 9.81
beta_f = 1/300.0 # [1/K]
nu_f = 15.89e-6 # 0.01 # 15.89e-6 # 0.01 # [m2/s] kinematic viscosity
nu_f *= 10
alpha_f = 22.5/10**6 # m2/s
alpha_f *= 1 # high Pe (stab needed)
# alpha_f *= 10 # moderate Pe
# alpha_f *= 20 # threshold: when alpha factor < 20, ringing appears
# alpha_f = 0.0020833333333333333 # Pe = 1
# alpha_f *= 100 # low Pe (stab not needed)
rho_f = 1.1314 # kg/m3
# cp_f = 1.004*1000 # J/kg*K
mu_f = nu_f * rho_f # dynamic viscosity
# k_f = 0.0263 # W/m*K
# # alpha_f = k_f/(rho_f*cp_f) # m2/s

# from https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code1.html
# g_f = -9.81
# beta_f = 0.01
# alpha_f = 0.01 #22.5/10**6 # m2/s
# rho_f = 1.0 # kg/m3
# mu_f = 0.01

Ra = (g_f*beta_f/(nu_f*alpha_f))*(T0_bottom_wall-T0_top_wall)*(y_max-y_min)
Pe_approx = u_hub * l_characteristic / (2.0 * alpha_f)
Re_approx = u_hub * panel_width / nu_f

print('g = {}'.format(g_f))
print('alpha = {:.2E}'.format(alpha_f))
print('mu = {:.2E}'.format(mu_f))
print('Ra = {:.2E}'.format(Ra))
print('Pe approx = {:.2E}'.format(Pe_approx))
print('Re approx = {:.2E}'.format(Re_approx))
# exit()

g = Constant(mesh, PETSc.ScalarType((0, g_f))) # negative? YES
beta = Constant(mesh, PETSc.ScalarType(beta_f)) # [1/K] thermal expansion coefficient
alpha = Constant(mesh, PETSc.ScalarType(alpha_f)) # thermal diffusivity [m2/s]
rho = Constant(mesh, PETSc.ScalarType(rho_f)) # density [kg/m3]
mu = Constant(mesh, PETSc.ScalarType(mu_f)) # dynamic viscosity [Ns/m2] # Re = 100
nu = Constant(mesh, PETSc.ScalarType(nu_f)) # dynamic viscosity [Ns/m2] # Re = 100

dt = Constant(mesh, PETSc.ScalarType(dt_num))

# ================================================================
# Build Function Spaces and Functions
# ================================================================

v_cg2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
q_cg1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)

V = FunctionSpace(mesh, v_cg2)  # velocity
Q = FunctionSpace(mesh, q_cg1)  # pressure
S = FunctionSpace(mesh, q_cg1)  # temperature

# velocity
u = TrialFunction(V)
v = TestFunction(V)
u_n = Function(V)
u_n.name = "u_n"
u_ = Function(V)

# pressure
p = TrialFunction(Q)
q = TestFunction(Q)
p_n = Function(Q)
p_n.name = "p_n"
p_ = Function(Q)

# temperature
theta = TrialFunction(S)
s = TestFunction(S)
T_n = Function(S)  # for outputting T, calculated from theta for each timestep
T_n.name = "T_n"
T_ = Function(S)

# %% ================================================================
# Build Boundary Conditions
# ================================================================
class InletVelocity():
    # copied from https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
    def __init__(self):
        pass

    def __call__(self, x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
        # print('x.shape vel = ', x.shape)
        if inflow == 'uniform':
            values[0] = u0

        elif inflow == 'loglaw':
            # values[0] = ((u_hub) * np.log(((x[1]) - d0) / z0) / (np.log((z_hub - d0) / z0)))
            values[0] = ((u_hub) * np.log(((x[1]) - d0) / z0) / (np.log((z_hub - d0) / z0)))
        return values
    
class LowerWallTemperature():
    # copied from https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
    def __init__(self):
        pass

    def __call__(self, x):
        values = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
        x0 = 0.75 * x_max # start of ramp down
        values[0] = (T0_bottom_wall + ((x[0]-x0) / x_max) * (T_ambient - T0_bottom_wall))
        return values

def left_wall(x):
    return np.isclose(x[0], x_min)

def right_wall(x):
    return np.isclose(x[0], x_max)

def bottom_wall(x):
    return np.isclose(x[1], y_min)

def top_wall(x):
    return np.isclose(x[1], y_max)

def bottom_left_corner(x):
    return np.logical_and(np.isclose(x[1], y_min), np.isclose(x[0], x_min))

def upper_right_corner(x):
    return np.logical_and(np.isclose(x[1], y_max), np.isclose(x[0], x_max))

def internal_boundaries(x):
    tol = 1e-3
    x_test = np.logical_and(x_min + tol < x[0], x[0] < x_max - tol)
    y_test = np.logical_and(y_min + tol < x[1], x[1] < y_max - tol)
    return np.logical_and(x_test, y_test)

# Velocity Boundary Conditions
# Inlet
if inflow == 'loglaw':
    upper_cells = locate_entities(mesh, mesh.geometry.dim, lambda x: x[1] > d0 + z0)
    u_inlet = Function(V)
    u_inlet.interpolate(lambda x: np.zeros((mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType))
    left_wall_dofs = locate_dofs_geometrical(V, left_wall)
    inlet_velocity = InletVelocity()
    u_inlet.interpolate(inlet_velocity, upper_cells)

else:
    u_inlet = Function(V)
    u_inlet.interpolate(lambda x: np.zeros((mesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType))
    left_wall_dofs = locate_dofs_geometrical(V, left_wall)
    inlet_velocity = InletVelocity()
    u_inlet.interpolate(inlet_velocity)

bcu_inflow = dirichletbc(u_inlet, left_wall_dofs)

u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
# u_inflow = np.array((1,0), dtype=PETSc.ScalarType) # ux, uy = 1, 0

bottom_wall_dofs = locate_dofs_geometrical(V, bottom_wall)
bcu_bottom_wall = dirichletbc(u_noslip, bottom_wall_dofs, V)

# slip at top wall
top_wall_entities = locate_entities_boundary(mesh, mesh.geometry.dim-1, top_wall)
top_wall_dofs = locate_dofs_topological(V.sub(1), mesh.geometry.dim-1, top_wall_entities)
zero_scalar = Constant(mesh, PETSc.ScalarType(0.0))
bcu_top_wall = dirichletbc(zero_scalar, top_wall_dofs, V.sub(1))

bcu = [bcu_inflow, bcu_bottom_wall, bcu_top_wall]

if pv_panel_flag:
    boundary_facets = locate_entities_boundary(
        mesh, mesh.topology.dim - 1, internal_boundaries
    )
    boundary_dofs = locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bcu_internal_walls = dirichletbc(u_noslip, boundary_dofs, V)

    bcu.append(bcu_internal_walls)

u_n.interpolate(u_inlet)
# set_bc(u_n.vector,bcu)

# Pressure Boundary Conditions
right_wall_entities = locate_entities_boundary(mesh, mesh.geometry.dim-1, right_wall)
right_wall_dofs = locate_dofs_topological(Q, mesh.geometry.dim-1, right_wall_entities)
zero_scalar = Constant(mesh, PETSc.ScalarType(0.0))
bcp_outlet = dirichletbc(zero_scalar, right_wall_dofs, Q)
bcp = [bcp_outlet]

set_bc(p_n.vector,bcp)

# Temperature Boundary Conditions
T_r = Constant(mesh, PETSc.ScalarType(T_ambient)) # reference temperature

# Interpolate initial temperature vertically for a smooth gradient
# T_n.interpolate(lambda x: (T0_bottom_wall + (x[1] / y_max) * (T0_top_wall - T0_bottom_wall)))

# Initialize constant fluid temperature everywhere in domain
T_n.x.array[:] = PETSc.ScalarType(T_ambient)

# nonuniform temperature bc along bottom wall
if t_bc_flag == 'rampdown':
    rampdown_cells = locate_entities(mesh, mesh.geometry.dim, lambda x: x[0] > (0.75*x_max))
    T_bottom = Function(S)
    T_bottom.interpolate(lambda x: np.full((1, x.shape[1]), T0_bottom_wall, dtype=PETSc.ScalarType)) # how do I initialize as a constant?
    bottom_wall_dofs = locate_dofs_geometrical(S, bottom_wall)
    bottom_wall_temperature = LowerWallTemperature()
    T_bottom.interpolate(bottom_wall_temperature, rampdown_cells)


elif t_bc_flag == 'stepchange':
    heated_cells = locate_entities(mesh, mesh.geometry.dim, lambda x: x[0] < (0.75*x_max))
    T_bottom = Function(S)
    T_bottom.interpolate(lambda x: np.full((1, x.shape[1]), T_ambient, dtype=PETSc.ScalarType)) # how do I initialize as a constant?
    bottom_wall_dofs = locate_dofs_geometrical(S, bottom_wall)
    bottom_wall_temperature = LowerWallTemperature()
    T_bottom.interpolate(bottom_wall_temperature, heated_cells)

bcT_bottom_wall = dirichletbc(T_bottom, bottom_wall_dofs)

left_wall_dofs = locate_dofs_geometrical(S, left_wall)
bcT_left_wall = dirichletbc(PETSc.ScalarType(T_ambient), left_wall_dofs, S)

bcT = [bcT_left_wall, bcT_bottom_wall]
# exit()

if pv_panel_flag:

    print("applying pv panel temp = {}".format(T0_pv_panel))
    boundary_facets = locate_entities_boundary(
        mesh, mesh.topology.dim - 1, internal_boundaries
    )
    boundary_dofs = locate_dofs_topological(S, mesh.topology.dim - 1, boundary_facets)
    bcT_internal_walls = dirichletbc(
        PETSc.ScalarType(T0_pv_panel), boundary_dofs, S
    )

    bcT.append(bcT_internal_walls)


# ================================================================
# Build All Forms
# ==================================================================

# step 1: tentative velocity

# Crank-Nicolson velocity
U_CN = 0.5 * (u + u_)

# Adams-Bashforth velocity
U_AB = 1.5 * u_ - 0.5 * u_n

use_pressure_in_F1 = True

# using nu (pvade form)
F1 = (1.0 / dt) * inner(u - u_n, v) * dx
F1 += inner(dot(U_AB, nabla_grad(U_CN)), v) * dx # convection
F1 += nu * inner(grad(U_CN), grad(v)) * dx # viscosity # + or - ??
F1 -= beta * inner((T_n-T_r) * g, v) * dx # buoyancy # THIS ONE WITH POSITIVE G INPUT PARAMETER
if use_pressure_in_F1:
    F1 += (1.0 / rho) *inner(grad(p_), v) * dx

# using rho and mu
# F1 = (rho / dt) * inner(u - u_n, v) * dx
# F1 += rho * inner(dot(U_AB, nabla_grad(U_CN)), v) * dx # convection
# F1 += mu * inner(grad(U_CN), grad(v)) * dx # viscosity # + or - ??
# F1 -= rho * beta * inner((T_n-T_r) * g, v) * dx # buoyancy # THIS ONE WITH POSITIVE G INPUT PARAMETER
# if use_pressure_in_F1:
#     F1 += inner(grad(p_), v) * dx

a1 = form(lhs(F1))  # dependent on u
L1 = form(rhs(F1))

# step 2: pressure correction
a2 = form(inner(nabla_grad(p), nabla_grad(q)) * dx)
if use_pressure_in_F1:
    L2 = form(dot(nabla_grad(p_), nabla_grad(q))*dx - (rho / dt) * div(u_) * q * dx)  # needs to be reassembled
else:
    L2 = form( - (rho / dt) * div(u_) * q * dx)  # needs to be reassembled

# step 3: velocity update
a3 = form(inner(u, v) * dx)  # doesn't need to be reassembled
if use_pressure_in_F1:
    L3 = form(inner(u_, v) * dx - (dt/rho) * inner(grad(p_ - p_n), v) * dx) # u_ is known
else:
    L3 = form(inner(u_, v) * dx - (dt/rho) * inner(grad(p_), v) * dx) # u_ is known

if stabilizing:
    # Residual, the "strong" form of the governing equation
    r = (1 / dt)*(theta - T_n) + dot(u_, nabla_grad(theta)) - alpha*div(grad(theta)) # this is the one

# how to print these terms?
F4 = (1 / dt) * inner(theta - T_n, s) * dx # theta = unknown, T_n = temp from previous timestep
F4 += alpha * inner(nabla_grad(theta), nabla_grad(s)) * dx
F4 += inner(dot(u_, nabla_grad(theta)), s) * dx

if stabilizing:
    eps = 1e-5
    # Add SUPG stabilisation terms 
    # https://fenicsproject.org/qa/13458/how-implement-supg-properly-advection-dominated-equation/
    # vnorm = sqrt(dot(u_, u_)) # magnitude of u
    # h = CellDiameter(mesh)
    # # delta = h/(2.0*vnorm)
    # # delta = h/(vnorm) # "when diffusion is not important"
    # delta = h/(vnorm + sqrt(2*alpha/h)) # "when diffusion is important (alpha > 1e-5)" THIS SEEMS TO WORK FOR BOTH LOW AND HIGH PE
    # # if Pe_approx > 1.0: # might need to adjust this threshold value
    # #     delta = h/(vnorm + sqrt(2*alpha/h)) # "when diffusion is important (alpha > 1e-5)"
    # # else:
    # #     delta = h/(vnorm) # "when diffusion is not important"
    # # delta = (0.5*h)/(2.0*vnorm+eps)
    # # delta = h/(2.0*sqrt(vnorm))
    # # delta = l_characteristic/(2.0*vnorm)
    # stab = delta * dot(u_, grad(s)) * r * dx
    # # stab = 0.0

    # coeff as a function of Peclet number
    # u_mag = sqrt(dot(u_, u_)) # magnitude of vector
    # h = CellDiameter(mesh)
    # Pe = u_mag * h / (2.0 * alpha)
    # # Pe = vnorm * h / (2.0 * alpha) # BJS check if this should be vnorm (1 of 2) - yes, u_mag
    # beta_coeff = calc_beta_coeff(Pe) # goes to zero as Pe goes to zero
    # # tau = beta_coeff * h / (2.0 * u_mag) # works, but some hot/cold spots
    # tau = beta_coeff * h / (2.0 * sqrt(u_mag)) # works, but still a little ringing
    # # tau = beta_coeff * h / (2.0 * vnorm/2) # does not work
    # stab = tau * dot(u_, grad(s)) * r * dx

    # Donea and Huerta 2003 (Eq 2.64)
    h = CellDiameter(mesh)
    u_mag = sqrt(dot(u_, u_)) # magnitude of vector
    # tau = ((2*u_mag/h) + (4*alpha/(h**2)))**(-1)
    Pe = u_mag*h/(2*alpha)
    tau = (h/(2*u_mag))*(1+1/Pe)**(-1)
    stab = tau * dot(u_, grad(s)) * r * dx

    F4 += stab

a4 = form(lhs(F4))  # dependent on u
L4 = form(rhs(F4))

# Solver for step 1
solver1 = PETSc.KSP().create(mesh.comm)
solver1.setType(PETSc.KSP.Type.GMRES)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.JACOBI)

# Solver for step 2
solver2 = PETSc.KSP().create(mesh.comm)
# solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.GMRES)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for step 3
solver3 = PETSc.KSP().create(mesh.comm)
solver3.setType(PETSc.KSP.Type.GMRES)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.JACOBI)

# Solver for step 4
# solver4 = PETSc.KSP().create(mesh.comm)
# solver4.setType(PETSc.KSP.Type.PREONLY)
# pc4 = solver4.getPC()
# pc4.setType(PETSc.PC.Type.LU) # works
solver4 = PETSc.KSP().create(mesh.comm)
solver4.setType(PETSc.KSP.Type.GMRES)
pc4 = solver4.getPC()
pc4.setType(PETSc.PC.Type.LU) # needs LU to run without blowing up for high Pe cases without stabilization

# ================================================================
# Begin Time Iteration
# ================================================================
eps = 3.0e-16
t = dt_num  # dt # 0.0
ct = 1  # 0
save_interval = 1 # 10  # 50

with io.XDMFFile(mesh.comm, save_fn+".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_n, 0)
    xdmf.write_function(p_n, 0)
    xdmf.write_function(T_n, 0)

A1 = assemble_matrix(a1, bcs=bcu)
A1.assemble()
b1 = assemble_vector(L1)

A2 = assemble_matrix(a2, bcs=bcp)
A2.assemble()
b2 = assemble_vector(L2)

A3 = assemble_matrix(a3)
A3.assemble()
b3 = assemble_vector(L3)

A4 = assemble_matrix(a4, bcs=bcT)
A4.assemble()
b4 = assemble_vector(L4)

while t < t_final + eps:
    # ================================================================
    # Assemble and Build Solvers
    # ================================================================

    solver1.setOperators(A1)

    solver2.setOperators(A2)
    solver3.setOperators(A3)
    solver4.setOperators(A4)

    A1.zeroEntries()  # resets the matrix
    A1 = assemble_matrix(A1, a1, bcs=bcu)
    A1.assemble()

    # Step 1: Tentative velocity solve
    with b1.localForm() as loc_1:
        loc_1.set(0)
    b1 = assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u_.vector)
    u_.x.scatter_forward()

    # Step 2: Pressure corrrection step
    with b2.localForm() as loc_2:
        loc_2.set(0)
    b2 = assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, p_.vector)
    p_.x.scatter_forward()

    # Step 3: Velocity correction step
    with b3.localForm() as loc_3:
        loc_3.set(0)
    b3 = assemble_vector(b3, L3)
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, u_.vector)
    u_.x.scatter_forward()

    A4.zeroEntries()
    A4 = assemble_matrix(A4, a4, bcs=bcT)
    A4.assemble()
    
    # Step 4: Temperature corrrection step
    with b4.localForm() as loc_4:
        loc_4.set(0)
    b4 = assemble_vector(b4, L4)
    apply_lifting(b4, [a4], [bcT])
    b4.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b4, bcT)
    solver4.solve(b4, T_.vector)
    T_.x.scatter_forward()

    # Update variable with solution from this time step
    u_n.x.array[:] = u_.x.array[:]
    p_n.x.array[:] = p_.x.array[:]
    T_n.x.array[:] = T_.x.array[:]

    # print(T_n.x.array[:])
    u_n_max = mesh.comm.allreduce(np.amax(u_n.vector.array), op=MPI.MAX)
    p_n_max = mesh.comm.allreduce(np.amax(p_n.vector.array), op=MPI.MAX)
    T_n_max = mesh.comm.allreduce(np.amax(T_n.vector.array), op=MPI.MAX)
    T_n_sum = mesh.comm.allreduce(np.sum(T_n.vector.array), op=MPI.SUM)

    if ct % save_interval == 0:
        with io.XDMFFile(mesh.comm, save_fn+".xdmf", "a") as xdmf:
            xdmf.write_function(u_n, t)
            xdmf.write_function(p_n, t)
            xdmf.write_function(T_n, t)

        if mesh.comm.Get_rank() == 0:
            print(
                "Time = %.6f, u_max = %.6e, p_max = %.6e, T_max = %.6e, T_sum = %.6e"
                % (t, u_n_max, p_n_max, T_n_max, T_n_sum)
            )

    # Move to next step
    t += float(dt)
    ct += 1

print('g = {}'.format(g_f))
print('alpha = {:.2E}'.format(alpha_f))
print('mu = {:.2E}'.format(mu_f))
print('Ra = {:.2E}'.format(Ra))
print('Pe approx = {:.2E}'.format(Pe_approx))
print('Re approx = {:.2E}'.format(Re_approx))
# Rayleigh-Benard Convection Flow
# copied from Walid Arsalene's FEniCS code, adapted to FEniCSx by Brooke Stanislawski

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

from dolfinx import io
from dolfinx.fem import Constant, Function, FunctionSpace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
# from dolfinx.io import VTXWriter
from dolfinx.mesh import create_rectangle, CellType
# from dolfinx.plot import vtk_mesh
from ufl import (FacetNormal, FiniteElement, Identity, TestFunction, TrialFunction, VectorElement,
                 div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym)

# ================================================================
# Build Mesh
# ================================================================

x_min = 0.0
x_max = 3.0

y_min = 0.0
y_max = 1.0

h = 0.05
nx = 50 # 150 # int((x_max - x_min)/h)
ny = 10 # 50 # int((y_max - y_min)/h)

mesh = create_rectangle(MPI.COMM_WORLD, [np.array([x_min, y_min]), np.array([x_max, y_max])],
                               [nx, ny], CellType.triangle)
# Two key physical parameters are the Rayleigh number (Ra), which
# measures the ratio of energy from buoyant forces to viscous
# dissipation and heat conduction and the
# Prandtl number (Pr), which measures the ratio of viscosity to heat
# conduction.

# ================================================================
# Define Constants
# ================================================================

# Ra = Constant(1e8)
Ra = Constant(mesh, PETSc.ScalarType(1e5))
# Ra = Constant(mesh, PETSc.ScalarType(2500))

Pr = Constant(mesh, PETSc.ScalarType(0.7))
# print('Pr = ', Pr.value)

g = Constant(mesh, PETSc.ScalarType((0, 1)))

nu = Constant(mesh, PETSc.ScalarType(1))

# dt = Constant(mesh, PETSc.ScalarType(0.000025))
dt = Constant(mesh, PETSc.ScalarType(0.0001))

# ================================================================
# Build Function Spaces and Functions
# ================================================================
X_PERIODIC = False
# X_PERIODIC = True

# if X_PERIODIC:
#     class XPeriodicBoundary(SubDomain):
#         # Left boundary is "target domain" G
#         def inside(self, x):
#             return np.isclose(x[0], x_min)

#         # Map right boundary (H) to left boundary (G)
#         def map(self, x, y):
#             y[0] = x[0] - (x_max - x_min)
#             y[1] = x[1]

#     v_cg2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
#     V = FunctionSpace(mesh, v_cg2, constrained_domain=XPeriodicBoundary()) # velocity
#     print('applied periodic BCs')
#     # V = VectorFunctionSpace(mesh, 'P', 2, constrained_domain=XPeriodicBoundary())
#     # Q = FunctionSpace(mesh, 'P', 1, constrained_domain=XPeriodicBoundary())
#     # S = FunctionSpace(mesh, 'P', 1, constrained_domain=XPeriodicBoundary())

# else:
v_cg2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
q_cg1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# s_cg1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, v_cg2) # velocity
Q = FunctionSpace(mesh, q_cg1) # pressure
S = FunctionSpace(mesh, q_cg1) # temperature

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
T_n = Function(S) # for outputting T, calculated from theta for each timestep
T_n.name = "T_n"
theta_n = Function(S)
theta_ = Function(S) 

#%% ================================================================
# Build Boundary Conditions
# ================================================================

def left_wall( x):
    return np.isclose(x[0], x_min)

def right_wall( x):
    return np.isclose(x[0], x_max)

def bottom_wall( x):
    return np.isclose(x[1], y_min)

def top_wall( x):
    return np.isclose(x[1], y_max)

# bottom wall should be 0
# class for internal boundaries where all walls are zero
# maybe do square within a square so propoagation is the same in all directions

# Velocity Boundary Conditions
if not X_PERIODIC:
    left_wall_dofs = locate_dofs_geometrical(V, left_wall)
    u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bcu_left_wall = dirichletbc(u_noslip, left_wall_dofs, V)

    right_wall_dofs = locate_dofs_geometrical(V, right_wall)
    bcu_right_wall = dirichletbc(u_noslip, right_wall_dofs, V)

bottom_wall_dofs = locate_dofs_geometrical(V, bottom_wall)
bcu_bottom_wall = dirichletbc(u_noslip, bottom_wall_dofs, V)

top_wall_dofs = locate_dofs_geometrical(V, top_wall)
bcu_top_wall = dirichletbc(u_noslip, top_wall_dofs, V)

bcu = [bcu_left_wall, bcu_right_wall, bcu_bottom_wall, bcu_top_wall]

# Temperature Boundary Conditions
# # visualize temperature variation along wall
# y = np.linspace(y_min, y_max, ny)
# fig, axs = plt.subplots(2)
# axs[0].plot(0.1+0.1*np.sin(2*y),label='funct output')
# axs[1].plot(T_bc.vector.array[:],label='interp')
# plt.legend()
# plt.show()

# temperature boundary conditions
T0_top = 0
T0_bottom = 1
T0_val = 0.5 # 0.5 #10 # TODO - change this to expression
DeltaT = T0_bottom-T0_top # ? should this be defined as Constant(mesh, PETSc.ScalarType(bottom-top)) ?

# Interpolate initial condition
# T0_int = Function(S)
T_n.interpolate(lambda x: (T0_bottom + (x[1]/y_max)*(T0_top-T0_bottom)))

# theta0_int = Function(S)
theta_n.interpolate(lambda x: (-x[1]/y_max))

#initialize T_n
# T_n.x.array[:] = DeltaT*theta_n.x.array[:] + T0_bottom # is this necessary?

print('applying bottom wall temp = {}'.format((T0_bottom-T0_bottom)/DeltaT))
bottom_wall_dofs = locate_dofs_geometrical(S, bottom_wall)
bcT_bottom_wall = dirichletbc(PETSc.ScalarType((T0_bottom-T0_bottom)/DeltaT), bottom_wall_dofs, S)

print('applying top wall temp = {}'.format((T0_top-T0_bottom)/DeltaT))
top_wall_dofs = locate_dofs_geometrical(S, top_wall)
bcT_top_wall = dirichletbc(PETSc.ScalarType((T0_top-T0_bottom)/DeltaT), top_wall_dofs, S)

bcT = [bcT_top_wall, bcT_bottom_wall]
# bcT = [T_bc]

# Pressure Boundary Conditions from fenics code
# pressure_bc = 0
# bcp_bottom_wall = dirichletbc(PETSc.ScalarType(pressure_bc), bottom_wall_dofs, Q)
# bcp_top_wall = dirichletbc(PETSc.ScalarType(pressure_bc), top_wall_dofs, Q)
# bcp_left_wall = dirichletbc(PETSc.ScalarType(pressure_bc), left_wall_dofs, Q)
# bcp_right_wall = dirichletbc(PETSc.ScalarType(pressure_bc), right_wall_dofs, Q)

bcp = [] # [bcp_left_wall, bcp_right_wall, bcp_bottom_wall, bcp_top_wall]

# ================================================================
# Build All Forms
# ==================================================================

# step 1: tentative velocity
# chorin (removed the pressure term)
F1 = (1 / Pr) * ((1 / dt) * inner(u - u_n, v) * dx 
                + inner(nabla_grad(u_n) * u_n, v) * dx) # this might be dot not * ? 
F1 += nu * inner(nabla_grad(u), nabla_grad(v)) * dx 
F1 -= Ra*inner(theta_n*g,v)*dx

a1 = form(lhs(F1)) # dependent on u
L1 = form(rhs(F1))

# step 2: pressure correction
a2 = form(inner(nabla_grad(p), nabla_grad(q))*dx)
L2 = form(-(1.0/dt)*div(u_)*q*dx) # needs to be reassembled

# step 3: velocity update
a3 = form(inner(u, v)*dx) # doesn't need to be reassembled
L3 = form(inner(u_, v)*dx - dt*inner(nabla_grad(p_), v)*dx)

# step 4: temperature update?
a4 = form((1/dt)*inner((theta), s)*dx 
        + inner(nabla_grad(theta), nabla_grad(s))*dx 
        + inner(dot(u_, nabla_grad(theta)), s)*dx) # needs to be reassembled bc of u_
L4 = form((1/dt)*inner(theta_n, s)*dx) # needs to be reassembled bc of theta_n


# [BJS] First Pass - closest alignment to FEniCS RB code as possible (not FEniCSX Nav-Stokes code)
# Solver for step 1
solver1 = PETSc.KSP().create(mesh.comm)
# solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.GMRES) # TODO - test solution with BCGS
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.HYPRE)
pc1.setHYPREType("boomeramg")

# Solver for step 2
solver2 = PETSc.KSP().create(mesh.comm)
# solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.GMRES) # TODO - test solution with BCGS
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
# pc2.setHYPREType("boomeramg") # TODO - test solution with this instead (for speed?)

# Solver for step 3
solver3 = PETSc.KSP().create(mesh.comm)
# solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.GMRES)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.JACOBI) # TODO - test solution with SOR

# Solver for step 2
solver4 = PETSc.KSP().create(mesh.comm)
# solver4.setOperators(A4)
solver4.setType(PETSc.KSP.Type.GMRES)
pc4 = solver4.getPC()
pc4.setType(PETSc.PC.Type.HYPRE)
pc4.setHYPREType("boomeramg")

# ================================================================
# Begin Time Iteration
# ================================================================

eps = 3.0e-16
t = 0.0001 #dt # 0.0
ct = 1 #0
save_interval = 1 #50

t_final = 0.1 #0.5 # 0.5 #0.1 # 0.000075

with io.XDMFFile(mesh.comm, "rayleigh-benard.xdmf", "w") as xdmf:

    xdmf.write_mesh(mesh)
    xdmf.write_function(u_n, 0)
    xdmf.write_function(p_n, 0)
    xdmf.write_function(T_n, 0)
    xdmf.write_function(theta_n, 0)

A1 = assemble_matrix(a1, bcs=bcu)
A1.assemble()
b1 = assemble_vector(L1)

A2 = assemble_matrix(a2, bcs=bcp)
A2.assemble()
b2 = assemble_vector(L2)

A3 = assemble_matrix(a3, bcs=bcu)
A3.assemble()
b3 = assemble_vector(L3)

A4 = assemble_matrix(a4, bcs=bcT)
A4.assemble()
b4 = assemble_vector(L4)

while t < t_final + eps:

    # why is this required?
    T_n.x.array[:] = DeltaT*theta_n.x.array[:] + T0_bottom

    with io.XDMFFile(mesh.comm, "rayleigh-benard.xdmf", "a") as xdmf:

        xdmf.write_function(u_n, t)
        xdmf.write_function(p_n, t)
        xdmf.write_function(T_n, t)
        xdmf.write_function(theta_n, t)

    # ================================================================
    # Assemble and Build Solvers
    # ================================================================

    A1.zeroEntries() # resets the matrix
    A1 = assemble_matrix(A1,a1,bcs=bcu)
    A1.assemble()
    solver1.setOperators(A1)

    # could be removed? ----------
    # A2.zeroEntries()
    # A2 = assemble_matrix(A2,a2, bcs=bcp)
    # A2.assemble()
    solver2.setOperators(A2)

    # A3.zeroEntries()
    # A3 = assemble_matrix(A3, a3, bcs=bcu)
    # A3.assemble()
    solver3.setOperators(A3)
    # b3 = create_vector(L3)
    # could be removed? ----------

    A4.zeroEntries()
    A4 = assemble_matrix(A4, a4, bcs=bcT)
    A4.assemble()
    solver4.setOperators(A4)
    # b4 = create_vector(L4)

    # Step 1: Tentative velocity solve
    with b1.localForm() as loc_1:
        loc_1.set(0)
    b1=assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u_.vector)
    u_.x.scatter_forward()

    # Step 2: Pressure corrrection step
    with b2.localForm() as loc_2:
        loc_2.set(0)
    b2=assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, p_.vector)
    p_.x.scatter_forward()

    # Step 3: Velocity correction step
    with b3.localForm() as loc_3:
        loc_3.set(0)
    b3=assemble_vector(b3, L3)
    apply_lifting(b3, [a3], [bcu])
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    # set_bc(b3, bcu)
    solver3.solve(b3, u_.vector)
    u_.x.scatter_forward()

    # Step 4: Temperature corrrection step
    with b4.localForm() as loc_4:
        loc_4.set(0)
    b4=assemble_vector(b4, L4)
    apply_lifting(b4, [a4], [bcT])
    b4.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b4, bcT)
    solver4.solve(b4, theta_.vector)
    theta_.x.scatter_forward()

    

    # Update variable with solution form this time step
    u_n.x.array[:] = u_.x.array[:]
    p_n.x.array[:] = p_.x.array[:]
    theta_n.x.array[:] = theta_.x.array[:]

    # print(T_n.x.array[:])

    u_n_max = mesh.comm.allreduce(np.max(u_n.vector.array), op=MPI.MAX)
    p_n_max = mesh.comm.allreduce(np.max(p_n.vector.array), op=MPI.MAX)
    T_n_max = mesh.comm.allreduce(np.max(T_n.vector.array), op=MPI.MAX)
    T_n_sum = mesh.comm.allreduce(np.sum(T_n.vector.array), op=MPI.SUM)

    if ct % save_interval == 0:
        print('Time = %.6f, u_max = %.6e, p_max = %.6e, T_max = %.6e, T_sum = %.6e' % (t, u_n_max, p_n_max, T_n_max, T_n_sum))

    # Move to next step
    t += float(dt)
    ct += 1


# visualizing variables
# ================================================================

# print(u_n.vector.array[:])

# plt.scatter(mesh.geometry.x[:,0], mesh.geometry.x[:,1], s=10, c=p_k.vector.array[:])
# plt.scatter(coords[0, :], coords[:,1], s=10, c=p_k.vector.array[:])
# plt.show()

# coords_better = V.tabulate_dof_coordinates()
# print('size1 = ',np.shape(coords_better[0, :]))
# print('size2 = ',np.shape(coords_better[0, :]))
# plt.scatter(coords_better[:, 0], coords_better[:, 1], c=np.sqrt(u_k.vector.array[0::2]**2 + u_k.vector.array[1::2]**2))
# plt.show()
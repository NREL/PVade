#
# ..    # gedit: set fileencoding=utf8 :
#
# .. raw:: html
#
#  <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><p align="center"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png"/></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a></p>
#
# .. _ReissnerMindlinQuads:
#
# ==========================================
# Reissner-Mindlin plate with Quadrilaterals
# ==========================================
#
# -------------
# Introduction
# -------------
#
# This program solves the Reissner-Mindlin plate equations on the unit
# square with uniform transverse loading and fully clamped boundary conditions.
# The corresponding file can be obtained from :download:`reissner_mindlin_quads.py`.
#
# It uses quadrilateral cells and selective reduced integration (SRI) to
# remove shear-locking issues in the thin plate limit. Both linear and
# quadratic interpolation are considered for the transverse deflection
# :math:`w` and rotation :math:`\underline{\theta}`.
#
# .. note:: Note that for a structured square grid such as this example, quadratic
#  quadrangles will not exhibit shear locking because of the strong symmetry (similar
#  to the criss-crossed configuration which does not lock). However, perturbating
#  the mesh coordinates to generate skewed elements suffice to exhibit shear locking.
#
# The solution for :math:`w` in this demo will look as follows:
#
# .. image:: clamped_40x40.png
#    :scale: 40 %
#
#
#
# ---------------
# Implementation
# ---------------
#
#
# Material parameters for isotropic linear elastic behavior are first defined::

from contextlib import ExitStack

import numpy as np
import dolfinx
import ufl
from dolfinx import la, fem 
from dolfinx.fem import (Expression, Function, FunctionSpace,
                         VectorFunctionSpace, dirichletbc, form,
                         locate_dofs_topological)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               set_bc)
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import (CellType, GhostMode, create_box,
                          locate_entities_boundary)
# from ufl import dx, grad, inner

from mpi4py import MPI
from petsc4py import PETSc
import gmsh
import meshio
import os 
# from basix.ufl import mixed_element, element
from dolfinx import fem, io, mesh

dtype = PETSc.ScalarType

N = 50
# Get MPI communicators
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_procs = 0

msh =  dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[0, 0],[10, 1]], [N,N],
                                      CellType.quadrilateral,\
                                            GhostMode.shared_facet)

with io.XDMFFile(msh.comm, "out_mixed_poisson/mesh.xdmf", "w") as file:
    file.write_mesh(msh)

# dolfinx.mesh.create_rectangle(comm,[np.array([0.0, 0.0]),\
#                                            np.array([1.0, 1.0])],N, N, \
#                                            CellType.quadrilateral,\
#                                             GhostMode.shared_facet)

E = fem.Constant(msh,1e3)
nu = fem.Constant(msh,0.3)

# Plate bending stiffness :math:`D=\dfrac{Eh^3}{12(1-\nu^2)}` and shear stiffness :math:`F = \kappa Gh`
# with a shear correction factor :math:`\kappa = 5/6` for a homogeneous plate
# of thickness :math:`h`::

thick = 1e-3
D = E*thick**3/(1-nu**2)/12.
F = E/2/(1+nu)*thick*5./6.

# The uniform loading :math:`f` is scaled by the plate thickness so that the deflection converges to a
# constant value in the thin plate Love-Kirchhoff limit::

# f = fem.Constant(mesh,-thick**3)

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::




# Continuous interpolation using of degree :math:`d=\texttt{deg}` is chosen for both deflection and rotation::

deg = 1
We = ufl.FiniteElement("Lagrange", msh.ufl_cell(), deg)
Te = ufl.VectorElement("Lagrange", msh.ufl_cell(), deg)
V =  fem.FunctionSpace(msh, We*Te )

# Clamped boundary conditions on the lateral boundary are defined as::

def border(x, on_boundary):
    return on_boundary

def boundary_top(x):
    return np.isclose(x[1], 1.0) 
def boundary_bottom(x):
    return np.isclose(x[1], 0.0) 
def boundary_left(x):
    return np.isclose(x[0], 0.0) 
def boundary_right(x):
    return np.isclose(x[0], 10.0) 

def connection_point_up(x):
    return np.isclose(x[1], 0.76) 
def connection_point_down(x):
    return np.isclose(x[1], 0.24) 

msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
msh.topology.create_connectivity(msh.topology.dim-2, msh.topology.dim-1)

fdim = msh.topology.dim - 1
facets_top = dolfinx.mesh.locate_entities_boundary(msh, fdim, boundary_top)
facets_bottom = dolfinx.mesh.locate_entities_boundary(msh, fdim, boundary_bottom)
facets_left = dolfinx.mesh.locate_entities_boundary(msh, fdim, boundary_left)
facets_right = dolfinx.mesh.locate_entities_boundary(msh, fdim, boundary_right)

# Q, _ = V.sub(0).collapse()

facet_uppoint = dolfinx.mesh.locate_entities(msh, 1, connection_point_up)
facet_downpoint = dolfinx.mesh.locate_entities(msh, 1, connection_point_down)

def f2(x):
    values = np.zeros((1, x.shape[1]))
    return values
def f1(x):
    values = np.zeros((2, x.shape[1]))
    return values



R, _ = V.sub(0).collapse()
# dofs_disp = fem.locate_dofs_topological((V.sub(0),R), fdim, [facets_top,facets_bottom,facets_left,facets_right])
dofs_disp = fem.locate_dofs_topological((V.sub(0),R), 1, [facet_uppoint])
f_h2 = fem.Function(R)
f_h2.interpolate(f2)

# x = R.tabulate_dof_coordinates()
# print(x)
# n = R.dim()                                                                      
# d = msh.geometry().dim()  
# dof_coordinates = R.dofmap().tabulate_all_coordinates(msh)                      
# dof_coordinates.resize((n, d))                                                   
# dof_x = dof_coordinates[:, 0]                                                    
# dof_y = dof_coordinates[:, 1]





zero_scalar = dolfinx.fem.Constant(msh, PETSc.ScalarType(0.0))
bc_dis = fem.dirichletbc(f_h2, dofs_disp, V.sub(0))




Q, _ = V.sub(1).collapse()
f_h1 = fem.Function(Q)
f_h1.interpolate(f1)
dofs_rot = fem.locate_dofs_topological((V.sub(1),Q), fdim, [facets_right])
zero_vector = dolfinx.fem.Constant(msh, PETSc.ScalarType((0.0,0.0,0.0)))
bc_rot = fem.dirichletbc(f_h1, dofs_rot, V.sub(1))


bc = [bc_dis,bc_rot]
# Some useful functions for implementing generalized constitutive relations are now
# defined::

def strain2voigt(eps):
    return ufl.as_vector([eps[0, 0], eps[1, 1], 2*eps[0, 1]])
def voigt2stress(S):
    return ufl.as_tensor([[S[0], S[2]], [S[2], S[1]]])
def curv(disp, rot):
    # (w, theta) = fem.split(u)
    return ufl.sym(ufl.grad(rot))
def shear_strain(disp, rot):
    # (w, theta) = fem.split(u)
    return rot-ufl.grad(disp)
def bending_moment(disp, rot):
    DD = ufl.as_tensor([[D, nu*D, 0], [nu*D, D, 0],[0, 0, D*(1-nu)/2.]])
    return voigt2stress(ufl.dot(DD,strain2voigt(curv(disp, rot))))
def shear_force(disp, rot):
    return F*shear_strain(disp, rot)


# The contribution of shear forces to the total energy is under-integrated using
# a custom quadrature rule of degree :math:`2d-2` i.e. for linear (:math:`d=1`)
# quadrilaterals, the shear energy is integrated as if it were constant (1 Gauss point instead of 2x2)
# and for quadratic (:math:`d=2`) quadrilaterals, as if it were quadratic (2x2 Gauss points instead of 3x3)::

u = Function(V)
u_ = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

(disp, rot) = ufl.TrialFunctions(V)
(_disp, _rot) = ufl.TestFunctions(V)


dx_shear = ufl.dx(metadata={"quadrature_degree": 2*deg-2})

x = ufl.SpatialCoordinate(msh)
# inflow_function = dolfinx.fem.Function(V.sub(0))
# inflow_values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
# inflow_function.interpolate(inflow_values)

f = -thick**3
# f = 10.0 * ufl.exp(-((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)) / 0.02)  
L = ufl.inner(f, disp) * ufl.dx
a = ufl.inner(bending_moment(disp, rot), curv(_disp, _rot))*ufl.dx \
    + ufl.dot(shear_force(disp, rot), shear_strain(_disp, _rot))*dx_shear


problem = fem.petsc.LinearProblem(a, L, bcs=bc, petsc_options={
                                  "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
try:
    w_h = problem.solve()
except PETSc.Error as e:
    if e.ierr == 92:
        print("The required PETSc solver/preconditioner is not available. Exiting.")
        print(e)
        exit(0)
    else:
        raise e

disp_h, rot_h = w_h.split()

with io.XDMFFile(msh.comm, "out_mixed_poisson/u.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(disp_h)
    file.write_function(rot_h)

print("Kirchhoff deflection:", -1.265319087e-3*float(f/D))
print("Reissner-Mindlin FE deflection:", -np.min(disp_h.vector.array[:])) # point evaluation for quads
                                                                        # is not implemented in fenics 2017.2





# We then solve for the solution and export the relevant fields to XDMF files ::

# solve(a == L, u, bc)

# (w, theta) = fem.split(u)

# Vw = FunctionSpace(mesh, We)
# Vt = FunctionSpace(mesh, Te)
# ww = u.sub(0, True)
# ww.rename("Deflection", "")
# tt = u.sub(1, True)
# tt.rename("Rotation", "")

# file_results = XDMFFile("RM_results.xdmf")
# file_results.parameters["flush_output"] = True
# file_results.parameters["functions_share_mesh"] = True
# file_results.write(ww, 0.)
# file_results.write(tt, 0.)

# # The solution is compared to the Kirchhoff analytical solution::

# print("Kirchhoff deflection:", -1.265319087e-3*float(f/D))
# print("Reissner-Mindlin FE deflection:", -min(ww.vector().get_local())) # point evaluation for quads
#                                                                         # is not implemented in fenics 2017.2

# # For :math:`h=0.001` and 50 quads per side, one finds :math:`w_{FE} = 1.38182\text{e-5}` for linear quads
# # and :math:`w_{FE} = 1.38176\text{e-5}` for quadratic quads against :math:`w_{\text{Kirchhoff}} = 1.38173\text{e-5}` for
# # the thin plate solution.

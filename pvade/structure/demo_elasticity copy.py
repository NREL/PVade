# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Elasticity
#
# Copyright © 2020-2022 Garth N. Wells and Michal Habera
#
# This demo solves the equations of static linear elasticity using a
# smoothed aggregation algebraic multigrid solver. The demo is
# implemented in {download}`demo_elasticity.py`.

# +
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
from ufl import dx, grad, inner

from mpi4py import MPI
from petsc4py import PETSc
import gmsh
import meshio
import os 

dtype = PETSc.ScalarType
# -

# ## Operator's nullspace
#
# Smooth aggregation algebraic multigrid solvers require the so-called
# 'near-nullspace', which is the nullspace of the operator in the
# absence of boundary conditions. The below function builds a PETSc
# NullSpace object. For this 3D elasticity problem the nullspace is
# spanned by six vectors -- three translation modes and three rotation
# modes.


def build_nullspace(V):
    """Build PETSc nullspace for 3D elasticity"""

    # Create list of vectors for building nullspace
    index_map = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    ns = [la.create_petsc_vector(index_map, bs) for i in range(6)]
    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in ns]
        basis = [np.asarray(x) for x in vec_local]

        # Get dof indices for each subspace (x, y and z dofs)
        dofs = [V.sub(i).dofmap.list.array for i in range(3)]

        # Build the three translational rigid body modes
        for i in range(3):
            basis[i][dofs[i]] = 1.0

        # Build the three rotational rigid body modes
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.array
        x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
        basis[3][dofs[0]] = -x1
        basis[3][dofs[1]] = x0
        basis[4][dofs[0]] = x2
        basis[4][dofs[2]] = -x0
        basis[5][dofs[2]] = x1
        basis[5][dofs[1]] = -x2

    # Orthonormalise the six vectors
    la.orthonormalize(ns)
    assert la.is_orthonormal(ns)

    return PETSc.NullSpace().create(vectors=ns)

def create_mesh():
     # Initialize Gmsh options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        # All ranks create a Gmsh model object
        gmsh_model = gmsh.model()
        gmsh_model.add("domain")
        gmsh_model.setCurrent("domain")

 


        for panel_id in range(num_rows):
            panel_box = gmsh_model.occ.addBox(
                -0.5 * panel_length,
                0.0,
                -0.5 * panel_thickness,
                panel_length,
                panel_width,
                panel_thickness,
            )

            # Rotate the panel currently centered at (0, 0, 0)
            gmsh_model.occ.rotate(
                [(3, panel_box)], 0, 0, 0, 0, -1, 0, tracker_angle_rad
            )

            # Translate the panel [panel_loc, 0, elev]
            gmsh_model.occ.translate(
                [(3, panel_box)],
                panel_id * spacing[0],
                0,
                elevation,
            )

            # # Remove each panel from the overall domain
            # gmsh_model.occ.cut([(3, domain)], [(3, panel_box)])

        gmsh_model.occ.synchronize()

        """Creates boundary tags using gmsh 
        """
        # Loop through all surfaces to find periodic tags
        surf_ids = gmsh_model.occ.getEntities(2)

        dom_tags = {}
        
        panel_id = 0 
        count = 0 
        for surf in surf_ids:
            tag = surf[1]

            com = gmsh_model.occ.getCenterOfMass(2, tag)
            print (com)
        
           
            tags = np.array([1,2,3,4,5,6])+6*(panel_id)

            if tag == tags[0]:
                    dom_tags[f"x_right_pannel_{panel_id}"] = [tag]
            elif tag == tags[1]:
                    dom_tags[f"x_left_pannel_{panel_id}"] = [tag]
            elif tag == tags[2]:
                    dom_tags[f"y_right_pannel_{panel_id}"] = [tag]
            elif tag == tags[3]:
                    dom_tags[f"y_left_pannel_{panel_id}"] = [tag]
            elif tag == tags[4]:
                    dom_tags[f"z_right_pannel_{panel_id}"] = [tag]
            elif tag == tags[5]:
                    dom_tags[f"z_left_pannel_{panel_id}"] = [tag]
            count = count + 1
            if count == 6: 
                panel_id = panel_id + 1 
                count = 0   

        for panel_id in range(num_rows): 
            marker = np.array([1,2,3,4,5,6])+6*(panel_id)
            panel_marker = 100*(panel_id+1)
            gmsh_model.addPhysicalGroup(2, dom_tags[f"x_right_pannel_{panel_id}"], marker[0])
            gmsh_model.setPhysicalName(2, marker[0], f"bottom_{panel_id}")
            gmsh_model.addPhysicalGroup(2, dom_tags[f"x_left_pannel_{panel_id}"],  marker[1])
            gmsh_model.setPhysicalName(2, marker[1], f"top_{panel_id}")
            gmsh_model.addPhysicalGroup(2, dom_tags[f"y_right_pannel_{panel_id}"], marker[2])
            gmsh_model.setPhysicalName(2, marker[2], f"left_{panel_id}")
            gmsh_model.addPhysicalGroup(2, dom_tags[f"y_left_pannel_{panel_id}"],  marker[3])
            gmsh_model.setPhysicalName(2, marker[3], f"right_{panel_id}")
            gmsh_model.addPhysicalGroup(2, dom_tags[f"z_right_pannel_{panel_id}"], marker[4])
            gmsh_model.setPhysicalName(2, marker[4], f"back_{panel_id}")
            gmsh_model.addPhysicalGroup(2, dom_tags[f"z_left_pannel_{panel_id}"],  marker[5])
            gmsh_model.setPhysicalName(2, marker[5], f"front_{panel_id}")

            # self.pv_model.addPhysicalGroup(3, [1], 11)
            # self.pv_model.setPhysicalName(3, 11, f"panel_{panel_id}")

            gmsh_model.addPhysicalGroup(3, [(panel_id+1)], panel_marker)
            gmsh_model.setPhysicalName(3, panel_marker, f"panel_{panel_id}")

        if rank == 0:
            all_pts = gmsh_model.occ.getEntities(0)
            gmsh_model.mesh.setSize(all_pts, 0.2)

        if rank == 0:
            print("Starting mesh generation... ", end="")

            # Generate the mesh
            # tic = time.time()
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            # gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
            gmsh_model.mesh.generate(3)
            gmsh_model.mesh.setOrder(2)
            gmsh_model.mesh.optimize("Netgen")
            gmsh_model.mesh.generate(3)
            # toc = time.time()
            # if self.rank == 0:
            #     print("Finished.")
            #     print(f"Total meshing time = {toc-tic:.1f} s")







            # All ranks receive their portion of the mesh from rank 0 (like an MPI scatter)
            msh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh_model, comm, 0,gdim=3)

        return msh,cell_tags,facet_tags


def write_mesh_file():
        if rank == 0:
            # Save the *.msh file and *.vtk file (the latter is for visualization only)
            print('Writing Mesh to %s... ' % ("out_elasticity/mesh"), end='')
            
            if os.path.exists("out_elasticity/mesh") == False: 
                os.makedirs("out_elasticity/mesh")
            gmsh.write('%s/mesh.msh' % ("out_elasticity/mesh"))
            gmsh.write('%s/mesh.vtk' % ("out_elasticity/mesh"))
            # def create_mesh(mesh, clean_points, cell_type):
            #     cells = mesh.get_cells_type(cell_type)
            #     cell_data = mesh.get_cell_data("gmsh:physical", cell_type)

            #     out_mesh = meshio.Mesh(points=clean_points, cells={
            #                         cell_type: cells}, cell_data={"name_to_read": [cell_data]})
            #     return out_mesh
                
            # mesh_from_file = meshio.read(f'{"out_elasticity/mesh"}/mesh.msh')
            # pts = mesh_from_file.points
            # tetra_mesh = create_mesh(mesh_from_file, pts, "tetra")
            # tri_mesh = create_mesh(mesh_from_file, pts, "triangle")

            # meshio.write(f'{"out_elasticity/mesh"}/mesh.xdmf', tetra_mesh)
            # meshio.write(f'{"out_elasticity/mesh"}/mesh_mf.xdmf', tri_mesh)
            print("Done.")
                   

# Create a box Mesh

# Get MPI communicators
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_procs = 0

# inputs 
num_rows = 7
panel_length = 2.0
panel_width = 7.0
panel_thickness = .1
tracker_angle_rad = np.radians(30.0)
spacing = [7.0]
elevation = 1.5


msh,mt,ft = create_mesh()

write_mesh_file()



ndim = msh.topology.dim

# Specify names for the mesh elements
msh.name = "panels"
mt.name = f"{msh.name}_cells"
ft.name = f"{msh.name}_facets"
# num_pannels = [1,2,3]
# x_spacing = 0


# for n in num_pannels:
#     msh = create_box(MPI.COMM_WORLD, [np.array([0.0+x_spacing, 0.0, 0.0]),
#                                   np.array([2.0+x_spacing, 7.0, .1])], [16, 16, 16],
#                  CellType.tetrahedron, GhostMode.shared_facet)
#     x_spacing = x_spacing +3


with XDMFFile(msh.comm, "out_elasticity/mesh.xdmf", "w") as file:
        file.write_mesh(msh)
    

# Create a centripetal source term ($f = \rho \omega^2 [x_0, \, x_1]$)
 
ω, ρ = 300.0, 10.0
x = ufl.SpatialCoordinate(msh)
# f = ufl.as_vector((0*ρ * ω**2 * x[0], ρ * ω**2 * x[1], 0*ρ * ω**2 * x[0]))


# Set the elasticity parameters and create a function that computes and
# expression for the stress given a displacement field.

# +
E = 1.0e9
ν = 0.3
μ = E / (2.0 * (1.0 + ν))
λ = E * ν / ((1.0 + ν) * (1.0 - 2.0 * ν))


def σ(v):
    """Return an expression for the stress σ given a displacement field"""
    return 2.0 * μ * ufl.sym(grad(v)) + λ * ufl.tr(ufl.sym(grad(v))) * ufl.Identity(len(v))
# -

# A function space space is created and the elasticity variational
# problem defined:
T = fem.Constant(msh, PETSc.ScalarType((0, 1.e-3, 0)))
f = fem.Constant(msh, PETSc.ScalarType((0, 0, -ρ*9.81)))
ds = ufl.Measure("ds", domain=msh)

V = VectorFunctionSpace(msh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = form(inner(σ(u), grad(v)) * dx)
# L =   form(inner(t, v) * ds)
L = form(ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds)
# L = ufl.dot(f, v) * ufl.dx + ufl.dot(t, v) * ufl.ds

# A homogeneous (zero) boundary condition is created on $x_0 = 0$ and
# $x_1 = 1$ by finding all boundary facets on $x_0 = 0$ and $x_1 = 1$,
# and then creating a Dirichlet boundary condition object.

# facets = locate_entities_boundary(msh, dim=2,
                                #   marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                #  np.isclose(x[1], 1.0)))
# bc = dirichletbc(np.zeros(3, dtype=dtype),
                #  locate_dofs_topological(V, entity_dim=2, entities=facets), V=V)



zero_vec = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0, 0.0)))
bc = []
for num_panel in range(num_rows):
    for marker in np.array([1])+6*num_panel:#range(6*num_rows): 
        dofs = locate_dofs_topological(V, 2, ft.find(marker))
        bc.append(dirichletbc(zero_vec, dofs, V))



# points = msh.geometry.x

# print(points)
# facets = locate_entities_boundary(msh, dim=2,
#                                   marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
#                                                                  np.isclose(x[1], 1.0)))

# # facets = locate_entities_boundary(msh, dim=msh.topology.dim-1,
# #                                   marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
# #                                                                  np.isclose(x[1], 1.0)))
# bc = dirichletbc(np.zeros(3, dtype=dtype),
#                  locate_dofs_topological(V, entity_dim=2, entities=facets), V=V)



# ## Assemble and solve
#
# The bilinear form `a` is assembled into a matrix `A`, with
# modifications for the Dirichlet boundary conditions. The line
# `A.assemble()` completes any parallel communication required to
# computed the matrix.

# +

# +
A = fem.petsc.assemble_matrix(a, bcs=bc)
A.assemble()
# -

# The linear form `L` is assembled into a vector `b`, and then modified
# by {py:func}`apply_lifting <dolfinx.fem.petsc.apply_lifting>` to
# account for the Dirichlet boundary conditions. After calling
# {py:func}`apply_lifting <dolfinx.fem.petsc.apply_lifting>`, the method
# `ghostUpdate` accumulates entries on the owning rank, and this is
# followed by setting the boundary values in `b`.

# +
b = fem.petsc.assemble_vector(L)
dolfinx.fem.petsc.apply_lifting(b, [a], bcs=[bc])
b.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        )
fem.petsc.set_bc(b, bc)
# -

# Create the near-nullspace and attach it to the PETSc matrix:

ns = build_nullspace(V)
A.setNearNullSpace(ns)

# Set PETSc solver options, create a PETSc Krylov solver, and attach the
# matrix `A` to the solver:

# +
# Set solver options
opts = PETSc.Options()
opts["ksp_type"] = "cg"
opts["ksp_rtol"] = 1.0e-10
opts["pc_type"] = "gamg"

# Use Chebyshev smoothing for multigrid
opts["mg_levels_ksp_type"] = "chebyshev"
opts["mg_levels_pc_type"] = "jacobi"

# Improve estimate of eigenvalues for Chebyshev smoothing
opts["mg_levels_esteig_ksp_type"] = "cg"
opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

# Create PETSc Krylov solver and turn convergence monitoring on
solver = PETSc.KSP().create(msh.comm)
solver.setFromOptions()

# Set matrix operator
solver.setOperators(A)
# -

# Create a solution {py:class}`Function<dolfinx.fem.Function>` `uh` and
# solve:

# +
uh = Function(V)

# Set a monitor, solve linear system, and display the solver
# configuration
solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
solver.solve(b, uh.vector)
solver.view()

# Scatter forward the solution vector to update ghost values
uh.x.scatter_forward()
# -

# ## Post-processing
#
# The computed solution is now post-processed. Expressions for the
# deviatoric and Von Mises stress are defined:

# +
sigma_dev = σ(uh) - (1 / 3) * ufl.tr(σ(uh)) * ufl.Identity(len(uh))
sigma_vm = ufl.sqrt((3 / 2) * inner(sigma_dev, sigma_dev))
# -

# Next, the Von Mises stress is interpolated in a piecewise-constant
# space by creating an {py:class}`Expression<dolfinx.fem.Expression>`
# that is interpolated into the
# {py:class}`Function<dolfinx.fem.Function>` `sigma_vm_h`.





# +
W = FunctionSpace(msh, ("Discontinuous Lagrange", 0))
sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points())
sigma_vm_h = Function(W)
sigma_vm_h.interpolate(sigma_vm_expr)
# -


n = ufl.FacetNormal(msh)

traction_vec = ufl.dot(sigma_dev,n)



# Save displacement field `uh` and the Von Mises stress `sigma_vm_h` in
# XDMF format files.

# +
with XDMFFile(msh.comm, "out_elasticity/displacements.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)
    file.write_function(traction_vec)

# Save solution to XDMF format
with XDMFFile(msh.comm, "out_elasticity/von_mises_stress.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(sigma_vm_h)
# -

# Finally, we compute the $L^2$ norm of the displacement solution
# vector. This is a collective operation (i.e., the method `norm` must
# be called from all MPI ranks), but we print the norm only on rank 0.

# +
unorm = uh.x.norm()
if msh.comm.rank == 0:
    print("Solution vector norm:", unorm)
# -

# The solution vector norm can be a useful check that the solver is
# computing the same result when running in serial and in parallel.

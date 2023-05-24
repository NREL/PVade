# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Elasticity using algebraic multigrid
#
# Copyright © 2020-2022 Garth N. Wells and Michal Habera
#
# This demo ({download}`demo_elasticity.py`) solves the equations of
# static linear elasticity using a smoothed aggregation algebraic
# multigrid solver. It illustrates how to:
#
# - Use a smoothed aggregation algebraic multigrid solver
# - Use {py:class}`Expression <dolfinx.fem.Expression>` to compute
#   derived quantities of a solution
#
# The required modules are first imported:

# +
import numpy as np
import ufl
from dolfinx.fem import ( Function, 
                         VectorFunctionSpace)
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import la
import dolfinx
import os 
dtype = PETSc.ScalarType
# -

# ## Create the operator near-nullspace
#
# Smooth aggregation algebraic multigrid solvers require the so-called
# 'near-nullspace', which is the nullspace of the operator in the
# absence of boundary conditions. The below function builds a
# `PETSc.NullSpace` object for a 3D elasticity problem. The nullspace is
# spanned by six vectors -- three translation modes and three rotation
# modes.

def write_mesh(mesh, mesh_filename, cell_tags=None, facet_tags=None):
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_filename, "w") as xdmf:
        xdmf.write_mesh(mesh)
        if cell_tags is not None:
            xdmf.write_meshtags(cell_tags)
        if facet_tags is not None:
            xdmf.write_meshtags(facet_tags)

def read_mesh(mesh_filename, mesh_name="mesh"):
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_filename, "r") as xdmf:
        mesh = xdmf.read_mesh(name=mesh_name)

        try:
            cell_tags = xdmf.read_meshtags(mesh, name="cell_tags")
        except:
            cell_tags = None

        ndim = mesh.topology.dim
        mesh.topology.create_connectivity(ndim - 1, ndim)

        try:
            facet_tags = xdmf.read_meshtags(mesh, "facet_tags")
        except:
            facet_tags = None

    return mesh, cell_tags, facet_tags

def build_nullspace(V):
    """Build PETSc nullspace for 3D elasticity"""

    # Create vectors that will span the nullspace
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    length1 = length0 + V.dofmap.index_map.num_ghosts
    basis = [np.zeros(bs * length1, dtype=dtype) for i in range(6)]

    # Get dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list.array for i in range(3)]

    # Set the three translational rigid body modes
    for i in range(3):
        basis[i][dofs[i]] = 1.0

    # Set the three rotational rigid body modes
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.array
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    basis[3][dofs[0]] = -x1
    basis[3][dofs[1]] = x0
    basis[4][dofs[0]] = x2
    basis[4][dofs[2]] = -x0
    basis[5][dofs[2]] = x1
    basis[5][dofs[1]] = -x2

    # Create PETSc Vec objects (excluding ghosts) and normalise
    basis_petsc = [PETSc.Vec().createWithArray(x[:bs * length0], bsize=3, comm=V.mesh.comm) for x in basis]
    la.orthonormalize(basis_petsc)
    assert la.is_orthonormal(basis_petsc)

    # Create and return a PETSc nullspace
    return PETSc.NullSpace().create(vectors=basis_petsc)
# import numpy as np
# from mpi4py import MPI
# import dolfinx, dolfinx.io, dolfinx.fem as fem, dolfinx.mesh

# ## Problem definition
msh1 = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10, dolfinx.mesh.CellType.tetrahedron)

submesh_entities = dolfinx.mesh.locate_entities(msh1, dim=3, marker=lambda x: (x[0] < 0.5) & (x[1] < 0.5))
msh2, entity_map, vertex_map, geom_map = dolfinx.mesh.create_submesh(msh1, msh1.topology.dim, submesh_entities)

TEST_WRITE_TO_READ_HACK = True
if TEST_WRITE_TO_READ_HACK:
    temp_sphere = os.path.join("sphere_temp.xdmf")
    write_mesh(msh2, temp_sphere)
    msh2, _, _ = read_mesh(temp_sphere)

    # temp_box = os.path.join(OUTPUT_DIR, "box_temp.xdmf")
    # write_mesh(box_submesh, temp_box)
    # box_submesh, _, _ = read_mesh(temp_box)


    # sphere_num_cells = get_num_cells_in_mesh(sphere_submesh)
    # box_num_cells = get_num_cells_in_mesh(box_submesh)
    # print(f"AFTER REPARTITION: Rank {rank:2d}: Sphere Cells = {sphere_num_cells:5d},  Box Cells = {box_num_cells:5d},  Total Cells = {sphere_num_cells+box_num_cells:5d}")


    # sphere_num_cells_global_2 = comm.gather(sphere_num_cells, root=0)




V1 = VectorFunctionSpace(msh1, ("Lagrange", 1))
V2 = VectorFunctionSpace(msh2, ("Lagrange", 1))

u1 = Function(V1)
u2 = Function(V2)

u1.interpolate(lambda x: np.vstack((x[0], x[1],x[2])))

u2.interpolate(u1)
u2.x.scatter_forward()

def store_vec_rank1(functionspace, vector):
    imap = vector.function_space.dofmap.index_map
    local_range = np.asarray(imap.local_range, dtype=np.int32) * vector.function_space.dofmap.index_map_bs
    size_global = imap.size_global * vector.function_space.dofmap.index_map_bs

    # Communicate ranges and local data
    ranges = MPI.COMM_WORLD.gather(local_range, root=0)
    data = MPI.COMM_WORLD.gather(vector.vector.array, root=0)
    # print(data)
    # Communicate local dof coordinates
    x = functionspace.tabulate_dof_coordinates()[:imap.size_local]
    x_glob = MPI.COMM_WORLD.gather(x, root=0)

    # Declare gathered parallel arrays
    global_array = np.zeros(size_global)
    global_x = np.zeros((size_global, 3))
    u_from_global = np.zeros(global_array.shape)

    x0 = functionspace.tabulate_dof_coordinates()

    if MPI.COMM_WORLD.rank == 0:
        # Create array with all values (received)
        for r, d in zip(ranges, data):
            global_array[r[0]:r[1]] = d

    return global_array

solution_vec = np.sort(np.array(store_vec_rank1(V2,u2)))
solution_vec = solution_vec.reshape((-1, 1))


if MPI.COMM_WORLD.size == 1:
    np.savetxt('solution_1rank_output.txt', solution_vec, delimiter=',')
    # with open('solution_1rank_output.txt', 'w') as filehandle:
    #     json.dump(solution_vec.toList(), filehandle)
else:
    vec_check = np.genfromtxt('solution_1rank_output.txt', delimiter=",")
    # my_file = open('solution_1rank_output.txt', 'r')
    # vec_check = my_file.read()


if MPI.COMM_WORLD.rank == 0 and MPI.COMM_WORLD.size > 1:
    for n in range(solution_vec.size):
        if abs(vec_check[n] - solution_vec[n]) > 1e-6 :
            print(f"solution vec incorrect in {n}th entry \n {vec_check[n]} compared to  {solution_vec[n]} np = {MPI.COMM_WORLD.size} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            exit()
    print(f"all values match with np = {MPI.COMM_WORLD.size}")

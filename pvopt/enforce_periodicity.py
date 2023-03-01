from dolfin import *

parameters["mesh_partitioner"] = "ParMETIS"
parameters["partitioning_approach"] = "REPARTITION"
parameters["ParMETIS_repartitioning_weight"] = 100000000.0
import numpy as np

comm = MPI.comm_world
rank = comm.Get_rank()
num_procs = comm.Get_size()


x_range = [-10.5, 10.5]
y_range = [-3.5, 3.5]
z_range = [0.0, 15.0]

method = 3

if method == 1:

    mesh_name = "periodic_3_panel/theta_30_coarse/mesh.xdmf"

    mesh = Mesh()
    with XDMFFile(MPI.comm_world, mesh_name) as infile:
        infile.read(mesh)

    # if rank == 0:
    #     print('Successfully read XDMF file')

    # mvc = MeshValueCollection('size_t', mesh, 1)
    # with XDMFFile(MPI.comm_world, mesh_name.split('.xdmf')[0]+'_mf.xdmf') as infile:
    #     infile.read(mvc, 'name_to_read')

    # mf = MeshFunction('size_t', mesh, mvc)

    # hdfile = HDF5File(MPI.comm_world, 'converted_mesh.h5', "w")
    # hdfile.write(mesh,'mesh')
    # hdfile.write(mf,'boundary_markers')
    # hdfile.close()

elif method == 2:
    mesh = Mesh(MPI.comm_world)
    h5file = HDF5File(
        MPI.comm_world, "periodic_3_panel/theta_30_coarse/h5_mesh.h5", "r"
    )
    h5file.read(mesh, "/mesh", False)
    # mf = MeshFunction("size_t", mesh, mesh.geometry().dim()-1)
    # h5file.read(mf, '/boundary_markers')

else:
    nn = 25
    pt1 = Point(x_range[0], y_range[0], z_range[0])
    pt2 = Point(x_range[1], y_range[1], z_range[1])
    mesh = BoxMesh(pt1, pt2, nn * 3, nn, nn * 3)

print("Finished reading mesh.")


class PeriodicBoundary(SubDomain):
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return on_boundary and near(x[0], x_range[0]) or on_boundary and near(x[1], y_range[0])
        return (
            near(x[0], x_range[0])
            and x[1] < y_range[1] - DOLFIN_EPS
            and on_boundary
            or near(x[1], y_range[0])
            and x[0] < x_range[1] - DOLFIN_EPS
            and on_boundary
        )

    # Map right boundary to left boundary and back boundary to front boundary
    def map(self, x, y):
        if near(x[0], x_range[1]) and near(x[1], y_range[1]):
            y[0] = x[0] - (x_range[1] - x_range[0])
            y[1] = x[1] - (y_range[1] - y_range[0])
            y[2] = x[2]

        elif near(x[0], x_range[1]):
            y[0] = x[0] - (x_range[1] - x_range[0])
            y[1] = x[1]
            y[2] = x[2]

        elif near(x[1], y_range[1]):
            y[0] = x[0]
            y[1] = x[1] - (y_range[1] - y_range[0])
            y[2] = x[2]

        else:
            y[0] = x[0] - 1000
            y[1] = x[1] - 1000
            y[2] = x[2] - 1000


# This gives wiggle room when mapping one periodic surface to another
periodic_map_tolerance = 1e-10

# Velocity (Vector)
V = VectorFunctionSpace(
    mesh, "P", 2, constrained_domain=PeriodicBoundary(map_tol=periodic_map_tolerance)
)
Q = FunctionSpace(
    mesh, "P", 1, constrained_domain=PeriodicBoundary(map_tol=periodic_map_tolerance)
)

interp_exp = Expression(("x[0]", "x[1]", "x[2]"), degree=2)
fn = project(interp_exp, V, solver_type="cg", preconditioner_type="jacobi")

fn_rank = Function(Q)
fn_rank.vector()[:] = rank
fn_rank.rename("rank", "rank")

fp_rank = File("enforce_periodicity/rank.pvd")
fp_rank << fn_rank
proj_method = 2

if proj_method == 1:
    interp_exp = Expression(("x[0]", "x[1]", "x[2]"), degree=2)
    fn = project(interp_exp, V, solver_type="cg")

elif proj_method == 2:

    fn = Function(V)
    coords = V.tabulate_dof_coordinates()[::3]
    vec = np.zeros(np.size(coords))
    eps = 1e-6

    for k, coords in enumerate(coords):
        if coords[0] < x_range[0] + eps:
            vec[3 * k] = -1.0
            # pass
        elif coords[0] > x_range[1] - eps:
            vec[3 * k] = 1.0
            # pass

    fn.vector()[:] = vec

fn.rename("function", "function")
fp = File("enforce_periodicity/fn.pvd")
fp << fn

print("Finished")

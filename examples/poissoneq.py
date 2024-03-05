import numpy as np
import sys
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from timeit import default_timer as timer
from dolfinx.common import TimingType, list_timings


start = timer()
elems = int(sys.argv[1])  # nelems
# Create mesh and define function space
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (2.0, 2.0)),
    n=(elems, elems),
    cell_type=mesh.CellType.triangle,
)
comm = MPI.COMM_WORLD
V = fem.FunctionSpace(msh, ("Lagrange", int(sys.argv[4])))


facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 2.0)),
)
dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0])
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds
solv = sys.argv[2]  # solver method
prec = sys.argv[3]  # preconditioner
# Compute solution
problem = LinearProblem(
    a, L, bcs=[bc], petsc_options={"ksp_type": solv, "pc_type": prec}
)
uh = problem.solve()

if comm.rank == 0:
    print("system of equation solved")
# Save solution in VTK format
# file = File("output/poisson.pvd")
# file << u

if comm.rank == 0:
    print("Simulation Done")
    end = timer()
    print(end - start)

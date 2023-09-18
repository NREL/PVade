"""Summary
"""
import dolfinx
import ufl

from petsc4py import PETSc
from mpi4py import MPI

import numpy as np
import scipy.interpolate as interp

import warnings

from pvade.structure.boundary_conditions import build_structure_boundary_conditions
from contextlib import ExitStack


class Elasticity:
    """This class solves the CFD problem"""

    def __init__(self, domain, structural_analysis):
        """Initialize the fluid solver

        This method initialize the Flow object, namely, it creates all the
        necessary function spaces on the mesh, initializes key counting and
        boolean variables and records certain characteristic quantities like
        the minimum cell size and the number of degrees of freedom attributed
        to both the pressure and velocity function spaces.

        Args:
            domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object

        """
        self.structural_analysis = structural_analysis
        # Store the comm and mpi info for convenience
        self.comm = domain.comm
        self.rank = domain.rank
        self.num_procs = domain.num_procs

        # domain.structure.msh = dolfinx.mesh.refine(domain.structure.msh,None)
        # domain.structure.msh = dolfinx.mesh.refine(domain.structure.msh)

        P1 = ufl.VectorElement("Lagrange", domain.structure.msh.ufl_cell(), 1)
        self.V = dolfinx.fem.FunctionSpace(domain.structure.msh, P1)

        self.W = dolfinx.fem.FunctionSpace(
            domain.structure.msh, ("Discontinuous Lagrange", 0)
        )

        self.first_call_to_solver = True

        self.num_V_dofs = (
            self.V.dofmap.index_map_bs * self.V.dofmap.index_map.size_global
        )

        # Store the dimension of the problem for convenience
        self.ndim = domain.structure.msh.topology.dim
        self.facet_dim = self.ndim - 1

        # find hmin in mesh
        num_cells = domain.structure.msh.topology.index_map(self.ndim).size_local
        h = dolfinx.cpp.mesh.h(domain.structure.msh, self.ndim, range(num_cells))

        # This value of hmin is local to the mesh portion owned by the process
        hmin_local = np.amin(h)

        # collect the minimum hmin from all processes
        self.hmin = np.zeros(1)
        self.comm.Allreduce(hmin_local, self.hmin, op=MPI.MIN)
        self.hmin = self.hmin[0]

        if self.rank == 0:
            print(f"hmin on structure = {self.hmin}")
            print(f"Total num dofs on structure = {self.num_V_dofs}")

    def build_boundary_conditions(self, domain, params):
        """Build the boundary conditions

        A method to manage the building of boundary conditions, including the
        steps of identifying entities on the boundary, marking those degrees
        of freedom either by the identified facets or a gmsh marker function,
        and finally assembling a list of Boundary objects that enforce the
        correct value.

        Args:
            domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object

        """

        # self.bcu, self.inflow_profile = build_velocity_boundary_conditions(
        #     domain, params, self.V
        # )

        self.bc = build_structure_boundary_conditions(domain, params, self.V)

    def build_forms(self, domain, params):
        """Builds all variational statements

        This method creates all the functions, expressions, and variational
        forms that will be needed for the numerical solution of Navier Stokes
        using a fractional step method. This includes the calculation of a
        tentative velocity, the calculation of the change in pressure
        required to correct the tentative velocity to enforce continuity, and
        the update to the velocity field to reflect this change in
        pressure.

        Args:
            domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object

        """
        # Define structural properties
        self.E = params.structure.elasticity_modulus  # 1.0e9
        self.ν = params.structure.poissons_ratio  # 0.3
        self.μ = self.E / (2.0 * (1.0 + self.ν))
        self.λ = self.E * self.ν / ((1.0 + self.ν) * (1.0 - 2.0 * self.ν))

        # Define trial and test functions for deformation
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        P3 = ufl.TensorElement("Lagrange", domain.structure.msh.ufl_cell(), 2)
        self.T = dolfinx.fem.FunctionSpace(domain.structure.msh, P3)

        self.stress = dolfinx.fem.Function(self.T, name="stress_fluid")

        self.sigma_vm_h = dolfinx.fem.Function(self.W, name="Stress")

        self.uh = dolfinx.fem.Function(self.V, name="Deformation")
        self.uh_old = dolfinx.fem.Function(self.V, name="Deformation_old")
        self.uh_delta = dolfinx.fem.Function(self.V, name="Deformation_change")

        # self.uh_exp = dolfinx.fem.Function(self.V,  name="Deformation")

        def σ(v):
            """Return an expression for the stress σ given a displacement field"""
            return 2.0 * self.μ * ufl.sym(ufl.grad(v)) + self.λ * ufl.tr(
                ufl.sym(ufl.grad(v))
            ) * ufl.Identity(len(v))

        # source term ($f = \rho \omega^2 [x_0, \, x_1]$)
        self.ω, self.ρ = 300.0, 10.0
        # x = ufl.SpatialCoordinate(domain.structure.msh)
        # self.f = ufl.as_vector((0*self.ρ * self.ω**2 * x[0], self.ρ * self.ω**2 * x[1], 0.0))
        # self.f_structure = dolfinx.fem.Constant(
        #     domain.structure.msh,
        #     (PETSc.ScalarType(0), PETSc.ScalarType(0), PETSc.ScalarType(0)),
        # )
        # self.f = ufl.as_vector((0*self.ρ * self.ω**2 * x[0], self.ρ * self.ω**2 * x[1], 0.0))
        # self.T = dolfinx.fem.Constant(domain.structure.msh, PETSc.ScalarType((0, 1.e-3, 0)))
        # self.f = dolfinx.fem.Constant(domain.structure.msh, PETSc.ScalarType((0,100,100)))
        self.f = dolfinx.fem.Constant(
            domain.structure.msh,
            PETSc.ScalarType(
                (
                    params.structure.body_force_x,
                    params.structure.body_force_y,
                    params.structure.body_force_z,
                )
            ),
        )
        self.ds = ufl.Measure("ds", domain=domain.structure.msh)
        n = ufl.FacetNormal(domain.structure.msh)
        self.a = dolfinx.fem.form(ufl.inner(σ(self.u), ufl.grad(self.v)) * ufl.dx)
        self.L = dolfinx.fem.form(
            ufl.dot(self.f, self.v) * ufl.dx
            + ufl.dot(ufl.dot(self.stress, n), self.v) * self.ds
        )

    def _assemble_system(self, params):
        """Pre-assemble all LHS matrices and RHS vectors

        Here we pre-assemble all the forms corresponding to the left-hand side
        matrices and right-hand side vectors once outside the time loop. This
        will enable us to re-use certain features like the sparsity pattern
        during the timestepping without any modification of the function
        calls.

        Args:
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object
        """
        # try:
        #     self.A.zeroEntries()
        # except:
        #     print("not zeroing")

        # try:
        #     with self.b.localForm() as loc:
        #         loc.set(0)
        # except:
        #     pass

        self.A = dolfinx.fem.petsc.assemble_matrix(self.a, bcs=self.bc)
        self.A.assemble()
        self.b = dolfinx.fem.petsc.assemble_vector(self.L)

        if self.first_call_to_solver:
            # Set solver options
            opts = PETSc.Options()
            opts["ksp_type"] = "cg"
            opts["ksp_rtol"] = 1.0e-6
            opts["pc_type"] = "gamg"

            # Use Chebyshev smoothing for multigrid
            opts["mg_levels_ksp_type"] = "chebyshev"
            opts["mg_levels_pc_type"] = "jacobi"

            # Improve estimate of eigenvalues for Chebyshev smoothing
            opts["mg_levels_esteig_ksp_type"] = "cg"
            opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10

            # Create PETSc Krylov solver and turn convergence monitoring on
            self.solver = PETSc.KSP().create(self.comm)
            self.solver.setFromOptions()

        # Set matrix operator
        self.solver.setOperators(self.A)

    def build_nullspace(self, V):
        """Build PETSc nullspace for 3D elasticity"""

        # Create list of vectors for building nullspace
        index_map = V.dofmap.index_map
        bs = V.dofmap.index_map_bs
        ns = [dolfinx.la.create_petsc_vector(index_map, bs) for i in range(6)]
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
        dolfinx.la.orthonormalize(ns)
        assert dolfinx.la.is_orthonormal(ns)

        return PETSc.NullSpace().create(vectors=ns)

    def solve(self, params, dataIO):
        """Solve for a single timestep advancement

        Here we perform the three-step solution process (tentative velocity,
        pressure correction, velocity update) to advance the fluid simulation
        a single timestep. Additionally, we calculate the new CFL number
        associated with the latest velocity solution.

        Args:
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object
        """

        def σ(v):
            """Return an expression for the stress σ given a displacement field"""
            return 2.0 * self.μ * ufl.sym(ufl.grad(v)) + self.λ * ufl.tr(
                ufl.sym(ufl.grad(v))
            ) * ufl.Identity(len(v))

        if self.first_call_to_solver:
            if self.rank == 0:
                print("Starting Strutural Solution")

        self._assemble_system(params)

        dolfinx.fem.petsc.apply_lifting(self.b, [self.a], bcs=[self.bc])
        self.b.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        )
        dolfinx.fem.petsc.set_bc(self.b, self.bc)
        # -

        # Create the near-nullspace and attach it to the PETSc matrix:

        ns = self.build_nullspace(self.V)
        self.A.setNearNullSpace(ns)

        # Store the old displacement/position for finite differencing
        self.uh_old.vector.array[:] = self.uh.vector.array[:]

        # self.solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
        # Compute displacement
        self.solver.solve(self.b, self.uh.vector)
        # self.solver.view()
        # Scatter forward the solution vector to update ghost values
        self.uh.x.scatter_forward()

        # Calculate the change in the displacement (new - old) this is what moves the mesh
        self.uh_delta.vector.array[:] = (
            self.uh.vector.array[:] - self.uh_old.vector.array[:]
        )

        sigma_dev = σ(self.uh) - (1 / 3) * ufl.tr(σ(self.uh)) * ufl.Identity(
            len(self.uh)
        )
        sigma_vm = ufl.sqrt((3 / 2) * ufl.inner(sigma_dev, sigma_dev))

        sigma_vm_expr = dolfinx.fem.Expression(
            sigma_vm, self.W.element.interpolation_points()
        )
        self.sigma_vm_h.interpolate(sigma_vm_expr)

        self.unorm = self.uh.x.norm()

        if self.first_call_to_solver:
            self.first_call_to_solver = False

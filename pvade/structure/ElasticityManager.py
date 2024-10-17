"""Summary
"""

import dolfinx
import ufl

from petsc4py import PETSc
from mpi4py import MPI

import numpy as np
import scipy.interpolate as interp

import warnings
import os

from pvade.structure.boundary_conditions import build_structure_boundary_conditions
from contextlib import ExitStack


class Elasticity:
    """This class solves the CFD problem"""

    def __init__(self, domain, structural_analysis, params):
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
        self.name = "structure"

        # Store the comm and mpi info for convenience
        self.comm = domain.comm
        self.rank = domain.rank
        self.num_procs = domain.num_procs

        # domain.structure.msh = dolfinx.mesh.refine(domain.structure.msh,None)
        # domain.structure.msh = dolfinx.mesh.refine(domain.structure.msh)

        P1 = ufl.VectorElement("Lagrange", domain.structure.msh.ufl_cell(), 2)
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

        # Mass density
        self.rho = dolfinx.fem.Constant(
            domain.structure.msh, params.structure.rho
        )  # Constant(0.)

        # Rayleigh damping coefficients
        self.eta_m = dolfinx.fem.Constant(domain.structure.msh, 0.0)  # Constant(0.)
        self.eta_k = dolfinx.fem.Constant(domain.structure.msh, 0.0)  # Constant(0.)

        # Generalized-alpha method parameters
        self.alpha_m = dolfinx.fem.Constant(domain.structure.msh, 0.2)
        self.alpha_f = dolfinx.fem.Constant(domain.structure.msh, 0.4)
        self.gamma = 0.5 + self.alpha_f - self.alpha_m
        self.beta = (self.gamma + 0.5) ** 2 / 4.0

        # Define structural properties
        self.E = params.structure.elasticity_modulus  # 1.0e9
        self.ν = params.structure.poissons_ratio  # 0.3
        self.μ = self.E / (2.0 * (1.0 + self.ν))
        self.λ = self.E * self.ν / ((1.0 + self.ν) * (1.0 - 2.0 * self.ν))

        if self.rank == 0:
            print(
                f"mu = {self.μ} lambda = {self.λ} E = {self.E} nu = {self.ν} density = {self.rho.value}"
            )

        # time step
        self.dt_st = dolfinx.fem.Constant(domain.structure.msh, (params.structure.dt))

        def _north_east_corner(x):
            eps = 1.0e-6

            # TODO: allow the probing of (x,y,z) points as something specified in the yaml file
            if params.general.geometry_module == "flag2d":
                # TEMP Hack for Turek and Hron Flag
                x1 = 0.6
                x2 = 0.2
                corner = [x1, x2]

                east_edge = np.logical_and(
                    corner[0] - eps < x[0], x[0] < corner[0] + eps
                )
                north_edge = np.logical_and(
                    corner[1] - eps < x[1], x[1] < corner[1] + eps
                )

                north_east_corner = np.logical_and(east_edge, north_edge)

            else:
                new_method = True

                if not new_method:
                    tracker_angle_rad = np.radians(params.pv_array.tracker_angle)
                    x1 = 0.5 * params.pv_array.panel_chord * np.cos(tracker_angle_rad)
                    x2 = (
                        0.5
                        * params.pv_array.panel_thickness
                        * np.sin(tracker_angle_rad)
                    )
                    corner = [x1 - x2, 0.5 * params.pv_array.panel_span]

                    east_edge = np.logical_and(
                        corner[0] - eps < x[0], x[0] < corner[0] + eps
                    )
                    north_edge = np.logical_and(
                        corner[1] - eps < x[1], x[1] < corner[1] + eps
                    )

                    north_east_corner = np.logical_and(east_edge, north_edge)

                else:
                    tracker_angle_rad = np.radians(params.pv_array.tracker_angle)

                    Ry = np.array(
                        [
                            [np.cos(tracker_angle_rad), 0.0, np.sin(tracker_angle_rad)],
                            [0.0, 1.0, 0.0],
                            [
                                -np.sin(tracker_angle_rad),
                                0.0,
                                np.cos(tracker_angle_rad),
                            ],
                        ]
                    )

                    array_rotation = (params.fluid.wind_direction + 90.0) % 360.0
                    array_rotation_rad = np.radians(array_rotation)

                    Rz = np.array(
                        [
                            [
                                np.cos(array_rotation_rad),
                                -np.sin(array_rotation_rad),
                                0.0,
                            ],
                            [
                                np.sin(array_rotation_rad),
                                np.cos(array_rotation_rad),
                                0.0,
                            ],
                            [0.0, 0.0, 1.0],
                        ]
                    )

                    reference_position = np.array(
                        [
                            0.5 * params.pv_array.panel_chord,
                            0.5 * params.pv_array.panel_span,
                            -0.5 * params.pv_array.panel_thickness,
                        ]
                    )

                    rotated_position = np.dot(reference_position, Ry.T)
                    rotated_position = np.dot(rotated_position, Rz.T)

                    final_position = rotated_position + np.array(
                        [0.0, 0.0, params.pv_array.elevation]
                    )

                    if params.rank == 0:
                        print(
                            f"Measuring panel deformation at (x, y, z) position {final_position}"
                        )

                    eps = 1.0e-4
                    near_x = np.logical_and(
                        final_position[0] - eps < x[0], x[0] < final_position[0] + eps
                    )
                    near_y = np.logical_and(
                        final_position[1] - eps < x[1], x[1] < final_position[1] + eps
                    )
                    near_z = np.logical_and(
                        final_position[2] - eps < x[2], x[2] < final_position[2] + eps
                    )

                    north_east_corner = np.logical_and(
                        near_x, np.logical_and(near_y, near_z)
                    )

            return north_east_corner

        north_east_corner_facets = dolfinx.mesh.locate_entities_boundary(
            domain.structure.msh, 0, _north_east_corner
        )

        self.north_east_corner_dofs = dolfinx.fem.locate_dofs_topological(
            self.V, 0, north_east_corner_facets
        )

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

    def update_a(self, u, u_old, v_old, a_old, dt, beta, ufl=True):
        # Update formula for acceleration
        # a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
        if ufl:
            dt_ = dt
            beta_ = beta
        else:
            dt_ = float(dt)
            beta_ = float(beta)
        return (u - u_old - dt_ * v_old) / beta_ / dt_**2 - (
            1 - 2 * beta_
        ) / 2 / beta_ * a_old

    # Update formula for velocity
    # v = dt * ((1-gamma)*a0 + gamma*a) + v0
    def update_v(self, a, u_old, v_old, a_old, dt, gamma, ufl=True):
        if ufl:
            dt_ = dt
            gamma_ = gamma
        else:
            dt_ = float(dt)
            gamma_ = float(gamma)
        return v_old + dt_ * ((1 - gamma_) * a_old + gamma_ * a)

    def update_fields(self, u, u_old, v_old, a_old, dt, beta, gamma):
        """Update fields at the end of each time step."""

        u_vec, u0_vec = u.x.array[:], u_old.x.array[:]
        v0_vec, a0_vec = v_old.x.array[:], a_old.x.array[:]

        a_vec = self.update_a(u_vec, u0_vec, v0_vec, a0_vec, dt, beta, ufl=False)
        v_vec = self.update_v(a_vec, u0_vec, v0_vec, a0_vec, dt, gamma, ufl=False)
        v_old.x.array[:] = v_vec
        a_old.x.array[:] = a_vec
        u_old.x.array[:] = u_vec

    def avg(self, x_old, x_new, alpha):
        return alpha * x_old + (1 - alpha) * x_new

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

        # Define trial and test functions for deformation
        # self.du = ufl.TrialFunction(self.V)
        self.u_ = ufl.TestFunction(self.V)

        P3 = ufl.TensorElement("Lagrange", domain.structure.msh.ufl_cell(), 2)
        self.T = dolfinx.fem.FunctionSpace(domain.structure.msh, P3)

        self.stress = dolfinx.fem.Function(self.T, name="stress_fluid")
        self.stress_old = dolfinx.fem.Function(self.T, name="stress_fluid_old")
        self.stress_predicted = dolfinx.fem.Function(
            self.T, name="stress_fluid_predicted"
        )

        self.sigma_vm_h = dolfinx.fem.Function(self.W, name="Stress")

        # discplacement
        self.u = dolfinx.fem.Function(self.V, name="Deformation")
        self.u_old = dolfinx.fem.Function(self.V, name="Deformation_old")
        self.u_delta = dolfinx.fem.Function(self.V, name="Deformation_change")

        # velocity
        self.v = dolfinx.fem.Function(self.V, name="Velocity")
        self.v_old = dolfinx.fem.Function(self.V, name="Velocity_old")

        # acceleration
        self.a = dolfinx.fem.Function(self.V, name="acceleration")
        self.a_old = dolfinx.fem.Function(self.V, name="acceleration_old")

        # dss = ufl.ds(subdomain_data=boundary_subdomains)

        # # Stress tensor
        # def sigma(r):
        #     return dolfinx.fem.form(2.0*self.μ*ufl.sym(ufl.grad(r)) + self.λ *ufl.tr(ufl.sym(ufl.grad(r)))*ufl.Identity(len(r)))

        # # Mass form
        # def m(u, u_):
        #     return dolfinx.fem.form(self.rho*ufl.inner(u, u_)*ufl.dx)

        # # Elastic stiffness form
        # def k(u, u_):
        #     return dolfinx.fem.form(ufl.inner(sigma(u), ufl.sym(ufl.grad(u_)))*ufl.dx)

        # # Rayleigh damping form
        # def c(u, u_):
        #     return dolfinx.fem.form(self.eta_m*m(u, u_) + self.eta_k*k(u, u_))

        # # Work of external forces
        # def Wext(u_):
        #     return ufl.dot(u_, self.f)*self.ds #dss(3)

        def sigma(r):
            return self.λ * ufl.nabla_div(r) * ufl.Identity(
                len(r)
            ) + 2 * self.μ * ufl.sym(ufl.grad(r))

        def m(u, u_):
            return self.rho * ufl.inner(u, u_) * ufl.dx

        def k(u, u_):
            return ufl.inner(sigma(u), ufl.grad(u_)) * ufl.dx

        def k(u, u_):
            # return ufl.inner(sigma(u),ufl.grad(u_))*ufl.dx
            # updated lagrangian form
            F = ufl.grad(u) + ufl.Identity(len(u))
            E = 0.5 * (F.T * F - ufl.Identity(len(u)))
            # S = self.λ *ufl.tr(E)*ufl.Identity(len(u))   + 2*self.μ * (E - ufl.tr(E)*ufl.Identity(len(u))  /3.0)
            S = self.λ * ufl.tr(E) * ufl.Identity(len(u)) + 2 * self.μ * (E)

            # return ufl.inner(F * S, ufl.grad(u_)) * ufl.dx
            return ufl.inner(P_(u), ufl.grad(u_)) * ufl.dx

        # The deformation gradient, F = I + dy/dX
        def F_(u):
            I = ufl.Identity(len(u))
            return I + ufl.grad(u)

        # The Cauchy-Green deformation tensor, C = F.T * F
        def C_(u):
            F = F_(u)
            return F.T * F

        # Green–Lagrange strain tensor, E = 0.5*(C - I)
        def E_(u):
            I = ufl.Identity(len(u))
            C = C_(u)

            return 0.5 * (C - I)
            # return 0.5 * (ufl.grad(u) + ufl.grad(u).T)

        # The second Piola–Kirchhoff stress, S
        def S_(u):
            E = E_(u)
            I = ufl.Identity(len(u))

            # return lamda * ufl.tr(E) * I + 2.0 * mu * (E - ufl.tr(E) * I / 3.0)
            # TODO: Why does the above form give a better result and where does it come from?

            S_svk = self.λ * ufl.tr(E) * I + 2.0 * self.μ * E
            return S_svk

        # The first Piola–Kirchhoff stress tensor, P = F*S
        def P_(u):
            F = F_(u)
            S = S_(u)
            # return ufl.inv(F) * S
            return F * S

        def c(u, u_):
            return self.eta_m * m(u, u_) + self.eta_k * k(u, u_)

        # self.uh_exp = dolfinx.fem.Function(self.V,  name="Deformation")

        def σ(v):
            """Return an expression for the stress σ given a displacement field"""
            return 2.0 * self.μ * ufl.sym(ufl.grad(v)) + self.λ * ufl.tr(
                ufl.sym(ufl.grad(v))
            ) * ufl.Identity(len(v))

        # source term ($f = \rho \omega^2 [x_0, \, x_1]$)
        # self.ω, self.ρ = 300.0, 10.0
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

        # Residual
        a_new = self.update_a(
            self.u, self.u_old, self.v_old, self.a_old, self.dt_st, self.beta, ufl=True
        )
        v_new = self.update_v(
            a_new, self.u_old, self.v_old, self.a_old, self.dt_st, self.gamma, ufl=True
        )

        F = ufl.grad(self.u) + ufl.Identity(len(self.u))
        J = ufl.det(F)
        self.res = (
            m(self.avg(self.a_old, a_new, self.alpha_m), self.u_)
            + c(self.avg(self.v_old, v_new, self.alpha_f), self.u_)
            + k(self.avg(self.u_old, self.u, self.alpha_f), self.u_)
            - self.rho * ufl.inner(self.f, self.u_) * ufl.dx
            - ufl.dot(ufl.dot(self.stress_predicted * J * ufl.inv(F.T), n), self.u_)
            * self.ds
        )  # - Wext(self.u)

        # self.a = dolfinx.fem.form(ufl.lhs(res))
        # self.L = dolfinx.fem.form(ufl.rhs(res))

        # self.a = dolfinx.fem.form(ufl.inner(σ(self.u), ufl.grad(self.v)) * ufl.dx)
        # self.L = dolfinx.fem.form(
        #     ufl.dot(self.f, self.v) * ufl.dx
        #     + ufl.dot(ufl.dot(self.stress, n), self.v) * self.ds
        # )

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

        if self.first_call_to_solver:
            self.problem = dolfinx.fem.petsc.NonlinearProblem(self.res, self.u, self.bc)
            self.solver = dolfinx.nls.petsc.NewtonSolver(self.comm, self.problem)
            self.solver.atol = 1e-8
            self.solver.rtol = 1e-8
            # self.solver.relaxation_parameter = 0.5
            # self.solver.max_it = 500
            # self.solver.convergence_criterion = "residual"
            self.solver.convergence_criterion = "incremental"

            # We can customize the linear solver used inside the NewtonSolver by
            # modifying the PETSc options
            ksp = self.solver.krylov_solver
            opts = PETSc.Options()
            option_prefix = ksp.getOptionsPrefix()

            opts[f"{option_prefix}ksp_type"] = "preonly"
            opts[f"{option_prefix}pc_type"] = "lu"

            # # opts[f"{option_prefix}ksp_type"] = "cg"
            # # opts[f"{option_prefix}pc_type"] = "gamg"
            # # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"

            ksp.setFromOptions()
        # self.A = dolfinx.fem.petsc.assemble_matrix(self.a, bcs=self.bc)
        # self.A.assemble()
        # self.b = dolfinx.fem.petsc.assemble_vector(self.L)

        # if self.first_call_to_solver:
        #     # Set solver options
        #     opts = PETSc.Options()
        #     opts["ksp_type"] = "cg"
        #     opts["ksp_rtol"] = 1.0e-6
        #     opts["pc_type"] = "gamg"

        #     # Use Chebyshev smoothing for multigrid
        #     opts["mg_levels_ksp_type"] = "chebyshev"
        #     opts["mg_levels_pc_type"] = "jacobi"

        #     # Improve estimate of eigenvalues for Chebyshev smoothing
        #     opts["mg_levels_esteig_ksp_type"] = "cg"
        #     opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10

        #     # Create PETSc Krylov solver and turn convergence monitoring on
        #     self.solver = PETSc.KSP().create(self.comm)
        #     self.solver.setFromOptions()

        # # Set matrix operator
        # self.solver.setOperators(self.A)

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
        # def σ(v):
        #     """Return an expression for the stress σ given a displacement field"""
        #     return 2.0 * self.μ * ufl.sym(ufl.grad(v)) + self.λ * ufl.tr(
        #         ufl.sym(ufl.grad(v))
        #     ) * ufl.Identity(len(v))

        if self.first_call_to_solver:
            if self.rank == 0:
                print("Starting Strutural Solution")

            self._assemble_system(params)

        num_its, converged = self.solver.solve(self.u)  # solve the current time step
        assert converged
        self.u.x.scatter_forward()

        # Calculate the change in the displacement (new - old) this is what moves the mesh
        self.u_delta.vector.array[:] = (
            self.u.vector.array[:] - self.u_old.vector.array[:]
        )
        self.u_delta.x.scatter_forward()

        # Update old fields with new quantities
        self.update_fields(
            self.u,
            self.u_old,
            self.v_old,
            self.a_old,
            self.dt_st,
            self.beta,
            self.gamma,
        )

        # sigma_dev = σ(self.u) - (1 / 3) * ufl.tr(σ(self.u)) * ufl.Identity(len(self.u))
        # sigma_vm = ufl.sqrt((3 / 2) * ufl.inner(sigma_dev, sigma_dev))

        # sigma_vm_expr = dolfinx.fem.Expression(
        #     sigma_vm, self.W.element.interpolation_points()
        # )
        # self.sigma_vm_h.interpolate(sigma_vm_expr)

        # self.unorm = self.u.x.norm()

        try:
            idx = self.north_east_corner_dofs[0]
            nw_corner_accel = self.u.vector.array[3 * idx : 3 * idx + 3].astype(
                np.float64
            )
            print(nw_corner_accel)
        except:
            nw_corner_accel = np.zeros(3, dtype=np.float64)

        nw_corner_accel_global = np.zeros((self.num_procs, 3), dtype=np.float64)

        self.comm.Gather(nw_corner_accel, nw_corner_accel_global, root=0)

        # print(f"Acceleration at North West corner = {nw_corner_accel}")

        if self.rank == 0:
            norm2 = np.sum(nw_corner_accel_global**2, axis=1)
            max_norm2_idx = np.argmax(norm2)
            np_accel = nw_corner_accel_global[max_norm2_idx, :]

            accel_pos_filename = os.path.join(
                params.general.output_dir_sol, "accel_pos.csv"
            )

            if self.first_call_to_solver:

                with open(accel_pos_filename, "w") as fp:
                    fp.write("#x-pos,y-pos,z-pos\n")
                    if self.ndim == 3:
                        fp.write(f"{np_accel[0]},{np_accel[1]},{np_accel[2]}\n")
                    elif self.ndim == 2:
                        fp.write(f"{np_accel[0]},{np_accel[1]}\n")

            else:
                with open(accel_pos_filename, "a") as fp:
                    if self.ndim == 3:
                        fp.write(f"{np_accel[0]},{np_accel[1]},{np_accel[2]}\n")
                    elif self.ndim == 2:
                        fp.write(f"{np_accel[0]},{np_accel[1]}\n")

        if self.first_call_to_solver:
            self.first_call_to_solver = False

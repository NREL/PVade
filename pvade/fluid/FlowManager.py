"""Summary
"""

import dolfinx
import ufl

from petsc4py import PETSc
from mpi4py import MPI

import numpy as np
import scipy.interpolate as interp

import warnings

from pvade.fluid.boundary_conditions import (
    build_velocity_boundary_conditions,
    build_pressure_boundary_conditions,
)


class Flow:
    """This class solves the CFD problem"""

    def __init__(self, domain, fluid_analysis):
        """Initialize the fluid solver

        This method initialize the Flow object, namely, it creates all the
        necessary function spaces on the mesh, initializes key counting and
        boolean variables and records certain characteristic quantities like
        the minimum cell size and the number of degrees of freedom attributed
        to both the pressure and velocity function spaces.

        Args:
            domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object

        """
        self.fluid_analysis = fluid_analysis
        self.name = "fluid"

        if fluid_analysis == False:
            pass
        else:
            # Store the comm and mpi info for convenience
            self.comm = domain.comm
            self.rank = domain.rank
            self.num_procs = domain.num_procs

            # Pressure (Scalar)
            P1 = ufl.FiniteElement("Lagrange", domain.fluid.msh.ufl_cell(), 1)
            self.Q = dolfinx.fem.FunctionSpace(domain.fluid.msh, P1)

            # Velocity (Vector)
            P2 = ufl.VectorElement("Lagrange", domain.fluid.msh.ufl_cell(), 2)
            self.V = dolfinx.fem.FunctionSpace(domain.fluid.msh, P2)

            # Stress (Tensor)
            P3 = ufl.TensorElement("Lagrange", domain.fluid.msh.ufl_cell(), 2)
            self.T = dolfinx.fem.FunctionSpace(domain.fluid.msh, P3)
            self.T_undeformed = dolfinx.fem.FunctionSpace(
                domain.fluid_undeformed.msh, P3
            )

            P4 = ufl.FiniteElement("DG", domain.fluid.msh.ufl_cell(), 0)

            self.DG = dolfinx.fem.FunctionSpace(domain.fluid.msh, P4)

            self.first_call_to_solver = True
            self.first_call_to_surface_pressure = True

            # Store the dimension of the problem for convenience
            self.ndim = domain.fluid.msh.topology.dim
            self.facet_dim = self.ndim - 1

            # find hmin in mesh
            num_cells = domain.fluid.msh.topology.index_map(self.ndim).size_local
            h = dolfinx.cpp.mesh.h(domain.fluid.msh, self.ndim, range(num_cells))

            # This value of hmin is local to the mesh portion owned by the process
            hmin_local = np.amin(h)

            # collect the minimum hmin from all processes
            self.hmin = np.zeros(1)
            self.comm.Allreduce(hmin_local, self.hmin, op=MPI.MIN)
            self.hmin = self.hmin[0]

            self.num_Q_dofs = (
                self.Q.dofmap.index_map_bs * self.Q.dofmap.index_map.size_global
            )
            self.num_V_dofs = (
                self.V.dofmap.index_map_bs * self.V.dofmap.index_map.size_global
            )

            if self.rank == 0:
                print(f"hmin on fluid  = {self.hmin}")
                print(f"Total num dofs on fluid= {self.num_Q_dofs + self.num_V_dofs}")

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

        self.bcu, self.inflow_profile, self.inflow_velocity, self.upper_cells = (
            build_velocity_boundary_conditions(domain, params, self.V, current_time=0.0)
        )

        self.bcp = build_pressure_boundary_conditions(domain, params, self.Q)
        # self.bcp = []

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
        # Define fluid properties
        self.dpdx = dolfinx.fem.Constant(domain.fluid.msh, (0.0, 0.0, 0.0))
        self.dt_c = dolfinx.fem.Constant(domain.fluid.msh, (params.solver.dt))
        self.rho_c = dolfinx.fem.Constant(domain.fluid.msh, (params.fluid.rho))
        nu = dolfinx.fem.Constant(domain.fluid.msh, (params.fluid.nu))

        # Define trial and test functions for velocity
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        # Define trial and test functions for pressure
        self.p = ufl.TrialFunction(self.Q)
        self.q = ufl.TestFunction(self.Q)

        # Define functions for velocity solutions
        self.u_k = dolfinx.fem.Function(self.V, name="velocity ")
        self.u_k1 = dolfinx.fem.Function(self.V)
        self.u_k2 = dolfinx.fem.Function(self.V)

        self.mesh_vel = dolfinx.fem.Function(self.V, name="mesh_displacement")
        self.mesh_vel_old = dolfinx.fem.Function(self.V)
        self.mesh_vel_bc = dolfinx.fem.Function(self.V, name="mesh_displacement_bc")
        self.mesh_vel_bc_old = dolfinx.fem.Function(self.V)
        # self.mesh_vel_composite = dolfinx.fem.Function(self.V)

        # Define functions for pressure solutions
        self.p_k = dolfinx.fem.Function(self.Q, name="pressure")
        self.p_k1 = dolfinx.fem.Function(self.Q)

        if params.fluid.initialize_with_inflow_bc:
            # self.inflow_profile = dolfinx.fem.Function(self.V)
            # self.inflow_profile.interpolate(lambda x: np.vstack((x[0], x[1],x[2])))

            # when using interpolate the assert fails
            self.u_k1.interpolate(self.inflow_profile)
            self.u_k2.interpolate(self.inflow_profile)
            self.u_k.interpolate(self.inflow_profile)

            # print(min(abs(self.u_k.x.array[:] - self.inflow_profile.x.array[:])))

            # flags = []

            # flags.append((self.u_k.x.array[:] == self.inflow_profile.x.array).all())
            # flags.append((self.u_k.x.array[:] == self.inflow_profile.x.array).all())
            # flags.append((self.u_k2.x.array[:] == self.inflow_profile.x.array).all())

            # assert all(flags), "initialiazation not done correctly"

        # Define expressions used in variational forms
        # Crank-Nicolson velocity
        U_CN = 0.5 * (self.u + self.u_k1)

        # Adams-Bashforth velocity
        U_AB = 1.5 * self.u_k1 - 0.5 * self.u_k2

        if params.fluid.turbulence_model is not None:
            if params.fluid.turbulence_model == "smagorinsky":
                # By default, don't use any eddy viscosity
                filter_scale = ufl.CellVolume(domain.fluid.msh) ** (
                    1.0 / domain.fluid.msh.topology.dim
                )

                # Strain rate tensor, 0.5*(du_i/dx_j + du_j/dx_i)
                Sij = ufl.sym(ufl.nabla_grad(U_AB))

                # ufl.sqrt(Sij*Sij)
                strainMag = (2.0 * ufl.inner(Sij, Sij)) ** 0.5

                # Smagorinsky dolfinx.fem.constant, typically close to 0.17
                Cs = params.fluid.c_s

                # Eddy viscosity
                self.nu_T = Cs**2 * filter_scale**2 * strainMag
                # self.nu_T = dolfinx.fem.Constant(domain.fluid.msh, 0.0)
            else:
                raise ValueError(
                    f"Turbulence model {params.fluid.turbulence_model} not recognized."
                )
        else:
            self.nu_T = dolfinx.fem.Constant(domain.fluid.msh, 0.0)

        # ================================================================#
        # DEFINE VARIATIONAL FORMS
        # ================================================================#
        U = 0.5 * (self.u_k1 + self.u)

        def epsilon(u):
            """Convenience expression for ufl.sym(ufl.nabla_grad(u))

            Args:
                u (dolfinx.fem.Function): A dolfinx function

            Returns:
                ufl.dolfinx.fem.form: ufl.sym(ufl.nabla_grad(u))
            """
            return ufl.sym(ufl.nabla_grad(u))

        # Define stress tensor
        def sigma(u, p, nu, rho):
            """Convenience expression for fluid stress, sigma

            Args:
                u (dolfinx.fem.Function): Velocity
                p (dolfinx.fem.Function): Pressure
                nu (float, dolfinx.fem.Function): Viscosity

            Returns:
                ufl.dolfinx.fem.form: Stress in fluid, $2\nu \epsilon (u)$
            """
            return (nu * rho) * (ufl.grad(u) + ufl.grad(u).T) - p * ufl.Identity(len(u))
            # return 2 * nu * rho * epsilon(u) - p * ufl.Identity(len(u))

        fractional_step_scheme = "IPCS"
        U = 0.5 * (self.u_k1 + self.u)
        f = dolfinx.fem.Constant(
            domain.fluid.msh,
            (PETSc.ScalarType(0), PETSc.ScalarType(0), PETSc.ScalarType(0)),
        )
        # Define variational problem for step 1: tentative velocity
        # self.U_ALE = (1.5*self.mesh_vel - 0.5*self.mesh_vel_old) / self.dt_c
        # self.U_ALE = 0.5*(domain.fluid_mesh_displacement + domain.fluid_mesh_displacement_old) / self.dt_c
        # self.U_ALE = self.mesh_vel_old / self.dt_c
        # self.U_ALE = 1.5 * self.mesh_vel -0.5 * self.mesh_vel_old  # / self.dt_c
        self.U_ALE = 0.5 * (self.mesh_vel + self.mesh_vel_old)  # / self.dt_c
        # self.U_ALE = self.mesh_vel  # / self.dt_c
        # self.U_ALE = 0.1*self.mesh_vel + 0.9*self.mesh_vel_old
        # self.U_ALE = domain.fluid_mesh_displacement / self.dt_c

        self.F1 = (
            (1.0 / self.dt_c) * ufl.inner(self.u - self.u_k1, self.v) * ufl.dx
            + ufl.inner(
                ufl.dot(
                    U_AB - self.U_ALE,
                    ufl.nabla_grad(U_CN),
                ),
                self.v,
            )
            * ufl.dx
            + (nu + self.nu_T) * ufl.inner(ufl.grad(U_CN), ufl.grad(self.v)) * ufl.dx
            + (1.0 / self.rho_c) * ufl.inner(ufl.grad(self.p_k1), self.v) * ufl.dx
            - (1.0 / self.rho_c) * ufl.inner(self.dpdx, self.v) * ufl.dx
        )

        self.a1 = dolfinx.fem.form(ufl.lhs(self.F1))
        self.L1 = dolfinx.fem.form(ufl.rhs(self.F1))

        # Define variational problem for step 2: pressure correction
        self.a2 = dolfinx.fem.form(
            ufl.dot(ufl.nabla_grad(self.p), ufl.nabla_grad(self.q)) * ufl.dx
        )
        self.L2 = dolfinx.fem.form(
            ufl.dot(ufl.nabla_grad(self.p_k1), ufl.nabla_grad(self.q)) * ufl.dx
            - (self.rho_c / self.dt_c) * ufl.div(self.u_k) * self.q * ufl.dx
        )

        # Define variational problem for step 3: velocity update
        self.a3 = dolfinx.fem.form(ufl.dot(self.u, self.v) * ufl.dx)
        self.L3 = dolfinx.fem.form(
            ufl.dot(self.u_k, self.v) * ufl.dx
            - (self.dt_c / self.rho_c)
            * ufl.dot(ufl.nabla_grad(self.p_k - self.p_k1), self.v)
            * ufl.dx
        )

        # Define a function and the dolfinx.fem.form of the stress
        self.panel_stress = dolfinx.fem.Function(self.T)
        self.panel_stress_undeformed = dolfinx.fem.Function(self.T_undeformed)
        self.stress = sigma(self.u_k, self.p_k, nu + self.nu_T, self.rho_c)

        # Define mesh normals
        self.n = ufl.FacetNormal(domain.fluid.msh)

        # Compute traction vector
        self.traction = ufl.dot(self.stress, -self.n)

        # Create a dolfinx.fem.form for projecting stress onto a tensor function space
        # e.g., panel_stress.assign(project(stress, T))
        self.a4 = dolfinx.fem.form(
            ufl.inner(ufl.TrialFunction(self.T), ufl.TestFunction(self.T)) * ufl.dx
        )
        self.L4 = dolfinx.fem.form(
            ufl.inner(self.stress, ufl.TestFunction(self.T)) * ufl.dx
        )

        # Create a dolfinx.fem.form for projecting CFL calculation onto DG function space
        cell_diam = ufl.CellDiameter(domain.fluid.msh)
        cfl_form = ufl.sqrt(ufl.inner(self.u_k, self.u_k)) * self.dt_c / cell_diam

        self.cfl_vec = dolfinx.fem.Function(self.DG)
        self.a5 = dolfinx.fem.form(
            ufl.inner(ufl.TrialFunction(self.DG), ufl.TestFunction(self.DG)) * ufl.dx
        )
        self.L5 = dolfinx.fem.form(
            ufl.inner(cfl_form, ufl.TestFunction(self.DG)) * ufl.dx
        )

        # Set up the functions to ensure a dolfinx.fem.constant flux through the outlet
        outlet_cells = dolfinx.mesh.locate_entities(
            domain.fluid.msh, self.ndim, lambda x: x[0] > params.domain.x_max - 1
        )
        self.flux_plane = dolfinx.fem.Function(self.V)
        self.flux_plane.interpolate(
            lambda x: (np.ones(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1])),
            outlet_cells,
        )

        # self.flux_plane = Expression('x[0] < cutoff ? 0.0 : 1.0', cutoff=domain.x_range[1]-1.0, degree=1)
        # self.flux_dx = ufl.Measure("dx", domain=domain.fluid.msh)  # not needed ?

        # self.vol = dolfinx.fem.petsc.assemble_matrix(self.flux_plane*self.flux_dx)
        # form1 = dolfinx.fem.form(self.flux_plane*self.flux_dx)
        # self.vol = dolfinx.fem.petsc.assemble_vector(form1)
        # self.J_initial = float(fem.petsc.assemble_matrix(self.flux_plane*self.u_k[0]*self.flux_dx)/self.vol)

        # if mpi_info['rank'] == 0:
        #     print('J_initial = %f' % (self.J_initial))

        # self.J_history = [self.J_initial]
        self.dpdx_history = [0.0]

        def _all_interior_surfaces(x):
            eps = 1.0e-5

            x_mid = np.logical_and(
                params.domain.x_min + eps < x[0], x[0] < params.domain.x_max - eps
            )
            y_mid = np.logical_and(
                params.domain.y_min + eps < x[1], x[1] < params.domain.y_max - eps
            )
            z_mid = np.logical_and(
                params.domain.z_min + eps < x[2], x[2] < params.domain.z_max - eps
            )

            all_interior_surfaces = np.logical_and(x_mid, np.logical_and(y_mid, z_mid))

            return all_interior_surfaces

        all_interior_facets = dolfinx.mesh.locate_entities_boundary(
            domain.fluid.msh, self.facet_dim, _all_interior_surfaces
        )

        self.all_interior_V_dofs = dolfinx.fem.locate_dofs_topological(
            self.V, self.facet_dim, all_interior_facets
        )

        self.zero_vec = dolfinx.fem.Constant(
            domain.fluid.msh, PETSc.ScalarType((0.0, 0.0, 0.0))
        )

        ds_fluid = ufl.Measure(
            "ds", domain=domain.fluid.msh, subdomain_data=domain.fluid.facet_tags
        )

        self.lift_form_list = []
        self.drag_form_list = []

        for panel_id in range(
            int(params.pv_array.stream_rows * params.pv_array.span_rows)
        ):
            self.lift_form_list.append(0)
            self.lift_form_list[-1] += self.traction[1] * ds_fluid(
                domain.domain_markers[f"left_{panel_id:.0f}"]["idx"]
            )
            self.lift_form_list[-1] += self.traction[1] * ds_fluid(
                domain.domain_markers[f"top_{panel_id:.0f}"]["idx"]
            )
            self.lift_form_list[-1] += self.traction[1] * ds_fluid(
                domain.domain_markers[f"right_{panel_id:.0f}"]["idx"]
            )
            self.lift_form_list[-1] += self.traction[1] * ds_fluid(
                domain.domain_markers[f"bottom_{panel_id:.0f}"]["idx"]
            )
            if self.ndim == 3:
                self.lift_form_list[-1] += self.traction[1] * ds_fluid(
                    domain.domain_markers[f"front_{panel_id:.0f}"]["idx"]
                )
                self.lift_form_list[-1] += self.traction[1] * ds_fluid(
                    domain.domain_markers[f"back_{panel_id:.0f}"]["idx"]
                )

            self.lift_form_list[-1] = dolfinx.fem.form(self.lift_form_list[-1])

            # for loc in ["top_0", "bottom_0", "left_0", "right_0"]:
            #     idx = domain.domain_markers[loc]["idx"]
            #     s = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0*ds_fluid(idx)))
            #     print(f"loc = {loc}, idx = {idx}, s = {s}")

            self.drag_form_list.append(0)
            self.drag_form_list[-1] += self.traction[0] * ds_fluid(
                domain.domain_markers[f"left_{panel_id:.0f}"]["idx"]
            )
            self.drag_form_list[-1] += self.traction[0] * ds_fluid(
                domain.domain_markers[f"top_{panel_id:.0f}"]["idx"]
            )
            self.drag_form_list[-1] += self.traction[0] * ds_fluid(
                domain.domain_markers[f"right_{panel_id:.0f}"]["idx"]
            )
            self.drag_form_list[-1] += self.traction[0] * ds_fluid(
                domain.domain_markers[f"bottom_{panel_id:.0f}"]["idx"]
            )
            if self.ndim == 3:
                self.drag_form_list[-1] += self.traction[0] * ds_fluid(
                    domain.domain_markers[f"front_{panel_id:.0f}"]["idx"]
                )
                self.drag_form_list[-1] += self.traction[0] * ds_fluid(
                    domain.domain_markers[f"back_{panel_id:.0f}"]["idx"]
                )

            self.drag_form_list[-1] = dolfinx.fem.form(self.drag_form_list[-1])

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

        self.A1 = dolfinx.fem.petsc.assemble_matrix(self.a1, bcs=self.bcu)
        self.A2 = dolfinx.fem.petsc.assemble_matrix(self.a2, bcs=self.bcp)
        self.A3 = dolfinx.fem.petsc.assemble_matrix(self.a3)
        self.A4 = dolfinx.fem.petsc.assemble_matrix(self.a4)
        self.A5 = dolfinx.fem.petsc.assemble_matrix(self.a5)

        self.A1.assemble()
        self.A2.assemble()
        self.A3.assemble()
        self.A4.assemble()
        self.A5.assemble()

        self.b1 = dolfinx.fem.petsc.assemble_vector(self.L1)
        self.b2 = dolfinx.fem.petsc.assemble_vector(self.L2)
        self.b3 = dolfinx.fem.petsc.assemble_vector(self.L3)
        self.b4 = dolfinx.fem.petsc.assemble_vector(self.L4)
        self.b5 = dolfinx.fem.petsc.assemble_vector(self.L5)

        self.solver_1 = PETSc.KSP().create(self.comm)
        self.solver_1.setOperators(self.A1)
        self.solver_1.setType(params.solver.solver1_ksp)
        self.solver_1.getPC().setType(params.solver.solver1_pc)
        self.solver_1.setFromOptions()

        self.solver_2 = PETSc.KSP().create(self.comm)
        self.solver_2.setOperators(self.A2)
        self.solver_2.setType(params.solver.solver2_ksp)
        self.solver_2.getPC().setType(params.solver.solver2_pc)
        self.solver_2.setFromOptions()

        self.solver_3 = PETSc.KSP().create(self.comm)
        self.solver_3.setOperators(self.A3)
        self.solver_3.setType(params.solver.solver3_ksp)
        self.solver_3.getPC().setType(params.solver.solver3_pc)
        self.solver_3.setFromOptions()

        self.solver_4 = PETSc.KSP().create(self.comm)
        self.solver_4.setOperators(self.A4)
        self.solver_4.setType(params.solver.solver4_ksp)
        self.solver_4.getPC().setType(params.solver.solver4_pc)
        self.solver_4.setFromOptions()

        self.solver_5 = PETSc.KSP().create(self.comm)
        self.solver_5.setOperators(self.A5)
        self.solver_5.setType("cg")
        self.solver_5.getPC().setType("jacobi")
        self.solver_5.setFromOptions()

    def solve(self, domain, params, current_time):
        """Solve for a single timestep advancement

        Here we perform the three-step solution process (tentative velocity,
        pressure correction, velocity update) to advance the fluid simulation
        a single timestep. Additionally, we calculate the new CFL number
        associated with the latest velocity solution.

        Args:
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object
        """

        if params.fluid.time_varying_inflow_bc:
            self.inflow_velocity.current_time = current_time
            if self.upper_cells is not None:
                self.inflow_profile.interpolate(self.inflow_velocity, self.upper_cells)
            else:
                self.inflow_profile.interpolate(self.inflow_velocity)

        if self.first_call_to_solver:
            if self.rank == 0:
                print("Starting Fluid Solution")

            self.bcu.append(
                dolfinx.fem.dirichletbc(self.mesh_vel, self.all_interior_V_dofs)
            )

            self._assemble_system(params)

        self.mesh_vel.interpolate(domain.fluid_mesh_displacement)
        # # self.mesh_vel.interpolate(domain.better_mesh_vel)

        self.mesh_vel.vector.array[:] = self.mesh_vel.vector.array / params.solver.dt
        self.mesh_vel.x.scatter_forward()

        # Calculate the tentative velocity
        self._solver_step_1(params)

        # Calculate the change in pressure to enforce incompressibility
        self._solver_step_2(params)

        # Update the velocity according to the pressure field
        self._solver_step_3(params)

        self._solver_step_4(params)
        # Compute the CFL number
        self.compute_cfl()

        self.compute_lift_and_drag(params, current_time)

        self.compute_pressure_drop_between_points(domain, params)

        # Update new -> old variables
        self.u_k2.x.array[:] = self.u_k1.x.array
        self.u_k1.x.array[:] = self.u_k.x.array
        self.p_k1.x.array[:] = self.p_k.x.array
        self.mesh_vel_old.x.array[:] = self.mesh_vel.x.array

        if self.first_call_to_solver:
            self.first_call_to_solver = False

    def _solver_step_1(self, params):
        """Solve step 1: tentative velocity

        Here we calculate the tentative velocity which, not guaranteed to be
        divergence free.

        Args:
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object
        """
        # Step 0: Re-assemble A1 since using an implicit convective term
        self.A1.zeroEntries()
        self.A1 = dolfinx.fem.petsc.assemble_matrix(self.A1, self.a1, bcs=self.bcu)
        self.A1.assemble()
        self.solver_1.setOperators(self.A1)

        # Step 1: Tentative velocity step
        with self.b1.localForm() as loc:
            loc.set(0)

        self.b1 = dolfinx.fem.petsc.assemble_vector(self.b1, self.L1)

        dolfinx.fem.petsc.apply_lifting(self.b1, [self.a1], [self.bcu])

        self.b1.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        )

        dolfinx.fem.petsc.set_bc(self.b1, self.bcu)

        self.solver_1.solve(self.b1, self.u_k.vector)
        self.u_k.x.scatter_forward()

    def _solver_step_2(self, params):
        """Solve step 2: pressure correction

        Here we calculate the pressure field that would be required to correct
        the tentative velocity such that it is divergence free.

        Args:
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object
        """
        # Step 0: Re-assemble A1 since using an implicit convective term
        # Step 2: Pressure correction step

        self.A2.zeroEntries()
        self.A2 = dolfinx.fem.petsc.assemble_matrix(self.A2, self.a2, bcs=self.bcp)
        self.A2.assemble()
        self.solver_2.setOperators(self.A2)

        with self.b2.localForm() as loc:
            loc.set(0)

        self.b2 = dolfinx.fem.petsc.assemble_vector(self.b2, self.L2)

        nullspace_testing = True

        if len(self.bcp) == 0 and nullspace_testing:
            if self.first_call_to_solver:
                # No pressure boundary conditions applied,
                # Therefore we need to remove the null space
                if params.rank == 0:
                    print("No pressure BC found, initializing null space")

                self.nullspace = PETSc.NullSpace().create(
                    constant=True, comm=params.comm
                )
                self.A2.setNullSpace(self.nullspace)

            self.nullspace.remove(self.b2)
        else:
            dolfinx.fem.petsc.apply_lifting(self.b2, [self.a2], [self.bcp])

            self.b2.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )

            dolfinx.fem.petsc.set_bc(self.b2, self.bcp)

        self.solver_2.solve(self.b2, self.p_k.vector)
        self.p_k.x.scatter_forward()

    def _solver_step_3(self, params):
        """Solve step 3: velocity update

        Here we update the tentative velocity with the effect of the modified,
        continuity-enforcing pressure field.

        Args:
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object
        """
        # Step 3: Velocity correction step

        self.A3.zeroEntries()
        self.A3 = dolfinx.fem.petsc.assemble_matrix(self.A3, self.a3)
        self.A3.assemble()
        self.solver_3.setOperators(self.A3)

        with self.b3.localForm() as loc:
            loc.set(0)

        self.b3 = dolfinx.fem.petsc.assemble_vector(self.b3, self.L3)

        self.b3.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        )

        self.solver_3.solve(self.b3, self.u_k.vector)
        self.u_k.x.scatter_forward()

    def _solver_step_4(self, params):
        """Solve step 3: velocity update

        Here we update the tentative velocity with the effect of the modified,
        continuity-enforcing pressure field.

        Args:
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object
        """
        self.A4.zeroEntries()
        self.A4 = dolfinx.fem.petsc.assemble_matrix(self.A4, self.a4)
        self.A4.assemble()
        self.solver_4.setOperators(self.A4)

        # Step 3: Velocity correction step
        with self.b4.localForm() as loc:
            loc.set(0)

        self.b4 = dolfinx.fem.petsc.assemble_vector(self.b4, self.L4)

        self.b4.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        )

        self.solver_4.solve(self.b4, self.panel_stress.vector)
        self.panel_stress.x.scatter_forward()
        self.panel_stress_undeformed.x.array[:] = self.panel_stress.x.array[:]
        self.panel_stress_undeformed.x.scatter_forward()

    def compute_cfl(self):
        """Solve for the CFL number

        Using the velocity, timestep size, and cell sizes, we calculate a CFL
        number at every mesh cell. From that, we select the single highest
        value of CFL number and record it for the purposes of monitoring
        simulation stability.

        Args:
            None
        """

        self.A5.zeroEntries()
        self.A5 = dolfinx.fem.petsc.assemble_matrix(self.A5, self.a5)
        self.A5.assemble()
        self.solver_5.setOperators(self.A5)

        with self.b5.localForm() as loc:
            loc.set(0)

        self.b5 = dolfinx.fem.petsc.assemble_vector(self.b5, self.L5)

        self.b5.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        )

        self.solver_5.solve(self.b5, self.cfl_vec.vector)
        self.cfl_vec.x.scatter_forward()

        cfl_max_local = np.amax(self.cfl_vec.vector.array)

        # collect the minimum hmin from all processes
        self.cfl_max = np.zeros(1)
        self.comm.Allreduce(cfl_max_local, self.cfl_max, op=MPI.MAX)
        self.cfl_max = self.cfl_max[0]

    def compute_lift_and_drag(self, params, current_time):

        self.lift_coeff_list = []
        self.drag_coeff_list = []

        for panel_id in range(params.pv_array.num_panels):
            lift_coeff_local = dolfinx.fem.assemble_scalar(
                self.lift_form_list[panel_id]
            )
            lift_coeff_array = np.zeros(self.num_procs, dtype=np.float64)
            self.comm.Gather(
                np.array(lift_coeff_local, dtype=np.float64), lift_coeff_array, root=0
            )

            drag_coeff_local = dolfinx.fem.assemble_scalar(
                self.drag_form_list[panel_id]
            )
            drag_coeff_array = np.zeros(self.num_procs, dtype=np.float64)
            self.comm.Gather(
                np.array(drag_coeff_local, dtype=np.float64), drag_coeff_array, root=0
            )

            if self.rank == 0:
                self.lift_coeff_list.append(np.sum(lift_coeff_array))
                self.drag_coeff_list.append(np.sum(drag_coeff_array))

        if self.rank == 0:
            if self.first_call_to_solver:
                self.lift_and_drag_filename = (
                    f"{params.general.output_dir_sol}/lift_and_drag.csv"
                )

                with open(self.lift_and_drag_filename, "w") as fp:
                    fp.write("#Time")

                    for panel_id in range(params.pv_array.num_panels):
                        fp.write(
                            f",Lift_{panel_id:.0f},Drag_{panel_id:.0f},Lift_ND_{panel_id:.0f},Drag_ND_{panel_id:.0f}"
                        )

                    fp.write("\n")

            with open(self.lift_and_drag_filename, "a") as fp:
                fp.write(f"{current_time:.9e}")

                for panel_id in range(params.pv_array.num_panels):

                    lift_coeff = self.lift_coeff_list[panel_id]
                    drag_coeff = self.drag_coeff_list[panel_id]

                    lift_coeff_nd = (
                        2.0
                        * lift_coeff
                        / (
                            params.fluid.rho
                            * params.fluid.u_ref**2
                            * 2.0
                            * params.pv_array.panel_span
                        )
                    )

                    drag_coeff_nd = (
                        2.0
                        * drag_coeff
                        / (
                            params.fluid.rho
                            * params.fluid.u_ref**2
                            * 2.0
                            * params.pv_array.panel_span
                        )
                    )

                    fp.write(
                        f",{lift_coeff:.9e},{drag_coeff:.9e},{lift_coeff_nd:.9e},{drag_coeff_nd:.9e}"
                    )

                    # print(f"Lift = {lift_coeff} ({lift_coeff_nd})")
                    # print(f"Drag = {drag_coeff} ({drag_coeff_nd})")

                fp.write("\n")

    def compute_pressure_drop_between_points(self, domain, params):
        bb_tree = dolfinx.geometry.BoundingBoxTree(
            domain.fluid.msh, domain.fluid.msh.topology.dim
        )

        eps = 1.0e-6

        # Find cells whose bounding-box collide with the the points
        points = np.array(
            [
                [0.2 - params.pv_array.panel_span - eps, 0.2, 0.0],
                [0.2 + params.pv_array.panel_span + eps, 0.2, 0.0],
            ]
        )

        cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, points)

        cells = []
        points_on_proc = []

        # Choose one of the cells that contains the point
        colliding_cells = dolfinx.geometry.compute_colliding_cells(
            domain.fluid.msh, cell_candidates, points
        )

        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        p_front_and_back = self.p_k.eval(points_on_proc, cells)

        if len(p_front_and_back) == 0:
            p_front_and_back = np.array(np.nan)

        p_front_and_back_global = self.comm.gather(p_front_and_back, root=0)
        if self.rank == 0:
            temp = [data.flatten() for data in p_front_and_back_global]
            p_front_and_back_global = np.concatenate(temp).ravel()
            delta_p = np.nanmax(p_front_and_back_global) - np.nanmin(
                p_front_and_back_global
            )
            # print(p_front_and_back_global)
            # print(f"delta_P = {delta_p}")

        # print(p_front)

    def adjust_dpdx_for_constant_flux(self):
        """Adjust the forcing term, ``dpdx``, to maintain flowrate

        Here we calculate what the value of the dolfinx.fem.constant driving
        pressure/force, ``dpdx``, should be adjusted to to maintain a
        dolfinx.fem.constant flux through the domain. This is a useful option if
        performing a periodic simulation in which the flux will slowly decay
        due to viscous dissipation. The amount of change in ``dpdx`` is
        calculated by comparing the current flux to the flux measured in the
        initial condition and then employing a PID controller to adjust the
        driving force to seek the target defined by the initial condition.

        Args:
            None
        """

        def pid_controller(J_init, J_history, dt):
            """Summary

            Args:
                J_init (TYPE): Description
                J_history (TYPE): Description
                dt (TYPE): Description

            Returns:
                TYPE: Description
            """
            assert type(J_history) is list

            # K_p = 0.1
            # K_i = 0.4
            # K_d = 0.001
            # K_p = 0.4
            # K_i = 0.4
            # K_d = 0.01
            K_p = 0.6
            K_i = 0.4
            K_d = 0.05

            err = J_init - np.array(J_history)

            c_p = K_p * err[-1]
            c_i = K_i * dt * np.sum(err)

            try:
                c_d = K_d * (err[-1] - err[-2]) / dt
            except:
                c_d = 0.0

            output = c_p + c_i + c_d

            return output

        # Compute the flow through the outflow flux plane
        # self.A1 = dolfinx.fem.petsc.assemble_matrix(self.a1
        J = float(
            dolfinx.fem.petsc.assemble_matrix(
                self.flux_plane * self.u_k[0] * self.flux_dx
            )
            / self.vol
        )
        self.J_history.append(J)

        dpdx_val = pid_controller(self.J_initial, self.J_history, float(self.dt_c))
        self.dpdx_history.append(dpdx_val)

        if dpdx_val < 0.0:
            print("WARNING: dpdx_val = %f" % (dpdx_val))

        self.dpdx.assign(dolfinx.fem.Constant((dpdx_val, 0.0, 0.0)))

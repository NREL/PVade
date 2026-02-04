"""Summary"""

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
        # self.structural_analysis = structural_analysis
        # self.name = "structure"

        # Store the comm and mpi info for convenience
        self.comm = domain.comm
        self.rank = domain.rank
        self.num_procs = domain.num_procs

        P1 = ufl.VectorElement("Lagrange", domain.structure.msh.ufl_cell(), 2)
        self.V = dolfinx.fem.FunctionSpace(domain.structure.msh, P1)

        self.W = dolfinx.fem.FunctionSpace(
            domain.structure.msh, ("Discontinuous Lagrange", 0)
        )

        self.first_call_to_solver = True

        self.num_V_dofs = (
            self.V.dofmap.index_map_bs * self.V.dofmap.index_map.size_global
        )

        # Rayleigh damping coefficients
        self.eta_m = dolfinx.fem.Constant(domain.structure.msh, 0.0)  # Constant(0.)
        self.eta_k = dolfinx.fem.Constant(domain.structure.msh, 0.0)  # Constant(0.)

        # Generalized-alpha method parameters
        self.alpha_m = dolfinx.fem.Constant(domain.structure.msh, 0.2)
        self.alpha_f = dolfinx.fem.Constant(domain.structure.msh, 0.4)
        self.gamma = 0.5 + self.alpha_f - self.alpha_m
        self.beta = (self.gamma + 0.5) ** 2 / 4.0

        # time step
        self.dt_st = dolfinx.fem.Constant(domain.structure.msh, (params.structure.dt))

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

    def calculate_K_for_Robin_BC(self, domain, flow, params):

        # shear modulus of tube
        Gt = params.structure.elasticity_modulus_tube / (
            2 * (1 + params.structure.poissons_ratio_tube)
        )

        Ipt = (
            np.pi
            / 2
            * (
                params.pv_array.torque_tube_outer_radius**4
                - params.pv_array.torque_tube_inner_radius**4
            )
        )

        Gp = params.structure.elasticity_modulus / (
            2 * (1 + params.structure.poissons_ratio)
        )

        Ipp = 1 / 3.0 * params.pv_array.panel_thickness * params.pv_array.panel_chord**3

        # total number of pv rows
        total_num_panels = params.pv_array.stream_rows * params.pv_array.span_rows

        for panel_id in range(total_num_panels):
            # construct the matrix, size: modules_per_span+1 by modules_per_span+1
            theo_matrix = np.zeros(
                (
                    params.pv_array.modules_per_span + 1,
                    params.pv_array.modules_per_span + 1,
                )
            )
            theo_vector = np.zeros(params.pv_array.modules_per_span + 1)

            connector_locations = np.linspace(
                0, params.pv_array.panel_span, params.pv_array.modules_per_span + 1
            )

            # ?? double integral is not available until the second time step. Should make it be zero for the
            # first time step then it will be updated once the first step is finished? but the assemble is out of
            # the time loop. Should we solve fluid then structure?

            total_torque_on_this_panel_name = f"total_torque_panel_{panel_id:.0f}"
            total_torque_on_this_panel = getattr(
                flow, total_torque_on_this_panel_name
            ).value

            theo_matrix[params.pv_array.modules_per_span, :] = 1 / Gp / Ipp
            theo_vector[params.pv_array.modules_per_span] = (
                -total_torque_on_this_panel / Gp / Ipp
            )

            # contribution from panel:

            # contribution from C0 to matrix:

            theo_matrix[
                : params.pv_array.modules_per_span, : params.pv_array.fixed_location
            ] += connector_locations[params.pv_array.fixed_location] / (Gp * Ipp)
            theo_matrix[
                : params.pv_array.modules_per_span, 1 : params.pv_array.fixed_location
            ] -= connector_locations[1 : params.pv_array.fixed_location] / (Gp * Ipp)

            # contribution from C0 to vector:
            T_double_integral_at_fixed_location_name = f"double_integral_total_torque_panel_{panel_id:.0f}_{params.pv_array.fixed_location:.0f}"
            T_double_integral_at_fixed_location = getattr(
                flow, T_double_integral_at_fixed_location_name
            ).value
            theo_vector[: params.pv_array.modules_per_span] += (
                -T_double_integral_at_fixed_location / Gp / Ipp
            )

            # contribution of panels to matrix
            theo_matrix[0, 0] += -connector_locations[0] / (Gp * Ipp)
            theo_matrix[1, 0] += -connector_locations[1] / (Gp * Ipp)

            for i in range(2, params.pv_array.fixed_location):
                theo_matrix[i, :i] += -connector_locations[i] / (Gp * Ipp)
                theo_matrix[i, 1:i] += connector_locations[1:i] / (Gp * Ipp)

            for i in range(
                params.pv_array.fixed_location, params.pv_array.modules_per_span
            ):
                theo_matrix[i, : i + 1] += -connector_locations[i + 1] / (Gp * Ipp)
                theo_matrix[i, 1 : i + 1] += connector_locations[1 : i + 1] / (Gp * Ipp)

                # contribution of panels to vector
            T_double_integral_array = []
            for i in range(params.pv_array.modules_per_span + 1):
                name = f"double_integral_total_torque_panel_{panel_id:.0f}_{i:.0f}"
                T_double_integral_array.append(getattr(flow, name).value)
            T_double_integral_array = np.array(T_double_integral_array)
            theo_vector[: params.pv_array.modules_per_span] += (
                np.delete(T_double_integral_array, params.pv_array.fixed_location)
                / Gp
                / Ipp
            )

            # contribution from tube:

            # contribution from C0 to matrix:

            theo_matrix[
                : params.pv_array.modules_per_span,
                params.pv_array.fixed_location : params.pv_array.modules_per_span + 1,
            ] += -connector_locations[params.pv_array.fixed_location] / (Gt * Ipt)
            theo_matrix[
                : params.pv_array.modules_per_span, 1 : params.pv_array.fixed_location
            ] += -connector_locations[1 : params.pv_array.fixed_location] / (Gt * Ipt)

            # contribution from C0 to vector:
            theo_vector[: params.pv_array.modules_per_span] += (
                total_torque_on_this_panel
                * connector_locations[params.pv_array.fixed_location]
                / Gt
                / Ipt
            )

            # contribution of tube to matrix
            theo_matrix[
                0, 1 : params.pv_array.modules_per_span + 1
            ] += connector_locations[0] / (Gt * Ipt)
            theo_matrix[
                1, 1 : params.pv_array.modules_per_span + 1
            ] += connector_locations[1] / (Gt * Ipt)

            for i in range(2, params.pv_array.fixed_location):
                theo_matrix[
                    i, i : params.pv_array.modules_per_span + 1
                ] += connector_locations[i] / (Gt * Ipt)
                theo_matrix[i, 1:i] += connector_locations[1:i] / (Gt * Ipt)

            for i in range(
                params.pv_array.fixed_location, params.pv_array.modules_per_span
            ):
                theo_matrix[
                    i, i + 1 : params.pv_array.modules_per_span + 1
                ] += connector_locations[i + 1] / (Gt * Ipt)
                theo_matrix[i, 1 : i + 1] += connector_locations[1 : i + 1] / (Gt * Ipt)

                # contribution of tube to vector
            theo_vector[: params.pv_array.fixed_location] += (
                -total_torque_on_this_panel
                / Gt
                / Ipt
                * connector_locations[: params.pv_array.fixed_location]
            )
            theo_vector[
                params.pv_array.fixed_location : params.pv_array.modules_per_span
            ] += (
                -total_torque_on_this_panel
                / Gt
                / Ipt
                * connector_locations[params.pv_array.fixed_location]
            )

            R_reaction_torque = np.dot(
                np.linalg.inv(theo_matrix), theo_vector
            )  # this is the torque applied to tube

            # rotation of tube at each connector location
            tube_rotate_matrix = np.zeros(
                (params.pv_array.modules_per_span, params.pv_array.modules_per_span + 1)
            )
            tube_rotate_vector = np.zeros(params.pv_array.modules_per_span)

            # contribution from panel:

            # contribution from C0 to matrix:

            tube_rotate_matrix[
                : params.pv_array.modules_per_span, : params.pv_array.fixed_location
            ] += connector_locations[params.pv_array.fixed_location] / (Gp * Ipp)
            tube_rotate_matrix[
                : params.pv_array.modules_per_span, 1 : params.pv_array.fixed_location
            ] -= connector_locations[1 : params.pv_array.fixed_location] / (Gp * Ipp)

            # contribution from C0 to vector:
            tube_rotate_vector[: params.pv_array.modules_per_span] += (
                T_double_integral_array[params.pv_array.fixed_location] / Gp / Ipp
            )

            # contribution of panels to matrix
            tube_rotate_matrix[0, 0] += -connector_locations[0] / (Gp * Ipp)
            tube_rotate_matrix[1, 0] += -connector_locations[1] / (Gp * Ipp)

            for i in range(2, params.pv_array.fixed_location):
                tube_rotate_matrix[i, :i] += -connector_locations[i] / (Gp * Ipp)
                tube_rotate_matrix[i, 1:i] += connector_locations[1:i] / (Gp * Ipp)

            for i in range(
                params.pv_array.fixed_location, params.pv_array.modules_per_span
            ):
                tube_rotate_matrix[i, : i + 1] += -connector_locations[i + 1] / (
                    Gp * Ipp
                )
                tube_rotate_matrix[i, 1 : i + 1] += connector_locations[1 : i + 1] / (
                    Gp * Ipp
                )

                # contribution of panels to vector
            tube_rotate_vector[: params.pv_array.modules_per_span] += (
                -np.delete(T_double_integral_array, params.pv_array.fixed_location)
                / Gp
                / Ipp
            )

            phi = np.dot(tube_rotate_matrix, R_reaction_torque) + tube_rotate_vector

            if isinstance(params.pv_array.tracker_angle, list):
                if panel_id == 0:
                    assert (
                        len(params.pv_array.tracker_angle)
                        == params.pv_array.stream_rows * params.pv_array.span_rows
                    ), f"Length of tracker angle list ({len(params.pv_array.tracker_angle)}) not equal to total number of PV tables ({params.pv_array.stream_rows * params.pv_array.span_rows})."

                tracker_angle_rad = np.radians(params.pv_array.tracker_angle[panel_id])
            else:
                tracker_angle_rad = np.radians(params.pv_array.tracker_angle)

            if np.linalg.norm(phi) == 0:
                K = np.zeros(params.pv_array.modules_per_span)
            else:
                K = np.abs(
                    12
                    * (np.delete(R_reaction_torque, params.pv_array.fixed_location))
                    / (
                        (
                            params.pv_array.block_chord_div_by_panel_chord
                            * params.pv_array.panel_chord
                        )
                        ** 3
                    )
                    / np.cos(tracker_angle_rad + phi)
                    / (np.sin(tracker_angle_rad + phi) - np.sin(tracker_angle_rad))
                    / (
                        params.pv_array.block_chord_div_by_panel_chord
                        * params.pv_array.panel_chord
                        / 2
                    )
                )

            # assume K is 0 at the fixed connector
            K = np.flip(np.insert(K, params.pv_array.fixed_location, 0))

            # K[0] is the stiffness at the most back connector (highest y)
            # K[10] is the stiffness at the most front connector (lowest y)
            # for block_bot_surface, it is numbered from lowest y to highest y, so flip K

            # if there are 10 modules per array, there are 11 connectors, K has shape of 10, the K at the fixed connector is not included.
            for i in range(params.pv_array.modules_per_span + 1):

                name_K = f"spring_stiffness_{panel_id:.0f}_{i:.0f}"
                setattr(self, name_K, K[i])

        # num_panel_right_fixed = params.pv_array.modules_per_span // 2
        # num_panel_left_fixed = params.pv_array.modules_per_span - num_panel_right_fixed

        # # for right fixed part:

        # for panel_id in range(total_num_panels):
        #     total_torque_right_fixed = 0
        #     total_torque_left_fixed = 0

        #     for i in range(num_panel_left_fixed, params.pv_array.modules_per_span):
        #         name = f"total_torque_panel_{panel_id:.0f}_{i:.0f}"
        #         total_torque_right_fixed += getattr(flow, name)

        #     for i in range(num_panel_left_fixed):
        #         name = f"total_torque_panel_{panel_id:.0f}_{i:.0f}"
        #         total_torque_left_fixed += getattr(flow, name)

        #     theo_matrix_right_fixed = np.zeros((num_panel_right_fixed+1, num_panel_right_fixed+1))
        #     theo_vector_right_fixed = np.zeros(num_panel_right_fixed+1)

        #     theo_matrix_right_fixed[0, :] = 1.0
        #     theo_vector_right_fixed[0] = total_torque_right_fixed
        #     theo_matrix_right_fixed[1:, :-1] = params.pv_array.panel_span/params.pv_array.modules_per_span/Gt/Ipt

        #     # as the double integral of torque along the span from front to back, left fixed part to right fixed part, it need to be flipped
        #     T_double_integral_right_fixed = []
        #     T_right_fixed = []
        #     for i in range(params.pv_array.modules_per_span):
        #         name = f"double_integral_total_torque_panel_{panel_id:.0f}_{params.pv_array.modules_per_span-1-i:.0f}"
        #         T_double_integral_right_fixed.append(getattr(flow, name))
        #         name = f"total_torque_panel_{panel_id:.0f}_{params.pv_array.modules_per_span-1-i:.0f}"
        #         T_right_fixed.append(getattr(flow, name))
        #     for i in range(num_panel_right_fixed):
        #         for j in range(i+1):
        #             theo_vector_right_fixed[i+1] += T_double_integral_right_fixed[-1-j]/(Gp*Ipp)
        #             theo_vector_right_fixed[i+1] -= (i+1-j)*T_right_fixed[-1-j]/(Gp*Ipp)*params.pv_array.panel_span/params.pv_array.modules_per_span
        #         for j in range(i):
        #             theo_matrix_right_fixed[i+1, :-1-j-1] += params.pv_array.panel_span/params.pv_array.modules_per_span/(Gt*Ipt)

        #     for i in range(num_panel_right_fixed):
        #         for j in range(i+1):
        #                 theo_matrix_right_fixed[i+1, num_panel_right_fixed-j:] -= params.pv_array.panel_span/params.pv_array.modules_per_span/Gp/Ipp

        #     R_reaction_torque = np.dot(np.linalg.inv(theo_matrix_right_fixed), theo_vector_right_fixed)

        #     tube_rotate_matrix = np.ones((num_panel_right_fixed, num_panel_right_fixed))*params.pv_array.panel_span/params.pv_array.modules_per_span*(1.0/Gt/Ipt)

        #     for i in range(num_panel_right_fixed):
        #         for j in range(i):
        #             tube_rotate_matrix[i, :num_panel_right_fixed-1-j] += params.pv_array.panel_span/params.pv_array.modules_per_span*(1.0/Gt/Ipt)

        #     # check the standalone code, why it need to be flipped.
        #     phi = np.flip(np.dot(tube_rotate_matrix, R_reaction_torque[:-1]))

        #     # rotation of connector
        #     x_panel = np.arange(num_panel_right_fixed)*(params.pv_array.panel_span/params.pv_array.modules_per_span) # location of panels

        #     block_length = params.pv_array.block_chord_div_by_panel_chord * params.pv_array.panel_chord
        #     block_width = params.pv_array.block_chord_div_by_panel_chord*params.pv_array.panel_span/params.pv_array.modules_per_span/2

        #     # To Do: check the angle, is this correct?
        #     array_rotation = (params.fluid.wind_direction + 90.0) % 360.0
        #     array_rotation_rad = np.radians(array_rotation)

        #     for i in range(num_panel_right_fixed):

        #         phi_i = phi[i]

        #         K_i = 12*(R_reaction_torque[i])/((block_length)**3)/np.cos(array_rotation_rad+phi_i)/(np.sin(array_rotation_rad+phi_i)-np.sin(array_rotation_rad))/block_width

        #         name_K = f"spring_stiffness_{panel_id:.0f}_{params.pv_array.modules_per_span+1-i:.0f}"
        #         setattr(self, name_K, K_i)

        #     # for left fixed part:
        #     theo_matrix_left_fixed = np.zeros((num_panel_left_fixed+1, num_panel_left_fixed+1))
        #     theo_vector_left_fixed = np.zeros(num_panel_left_fixed+1)
        #     theo_matrix_left_fixed[0, :] = 1.0
        #     theo_vector_left_fixed[0] = total_torque_left_fixed

        #     T_double_integral_left_fixed = []
        #     T_left_fixed = []

        #     for i in range(num_panel_left_fixed):
        #         name = f"double_integral_total_torque_panel_{panel_id:.0f}_{num_panel_left_fixed-1-i:.0f}"
        #         T_double_integral_left_fixed.append(getattr(flow, name))
        #         name = f"total_torque_panel_{panel_id:.0f}_{num_panel_left_fixed-1-i:.0f}"
        #         T_left_fixed.append(getattr(flow, name))

        #     for i in range(num_panel_left_fixed):
        #         for j in range(i+1):
        #             theo_matrix_left_fixed[i+1, j] = (i+1-j)*params.pv_array.panel_span/params.pv_array.modules_per_span/(Gp*Ipp)
        #             theo_vector_left_fixed[i+1] += T_double_integral_left_fixed[j]/(Gp*Ipp)
        #         for j in range(i):
        #             theo_vector_left_fixed[i+1] += (i-j)*T_left_fixed[j]/(Gp*Ipp)*params.pv_array.panel_span/params.pv_array.modules_per_span

        #     for i in range(num_panel_left_fixed):
        #         theo_matrix_left_fixed[i+1:, i+1:] -= params.pv_array.panel_span/params.pv_array.modules_per_span/Gt/Ipt

        #     R_reaction_torque = np.dot(np.linalg.inv(theo_matrix_left_fixed), theo_vector_left_fixed)  # this is the torque applied to tube

        #     tube_rotate_matrix = np.zeros((num_panel_left_fixed, num_panel_left_fixed))
        #     for i in range(num_panel_left_fixed):
        #         tube_rotate_matrix[i:, i:] += params.pv_array.panel_span/params.pv_array.modules_per_span/Gt/Ipt

        #     phi = np.dot(tube_rotate_matrix, R_reaction_torque[1:])

        #     # rotation of connector
        #     x_panel = np.arange(num_panel_left_fixed)*(params.pv_array.panel_span/params.pv_array.modules_per_span)+params.pv_array.panel_span/params.pv_array.modules_per_span # location of panles

        #     for i in range(num_panel_left_fixed):

        #         phi_i = phi[i]

        #         K_i = 12*(R_reaction_torque[i+1])/((block_length)**3)/np.cos(array_rotation_rad+phi_i)/(np.sin(array_rotation_rad+phi_i)-np.sin(array_rotation_rad))/block_width

        #         name_K = f"spring_stiffness_{panel_id:.0f}_{num_panel_left_fixed-1-i:.0f}"

        #         setattr(self, name_K, K_i)

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

    def build_forms(self, domain, params, structure, flow):
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

        self.trial_tensor = ufl.TrialFunction(self.T)
        self.test_tensor = ufl.TestFunction(self.T)
        self.internal_stress = dolfinx.fem.Function(self.T, name="stress_structure")

        self.stress = dolfinx.fem.Function(self.T, name="stress_fluid")
        self.stress_old = dolfinx.fem.Function(self.T, name="stress_fluid_old")
        self.stress_predicted = dolfinx.fem.Function(
            self.T, name="stress_fluid_predicted"
        )

        # self.sigma_vm_h = dolfinx.fem.Function(self.W, name="Stress")

        # discplacement
        self.u = dolfinx.fem.Function(self.V, name="deformation")
        self.u_old = dolfinx.fem.Function(self.V, name="deformation_old")
        self.u_delta = dolfinx.fem.Function(self.V, name="deformation_change")

        # velocity
        self.v = dolfinx.fem.Function(self.V)
        self.v_old = dolfinx.fem.Function(self.V, name="velocity")

        # acceleration
        self.a = dolfinx.fem.Function(self.V)
        self.a_old = dolfinx.fem.Function(self.V, name="acceleration")

        # dss = ufl.ds(subdomain_data=boundary_subdomains)

        # def sigma(r):
        #     return dolfinx.fem.form(2.0*self.lame_mu*ufl.sym(ufl.grad(r)) + self.lame_lambda *ufl.tr(ufl.sym(ufl.grad(r)))*ufl.Identity(len(r)))

        # # Mass form
        # def m(u, u_):
        #     return dolfinx.fem.form(self.rho*ufl.inner(u, u_)*ufl.dx)

        # # Elastic stiffness form
        # def k_nominal(u, u_):
        #     return dolfinx.fem.form(ufl.inner(sigma(u), ufl.sym(ufl.grad(u_)))*ufl.dx)

        # # Rayleigh damping form
        # def c(u, u_):
        #     return dolfinx.fem.form(self.eta_m*m(u, u_) + self.eta_k*k_nominal(u, u_))

        # # Work of external forces
        # def Wext(u_):
        #     return ufl.dot(u_, self.f)*self.ds #dss(3)

        # def sigma(r):
        #     return structure.lame_lambda * ufl.nabla_div(r) * ufl.Identity(
        #         len(r)
        #     ) + 2 * structure.lame_mu * ufl.sym(ufl.grad(r))

        def m(u, u_):
            return structure.rho * ufl.inner(u, u_)

        def c(u, u_):
            return self.eta_m * m(u, u_) + self.eta_k * k_nominal(u, u_)

        # def k_cauchy(u, u_):
        #     return ufl.inner(sigma(u), ufl.grad(u_))

        def k_nominal(u, u_):
            return ufl.inner(P_(u), ufl.grad(u_))

        def k_nominal_connector(u, u_):
            return ufl.inner(P_connector(u), ufl.grad(u_))

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

            S_svk = structure.lame_lambda * ufl.tr(E) * I + 2.0 * structure.lame_mu * E
            return S_svk

        # The second Piola–Kirchhoff stress, S
        def S_connector(u):
            E = E_(u)
            I = ufl.Identity(len(u))

            # return lamda * ufl.tr(E) * I + 2.0 * mu * (E - ufl.tr(E) * I / 3.0)
            # TODO: Why does the above form give a better result and where does it come from?

            S_svk = (
                structure.lame_lambda_connector * ufl.tr(E) * I
                + 2.0 * structure.lame_mu_connector * E
            )
            return S_svk

        # The first Piola–Kirchhoff stress tensor, P = F*S
        def P_(u):
            F = F_(u)
            S = S_(u)
            # return ufl.inv(F) * S
            return F * S

        def P_connector(u):
            F = F_(u)
            S = S_connector(u)
            # return ufl.inv(F) * S
            return F * S

        # self.uh_exp = dolfinx.fem.Function(self.V,  name="Deformation")

        # def σ(v):
        #     """Return an expression for the stress σ given a displacement field"""
        #     return 2.0 * structure.lame_mu * ufl.sym(ufl.grad(v)) + structure.lame_lambda * ufl.tr(
        #         ufl.sym(ufl.grad(v))
        #     ) * ufl.Identity(len(v))

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
        if domain.ndim == 2:
            self.f = dolfinx.fem.Constant(
                domain.structure.msh,
                PETSc.ScalarType(
                    (
                        params.structure.body_force_x,
                        params.structure.body_force_y,
                    )
                ),
            )
        elif domain.ndim == 3:
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

        self.z_unit_vector = dolfinx.fem.Constant(
            domain.structure.msh, [0.0, 0.0, 1.0]
        )  # surface traction, N/m^2

        if (
            domain.modeling_torque_tube
            and params.general.geometry_module == "panels3d"
        ):
            self.calculate_K_for_Robin_BC(domain, flow, params)

        dx_structure = ufl.Measure(
            "dx", domain=domain.structure.msh, subdomain_data=domain.structure.cell_tags
        )

        if (
            domain.modeling_torque_tube
            and params.general.geometry_module == "panels3d"
        ):
            self.res = (
                m(self.avg(self.a_old, a_new, self.alpha_m), self.u_) * dx_structure
                + c(self.avg(self.v_old, v_new, self.alpha_f), self.u_) * dx_structure
                + k_nominal(self.avg(self.u_old, self.u, self.alpha_f), self.u_)
                * dx_structure(domain.domain_markers["modules"]["idx"])
                + k_nominal_connector(
                    self.avg(self.u_old, self.u, self.alpha_f), self.u_
                )
                * dx_structure(domain.domain_markers["connectors"]["idx"])
                - structure.rho
                * ufl.inner(self.f, self.u_)
                * dx_structure(domain.domain_markers["modules"]["idx"])
                - structure.rho_connector
                * ufl.inner(self.f, self.u_)
                * dx_structure(domain.domain_markers["connectors"]["idx"])
                - ufl.dot(ufl.dot(self.stress_predicted * J * ufl.inv(F.T), n), self.u_)
                * self.ds
            )  # - Wext(self.u)

            # Robin boundary condition terms
            for panel_id in range(
                params.pv_array.stream_rows * params.pv_array.span_rows
            ):
                for i in range(params.pv_array.modules_per_span + 1):
                    name_K = f"spring_stiffness_{panel_id:.0f}_{i:.0f}"
                    K_springs = dolfinx.fem.Constant(
                        domain.structure.msh, float(getattr(self, name_K))
                    )
                    self.res -= ufl.dot(
                        K_springs * self.u_, self.z_unit_vector
                    ) * self.ds(
                        domain.domain_markers[f"block_bottom_{panel_id:.0f}_{i:.0f}"][
                            "idx"
                        ]
                    )
        else:
            self.res = (
                m(self.avg(self.a_old, a_new, self.alpha_m), self.u_) * dx_structure
                + c(self.avg(self.v_old, v_new, self.alpha_f), self.u_) * dx_structure
                + k_nominal(self.avg(self.u_old, self.u, self.alpha_f), self.u_)
                * dx_structure
                - structure.rho * ufl.inner(self.f, self.u_) * dx_structure
                - ufl.dot(ufl.dot(self.stress_predicted * J * ufl.inv(F.T), n), self.u_)
                * self.ds
            )  # - Wext(self.u)

        # self.a = dolfinx.fem.form(ufl.lhs(res))
        # self.L = dolfinx.fem.form(ufl.rhs(res))

        # Save a form to project the first Piola–Kirchhoff, P_, stress tensor in the structure
        # u * v * dx = P_ * v * dx, where u and v are trial and test functions on tensor function space
        F_k_nominal_proj = ufl.inner(self.trial_tensor, self.test_tensor) * ufl.dx
        F_k_nominal_proj -= (
            ufl.inner(P_(self.avg(self.u_old, self.u, self.alpha_f)), self.test_tensor)
            * ufl.dx
        )

        self.k_nominal_proj = F_k_nominal_proj

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

    def solve(self, params, dataIO, structure):
        # def σ(v):
        #     """Return an expression for the stress σ given a displacement field"""
        #     return 2.0 * self.lame_mu * ufl.sym(ufl.grad(v)) + self.lame_lambda * ufl.tr(
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
            idx = structure.north_east_corner_dofs[0]
            north_east_corner_deformation = self.u.x.array[
                structure.ndim * idx : structure.ndim * idx + structure.ndim
            ].astype(np.float64)
            print(f"Deformation: {north_east_corner_deformation}")

        except:
            north_east_corner_deformation = np.zeros(structure.ndim, dtype=np.float64)

        try:
            idx = structure.north_east_corner_dofs[0]
            north_east_corner_acceleration = self.a_old.x.array[
                structure.ndim * idx : structure.ndim * idx + structure.ndim
            ].astype(np.float64)
            print(f"Acceleration: {north_east_corner_acceleration}")

        except:
            north_east_corner_acceleration = np.zeros(structure.ndim, dtype=np.float64)

        # Initialize a buffer to collect everything into
        north_east_corner_deformation_global = np.zeros(
            (self.num_procs, structure.ndim), dtype=np.float64
        )

        north_east_corner_acceleration_global = np.zeros(
            (self.num_procs, structure.ndim), dtype=np.float64
        )

        # Gather all points (many of which are zeros) to rank 0
        self.comm.Gather(
            north_east_corner_deformation, north_east_corner_deformation_global, root=0
        )

        self.comm.Gather(
            north_east_corner_acceleration,
            north_east_corner_acceleration_global,
            root=0,
        )

        if self.rank == 0:
            norm2 = np.sum(north_east_corner_deformation_global**2, axis=1)
            max_norm2_idx = np.argmax(norm2)
            np_deformation = north_east_corner_deformation_global[max_norm2_idx, :]

            norm2 = np.sum(north_east_corner_acceleration_global**2, axis=1)
            max_norm2_idx = np.argmax(norm2)
            np_acceleration = north_east_corner_acceleration_global[max_norm2_idx, :]

            accel_pos_filename = os.path.join(
                params.general.output_dir_sol, "accel_pos.csv"
            )

            if self.first_call_to_solver:

                with open(accel_pos_filename, "w") as fp:
                    if structure.ndim == 3:
                        fp.write(
                            "#x-deformation,y-deformation,z-deformation,x-acceleration,y-acceleration,z-acceleration\n"
                        )
                        fp.write(
                            f"{np_deformation[0]},{np_deformation[1]},{np_deformation[2]},"
                        )
                        fp.write(
                            f"{np_acceleration[0]},{np_acceleration[1]},{np_acceleration[2]}\n"
                        )
                    elif structure.ndim == 2:
                        fp.write(
                            "#x-deformation,y-deformation,x-acceleration,y-acceleration\n"
                        )
                        fp.write(f"{np_deformation[0]},{np_deformation[1]},")
                        fp.write(f"{np_acceleration[0]},{np_acceleration[1]}\n")

            else:
                with open(accel_pos_filename, "a") as fp:
                    if structure.ndim == 3:
                        fp.write(
                            f"{np_deformation[0]},{np_deformation[1]},{np_deformation[2]},"
                        )
                        fp.write(
                            f"{np_acceleration[0]},{np_acceleration[1]},{np_acceleration[2]}\n"
                        )
                    elif structure.ndim == 2:
                        fp.write(f"{np_deformation[0]},{np_deformation[1]},")
                        fp.write(f"{np_acceleration[0]},{np_acceleration[1]}\n")

        if self.first_call_to_solver:
            self.first_call_to_solver = False

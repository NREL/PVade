import dolfinx
from petsc4py import PETSc

import numpy as np
import h5py
import scipy.interpolate as interp

import warnings


def get_facet_dofs_by_gmsh_tag(domain, functionspace, location):
    """Associate degrees of freedom with marker functions

    This function uses the marker information in the gmsh specification to
    find the corresponding degrees of freedom for use in the actual
    boundary condition specification. Note that this method does not
    require access to the facet information computed with
    ``_locate_boundary_entities``.

    Args:
        domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object
        functionspace (dolfinx.fem.FunctionSpace): The dolfinx function space on which the boundary condition will be applied
        location (str, int): A string identifier of the boundary condition location, e.g., "x_min" or the equivalent integer tag, e.g., 1

    Returns:
        list: A list of the degrees of freedom associated with this tag and functionspace

    """

    facet_dim = domain.ndim - 1

    if isinstance(location, str):
        found_entities = domain.fluid.facet_tags.find(
            domain.domain_markers[location]["idx"]
        )
    elif isinstance(location, int):
        found_entities = domain.fluid.facet_tags.find(location)

    debugging = False

    if debugging:
        global_found_entities = domain.comm.gather(found_entities, root=0)

        if domain.rank == 0:
            flattened = np.hstack(global_found_entities)
            nnn = np.size(flattened)
            print(f"{location}, global_entities = ", nnn)

    # if len(found_entities) == 0:
    #     warnings.warn(f"Found no facets using location = {location}.")

    dofs = dolfinx.fem.locate_dofs_topological(functionspace, facet_dim, found_entities)

    return dofs


def build_vel_bc_by_type(bc_type, domain, functionspace, bc_location):
    """Build a dirichlet bc

    Args:
        bc_type (str): The type of simple boundary condition to apply, "slip" or "noslip"
        domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object
        functionspace (dolfinx.fem.FunctionSpace): The dolfinx function space on which the boundary condition will be applied
        bc_location (str, int): A string identifier of the boundary condition location, e.g., "x_min" or the equivalent integer tag, e.g., 1

    Returns:
        dolfinx.fem.dirichletbc: A dolfinx dirichlet boundary condition
    """

    if domain.rank == 0:
        print(f"Setting '{bc_type}' BC on {bc_location}")

    if bc_type == "noslip":
        if domain.ndim == 2:
            zero_vec = dolfinx.fem.Constant(
                domain.fluid.msh, PETSc.ScalarType((0.0, 0.0))
            )
        elif domain.ndim == 3:
            zero_vec = dolfinx.fem.Constant(
                domain.fluid.msh, PETSc.ScalarType((0.0, 0.0, 0.0))
            )

        dofs = get_facet_dofs_by_gmsh_tag(domain, functionspace, bc_location)

        bc = dolfinx.fem.dirichletbc(zero_vec, dofs, functionspace)

    elif bc_type == "slip":
        zero_scalar = dolfinx.fem.Constant(domain.fluid.msh, PETSc.ScalarType(0.0))

        if bc_location in ["x_min", "x_max"]:
            sub_id = 0
        elif bc_location in ["y_min", "y_max"]:
            sub_id = 1
        elif bc_location in ["z_min", "z_max"]:
            sub_id = 2

        dofs = get_facet_dofs_by_gmsh_tag(
            domain, functionspace.sub(sub_id), bc_location
        )

        bc = dolfinx.fem.dirichletbc(zero_scalar, dofs, functionspace.sub(sub_id))

    else:
        if domain.rank == 0:
            raise ValueError(f"{bc_type} BC not recognized")

    return bc


class InflowVelocity:
    def __init__(self, ndim, params, current_time):
        """Inflow velocity object

        Args:
            geom_dim (int): The geometric dimension (as opposed to the topological dimension, usually this is 3)
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object
        """
        self.ndim = ndim
        self.params = params
        self.current_time = current_time

        if self.params.fluid.velocity_profile_type == "specified_from_file":
            with h5py.File(self.params.fluid.h5_filename, "r") as fp:
                self.time_index = fp["time_index"][:]
                self.y_coordinates = fp["y_coordinates"][:]
                self.z_coordinates = fp["z_coordinates"][:]
                self.u = fp["u"][:]
                self.v = fp["v"][:]
                self.w = fp["w"][:]

            # Create the interpolators for the time-averaged values
            x0_bar = (self.z_coordinates, self.y_coordinates)

            self.interp_u_bar = interp.RegularGridInterpolator(x0_bar, np.mean(self.u, axis=0), bounds_error=False, fill_value=None)
            self.interp_v_bar = interp.RegularGridInterpolator(x0_bar, np.mean(self.v, axis=0), bounds_error=False, fill_value=None)
            self.interp_w_bar = interp.RegularGridInterpolator(x0_bar, np.mean(self.w, axis=0), bounds_error=False, fill_value=None)

            # Create the known axes for our interpolators (t0, z0, y0) for instantaneous values
            x0 = (self.time_index, self.z_coordinates, self.y_coordinates)
            
            self.interp_u = interp.RegularGridInterpolator(x0, self.u, bounds_error=False, fill_value=None)
            self.interp_v = interp.RegularGridInterpolator(x0, self.v, bounds_error=False, fill_value=None)
            self.interp_w = interp.RegularGridInterpolator(x0, self.w, bounds_error=False, fill_value=None)
            
            self.inflow_t_final = self.time_index[-1] # [s] time of last timestep of inflow file

            # compute effective u_ref
            self.compute_eff_u_ref(params)
            params.fluid.u_ref = self.u_ref # ?? is this bad practice?


    def __call__(self, x):
        """Define an inflow expression for use as boundary condition

        Args:
            x (np.ndarray): Array of coordinates

        Returns:
            np.ndarray: Value of velocity at each coordinate in input array
        """
        # print('shape of x = ', np.shape(x))
        
        # Preallocated velocity vector that we will fill
        inflow_values = np.zeros((self.ndim, x.shape[1]), dtype=PETSc.ScalarType)
        # print('shape of inflow_values = ', np.shape(inflow_values))

        # if self.first_call_to_inflow_velocity:
        #     print(f"creating {self.params.fluid.velocity_profile_type} inflow profile")

        # Assign time_vary_u_ref, for cases with time_varying_inflow_bc = 0.0:
        #     time_vary_u_ref = u_ref
        # for cases with time_varying_inflow_bc > 0.0:
        #     time_vary_u_ref goes from 0 -> u_ref smoothly over ramp_up_window time
        #     e.g., start velocity at 0, and by t=1.0 seconds, achieve full inflow speed

        ramp_window = self.params.fluid.ramp_window

        if ramp_window > 0.0 and self.current_time <= ramp_window:
            time_vary_u_ref = (
                self.params.fluid.u_ref
                * (1.0 - np.cos(np.pi / ramp_window * self.current_time))
                / 2.0
            )
        else:
            time_vary_u_ref = self.params.fluid.u_ref

        if self.params.fluid.velocity_profile_type == "uniform":
            inflow_values[0] = time_vary_u_ref

        elif self.params.fluid.velocity_profile_type == "parabolic":
            coeff = self.params.fluid.inflow_coeff

            # handle cyl2d and flag2d
            if self.ndim == 2:
                inflow_values[0] = (
                    coeff
                    * time_vary_u_ref
                    / self.params.domain.y_max**2  # 0.1681
                    * x[1]
                    * (self.params.domain.y_max - x[1])
                )

            # handle cyl3d
            elif self.ndim == 3:
                inflow_values[0] = (
                    coeff * time_vary_u_ref * x[1] * x[2] * self.params.domain.y_max
                    - x[1] * self.params.domain.z_max  # inflow_dy
                    - x[2] / self.params.domain.z_max**4  # inflow_dz
                )

        elif self.params.fluid.velocity_profile_type == "loglaw":
            z0 = self.params.fluid.z0
            d0 = self.params.fluid.d0
            z_hub = self.params.pv_array.elevation

            # handle panels3d
            if self.ndim == 3:
                inflow_values[0] = (
                    (time_vary_u_ref)
                    * np.log(((x[2]) - d0) / z0)
                    / (np.log((z_hub - d0) / z0))
                )

            # handle panels2d
            elif self.ndim == 2:
                # print("this is 2d")
                inflow_values[0] = (
                    (time_vary_u_ref)  # shouldn't this be u_star?
                    * np.log(((x[1]) - d0) / z0)
                    / (np.log((z_hub - d0) / z0))
                )

        elif self.params.fluid.velocity_profile_type == "specified_from_file":

            # only needed at time zero
            if self.current_time == 0.0:
                # These are the points over the entirety of the domain
                xi_bulk = np.vstack((x[2], x[1])).T

                u_vel_bar = self.interp_u_bar(xi_bulk) # this grabs u,v,w at current position
                v_vel_bar = self.interp_v_bar(xi_bulk)
                w_vel_bar = self.interp_w_bar(xi_bulk)

                # Assign the mean inflow values (maybe just make a separate interpolator with no time and all points like above?)
                # Make this assignment everywhere *knowing* the masked point values will be overwritten
                inflow_values[0, :] = u_vel_bar
                inflow_values[1, :] = v_vel_bar
                inflow_values[2, :] = w_vel_bar

            # assuming always a timeseries for now
            xi_0_mask = x[0] < self.params.domain.x_min + 1e-5      
            ti = self.current_time * np.ones(np.sum(xi_0_mask))

            # These are the points that define the inflow plane only
            xi = np.vstack((ti, x[2][xi_0_mask], x[1][xi_0_mask])).T
            
            u_vel = self.interp_u(xi) # this grabs u,v,w at current time and at x inlet
            v_vel = self.interp_v(xi)
            w_vel = self.interp_w(xi)
            
            # Assign the turbulent values (from the interpolator+file) to the masked inflow plane
            # this will overwrite the previously-assigned averaged values
            inflow_values[0, xi_0_mask] = u_vel
            inflow_values[1, xi_0_mask] = v_vel
            inflow_values[2, xi_0_mask] = w_vel

        # if self.first_call_to_inflow_velocity:
        #     print("inflow_values = ", inflow_values[0])
        #     self.first_call_to_inflow_velocity = False

        return inflow_values
    
    def compute_eff_u_ref(self, params):
        """ Compute time-averaged inflow profile from file

        Outputs:
            u_ref : effective u_ref computed from input inflow file - velocity at panel elevation
            u_inf : streamwise velocity in freestream (at highest elevation in domain)
            z0
        """

        # TODO - add handling for 2D case

        # compute the magnitude to account for other wind directions
        umag = (self.u**2 + self.v**2)**0.5
        umag_mean = np.average(np.average(umag, axis=0), axis=-1) # vertical profile (averaged in time and y)

        # extract u_ref at elevation height
        z_idx = np.argmin(abs(self.z_coordinates - params.pv_array.elevation))
        u_ref_calc = umag_mean[z_idx]

        self.u_ref = u_ref_calc

def get_inflow_profile_function(domain, params, functionspace, current_time):
    ndim = domain.ndim
    # print('ndim = ',ndim)

    # IMPORTANT: this is distinct from ndim because a mesh can
    # be 2D but have 3 componenets of position, e.g., a mesh
    # with triangular elements can still have a z dimension = 0.
    # This only gets used in making sure the inflow value
    # numpy array has the correct size, e.g.:
    #
    # np.zeros((geom_dim, x.shape[1]). dtype=...)
    # geom_dim = domain.fluid.msh.geometry.dim

    inflow_function = dolfinx.fem.Function(functionspace)
    # if params.general.debug_flag == True:
    #     print('functionspace = ',functionspace)

    inflow_velocity = InflowVelocity(ndim, params, current_time)

    upper_cells = None

    if params.fluid.velocity_profile_type == "parabolic":
        if domain.rank == 0:
            print("setting parabolic profile")
        inflow_function.interpolate(inflow_velocity)

    elif params.fluid.velocity_profile_type == "uniform":
        if domain.rank == 0:
            print("setting uniform profile")
        inflow_function.interpolate(inflow_velocity)

    elif params.fluid.velocity_profile_type == "loglaw":
        if domain.rank == 0:
            print("setting loglaw profile")
        z0 = params.fluid.z0
        d0 = params.fluid.d0
        if ndim == 3:
            upper_cells = dolfinx.mesh.locate_entities(
                domain.fluid.msh, ndim, lambda x: x[2] > d0 + z0
            )
            lower_cells = dolfinx.mesh.locate_entities(
                domain.fluid.msh, ndim, lambda x: x[2] <= d0 + z0
            )
        elif ndim == 2:
            upper_cells = dolfinx.mesh.locate_entities(
                domain.fluid.msh, ndim, lambda x: x[1] > d0 + z0
            )
            lower_cells = dolfinx.mesh.locate_entities(
                domain.fluid.msh, ndim, lambda x: x[1] <= d0 + z0
            )

        # what does this do? zero everything out before defining values of upper cells?
        inflow_function.interpolate(
            lambda x: np.zeros((ndim, x.shape[1]), dtype=PETSc.ScalarType)
        )

        if len(upper_cells) == 0:
            print(
                "Warning: z0 and d0 may be outside the size of the domain"
            )  # just a bandaid for now

        inflow_function.interpolate(inflow_velocity, upper_cells)

    elif params.fluid.velocity_profile_type == "specified_from_file":
        if domain.rank == 0:
            print("Setting inflow velocity from {}".format(params.fluid.h5_filename))
            if params.general.debug_flag:
                print('eff u_ref = {} m/s'.format(inflow_velocity.u_ref))
        inflow_function.interpolate(inflow_velocity)

        if params.solver.t_final > inflow_velocity.inflow_t_final:
            if domain.rank == 0:
                print("WARNING: t_final ({:.2f} s) exceeds the final time in input inflow velocity file ({:.2f} s). " \
                "Simulation will fail at that point.".format(
                    params.solver.t_final, inflow_velocity.inflow_t_final
                ))
    
    return inflow_function, inflow_velocity, upper_cells


def build_velocity_boundary_conditions(domain, params, functionspace, current_time):
    """Build all boundary conditions on velocity

    This method builds all the boundary conditions associated with velocity and stores in a list, ``bcu``.

    Args:
        domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object
        params (:obj:`pvade.Parameters.SimParams`): A SimParams object

    """
    # Define velocity boundary conditions

    ndim = domain.ndim

    bcu = []

    mesh_vel = dolfinx.fem.Function(functionspace)
    mesh_vel.interpolate(domain.fluid_mesh_displacement)
    mesh_vel.vector.array[:] /= params.solver.dt

    # Generate list of locations to loop over
    if ndim == 2:
        bc_location_list = ["y_min", "y_max"]

    elif ndim == 3:
        bc_location_list = ["y_min", "y_max", "z_min", "z_max"]

    # Iterate over all user-prescribed boundaries
    for bc_location in bc_location_list:
        bc_type = getattr(params.fluid, f"bc_{bc_location}")

        bc = build_vel_bc_by_type(bc_type, domain, functionspace, bc_location)

        bcu.append(bc)

    # Set the inflow boundary condition
    inflow_function, inflow_velocity, upper_cells = get_inflow_profile_function(
        domain, params, functionspace, current_time
    )

    if domain.rank == 0:
        print("inflow_function = ", inflow_function.x.array[:])

    dofs = get_facet_dofs_by_gmsh_tag(domain, functionspace, "x_min")
    bcu.append(dolfinx.fem.dirichletbc(inflow_function, dofs))

    # Set all interior surfaces to no slip *which sometimes means non-zero values*
    use_surface_vel_from_fsi = True

    for panel_id in range(params.pv_array.stream_rows * params.pv_array.span_rows):
        if (
            params.general.geometry_module == "panels2d"
            or params.general.geometry_module == "panels3d"
            or params.general.geometry_module == "heliostats3d"
            or params.general.geometry_module == "flag2d"
        ):
            for location in (
                f"bottom_{panel_id}",
                f"top_{panel_id}",
                f"left_{panel_id}",
                f"right_{panel_id}",
                f"front_{panel_id}",
                f"back_{panel_id}",
            ):
                if use_surface_vel_from_fsi:
                    # dofs = get_facet_dofs_by_gmsh_tag(domain, functionspace, location)
                    # bc = dolfinx.fem.dirichletbc(mesh_vel, dofs)
                    pass
                    # We need to do this each time at the beginning of the fluid solve!

                else:
                    # bc = build_vel_bc_by_type("noslip", domain, functionspace, "internal_surface")
                    bc = build_vel_bc_by_type("noslip", domain, functionspace, location)
                    bcu.append(bc)

        elif (
            params.general.geometry_module == "cylinder3d"
            or params.general.geometry_module == "cylinder2d"
        ):
            location = f"cylinder_side"
            bc = build_vel_bc_by_type("noslip", domain, functionspace, location)
            bcu.append(bc)

    return bcu, inflow_function, inflow_velocity, upper_cells


def build_pressure_boundary_conditions(domain, params, functionspace):
    """Build all boundary conditions on pressure

    This method builds all the boundary conditions associated with pressure and stores in a list, ``bcp``.

    Args:
        domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object
        params (:obj:`pvade.Parameters.SimParams`): A SimParams object
    """
    # Define pressure boundary conditions
    bcp = []

    zero_scalar = dolfinx.fem.Constant(domain.fluid.msh, PETSc.ScalarType(0.0))

    dofs = get_facet_dofs_by_gmsh_tag(domain, functionspace, "x_max")
    bcp.append(dolfinx.fem.dirichletbc(zero_scalar, dofs, functionspace))

    return bcp


def build_temperature_boundary_conditions(domain, params, functionspace):
    """Build all boundary conditions on temperature

    This method builds all the boundary conditions associated with temperature and stores in a list, ``bcT``.

    Args:
        domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object
        params (:obj:`pvade.Parameters.SimParams`): A SimParams object
    """

    class LowerWallTemperature:
        # copied from https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
        def __init__(self):
            pass

        def __call__(self, x):
            values = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)

            # linear rampdown of temperature along lower wall
            # x0 = 0.75 * x_max # start of ramp down
            # values[0] = (T0_bottom_wall + ((x[0]-x0) / x_max) * (T_ambient - T0_bottom_wall))

            # stepchange -> constant temperature along bottom wall # this method is most stable with pressure exit BC
            values[0] = PETSc.ScalarType(params.fluid.T_bottom)
            return values

    ndim = domain.ndim

    # Define temperature boundary conditions
    bcT = []

    # left wall BCs
    T_ambient_scalar = dolfinx.fem.Constant(
        domain.fluid.msh, PETSc.ScalarType(params.fluid.T_ambient)
    )
    left_wall_dofs = get_facet_dofs_by_gmsh_tag(domain, functionspace, "x_min")
    bcT.append(dolfinx.fem.dirichletbc(T_ambient_scalar, left_wall_dofs, functionspace))

    # if params.general.debug_flag == True:
    #     print('applied left wall temperature bc')

    # lower wall BCs ============================================================================
    # t_bc_flag = 'uniform' # potentially move to input file
    t_bc_flag = "stepchange"  # potentially move to input file

    if ndim == 2:
        bottom_wall_dofs = get_facet_dofs_by_gmsh_tag(domain, functionspace, "y_min")
    elif ndim == 3:
        bottom_wall_dofs = get_facet_dofs_by_gmsh_tag(domain, functionspace, "z_min")

    if t_bc_flag == "stepchange":
        heated_cells = None

        # x_span = params.domain.x_max - params.domain.x_min

        # only applying Tbottom to cells from x=0 to x=0.75*x_max to avoid jet at the outlet sfc due to pressure BCa at exit
        heated_cells = dolfinx.mesh.locate_entities(
            domain.fluid.msh, ndim, lambda x: x[0] < (0.375 * params.domain.x_max)
        )
        T_bottom_function = dolfinx.fem.Function(functionspace)

        # initialize all cells with ambient temperature
        T_bottom_function.interpolate(
            lambda x: np.full(
                (1, x.shape[1]), params.fluid.T_ambient, dtype=PETSc.ScalarType
            )
        )

        bottom_wall_temperature = LowerWallTemperature()
        T_bottom_function.interpolate(
            bottom_wall_temperature, heated_cells
        )  # apply T_bottom to only heated cells
        bcT.append(dolfinx.fem.dirichletbc(T_bottom_function, bottom_wall_dofs))

    elif t_bc_flag == "uniform":
        T_bottom_scalar = dolfinx.fem.Constant(
            domain.fluid.msh, PETSc.ScalarType(params.fluid.T_bottom)
        )
        bcT.append(
            dolfinx.fem.dirichletbc(T_bottom_scalar, bottom_wall_dofs, functionspace)
        )

    # panel surface BCs
    for panel_id in range(params.pv_array.stream_rows * params.pv_array.span_rows):
        if (
            params.general.geometry_module == "panels2d"
            or params.general.geometry_module == "panels3d"
            or params.general.geometry_module == "heliostats3d"
            or params.general.geometry_module == "flag2d"
        ):

            for location in (
                f"bottom_{panel_id}",
                f"top_{panel_id}",
                f"left_{panel_id}",
                f"right_{panel_id}",
                # f"front_{panel_id}", # not valid in panels2d?
                # f"back_{panel_id}",
            ):
                T0_pv_panel_scalar = dolfinx.fem.Constant(
                    domain.fluid.msh, PETSc.ScalarType(params.fluid.T0_panel)
                )

                panel_sfc_dofs = get_facet_dofs_by_gmsh_tag(
                    domain, functionspace, location
                )
                bc = dolfinx.fem.dirichletbc(
                    T0_pv_panel_scalar, panel_sfc_dofs, functionspace
                )

                bcT.append(bc)

    # if params.general.debug_flag == True:
    #     print('built temperature boundary conditions')

    return bcT
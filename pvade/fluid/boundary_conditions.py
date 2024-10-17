import dolfinx
from petsc4py import PETSc

import numpy as np

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
    def __init__(self, geom_dim, params, current_time):
        """Inflow velocity object

        Args:
            geom_dim (int): The geometric dimension (as opposed to the topological dimension, usually this is 3)
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object
        """
        self.geom_dim = geom_dim
        self.params = params
        self.current_time = current_time

    def __call__(self, x):
        """Define an inflow expression for use as boundary condition

        Args:
            x (np.ndarray): Array of coordinates

        Returns:
            np.ndarray: Value of velocity at each coordinate in input array
        """

        z0 = 0.05
        d0 = 0.5

        inflow_values = np.zeros((3, x.shape[1]), dtype=PETSc.ScalarType)

        H = 0.41
        inflow_dy = H - x[1]
        inflow_dz = H - x[2]

        u_hub = self.params.fluid.u_ref
        z_hub = self.params.pv_array.elevation

        if self.params.general.geometry_module == "cylinder3d":
            inflow_values[0] = (
                16.0
                * self.params.fluid.u_ref
                * x[1]
                * x[2]
                * inflow_dy
                * inflow_dz
                / H**4
            )
        elif self.params.general.geometry_module == "cylinder2d":
            inflow_values[0] = (
                4
                * (self.params.fluid.u_ref)
                * np.sin(np.pi / 8)
                * x[1]
                * (0.41 - x[1])
                / (0.41**2)
            )
        elif self.params.general.geometry_module == "flag2d":
            if self.current_time < 2.0:
                time_vary_u_ref = (
                    self.params.fluid.u_ref
                    * (1.0 - np.cos(np.pi / 2.0 * self.current_time))
                    / 2.0
                )
            else:
                time_vary_u_ref = self.params.fluid.u_ref

            inflow_values[0] = (
                1.5
                * time_vary_u_ref
                * 4.0
                / 0.1681
                * x[1]
                * (self.params.domain.y_max - x[1])
            )
        elif (
            self.params.general.geometry_module == "panels3d"
            or self.params.general.geometry_module == "heliostats3d"
        ):
            if self.current_time < 2.0 and self.params.fluid.time_varying_inflow_bc:
                time_vary_u_ref = (
                    self.params.fluid.u_ref
                    * (1.0 - np.cos(np.pi / 2.0 * self.current_time))
                    / 2.0
                )
            else:
                time_vary_u_ref = self.params.fluid.u_ref

            # inflow_values[0] = x[2]
            inflow_values[0] = (
                (time_vary_u_ref)
                * np.log(((x[2]) - d0) / z0)
                / (np.log((z_hub - d0) / z0))
            )
        elif self.params.general.geometry_module == "panels2d":
            inflow_values[0] = (
                (self.params.fluid.u_ref)
                * np.log(((x[1]) - d0) / z0)
                / (np.log((z_hub - d0) / z0))
            )

        return inflow_values


def get_inflow_profile_function(domain, params, functionspace, current_time):
    ndim = domain.ndim

    # IMPORTANT: this is distinct from ndim because a mesh can
    # be 2D but have 3 componenets of position, e.g., a mesh
    # with triangular elements can still have a z dimension = 0.
    # This only gets used in making sure the inflow value
    # numpy array has the correct size, e.g.:
    #
    # np.zeros((geom_dim, x.shape[1]). dtype=...)
    geom_dim = domain.fluid.msh.geometry.dim

    inflow_function = dolfinx.fem.Function(functionspace)

    inflow_velocity = InflowVelocity(geom_dim, params, current_time)

    upper_cells = None

    if params.general.geometry_module in ["cylinder3d", "cylinder2d", "flag2d"]:
        inflow_function.interpolate(inflow_velocity)

    else:
        z0 = 0.05
        d0 = 0.5
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

        inflow_function.interpolate(
            lambda x: np.zeros((geom_dim, x.shape[1]), dtype=PETSc.ScalarType)
        )

        inflow_function.interpolate(inflow_velocity, upper_cells)

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

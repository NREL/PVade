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
        found_entities = domain.structure.facet_tags.find(
            domain.domain_markers[location]["idx"]
        )
    elif isinstance(location, int):
        found_entities = domain.structure.facet_tags.find(location)

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
    def __init__(self, geom_dim, params):
        """Inflow velocity object

        Args:
            geom_dim (int): The geometric dimension (as opposed to the topological dimension, usually this is 3)
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object
        """
        self.geom_dim = geom_dim
        self.params = params

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

        if self.params.general.example == "cylinder3d":
            inflow_values[0] = (
                16.0
                * self.params.fluid.u_ref
                * x[1]
                * x[2]
                * inflow_dy
                * inflow_dz
                / H**4
            )
        elif self.params.general.example == "cylinder2d":
            inflow_values[0] = (
                4
                * (self.params.fluid.u_ref)
                * np.sin(np.pi / 8)
                * x[1]
                * (0.41 - x[1])
                / (0.41**2)
            )
        elif (
            self.params.general.example == "panels3d"
            or self.params.general.example == "heliostats3d"
        ):
            inflow_values[0] = (
                (self.params.fluid.u_ref)
                * np.log(((x[2]) - d0) / z0)
                / (np.log((z_hub - d0) / z0))
            )
        elif self.params.general.example == "panels2d":
            inflow_values[0] = (
                (self.params.fluid.u_ref)
                * np.log(((x[1]) - d0) / z0)
                / (np.log((z_hub - d0) / z0))
            )

        return inflow_values


def get_inflow_profile_function(domain, params, functionspace):
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

    inflow_velocity = InflowVelocity(geom_dim, params)

    if params.general.example in ["cylinder3d", "cylinder2d"]:
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

    return inflow_function


def build_velocity_boundary_conditions(domain, params, functionspace):
    """Build all boundary conditions on velocity

    This method builds all the boundary conditions associated with velocity and stores in a list, ``bcu``.

    Args:
        domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object
        params (:obj:`pvade.Parameters.SimParams`): A SimParams object

    """
    # Define velocity boundary conditions

    ndim = domain.ndim

    bcu = []

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

    # Set all interior surfaces to no slip
    bc = build_vel_bc_by_type("noslip", domain, functionspace, "internal_surface")
    bcu.append(bc)

    # Set the inflow boundary condition
    inflow_function = get_inflow_profile_function(domain, params, functionspace)
    dofs = get_facet_dofs_by_gmsh_tag(domain, functionspace, "x_min")
    bcu.append(dolfinx.fem.dirichletbc(inflow_function, dofs))

    return bcu, inflow_function


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


def build_structure_boundary_conditions(domain, params, functionspace):
    facet_dim = domain.ndim - 1
    zero_vec = dolfinx.fem.Constant(
        domain.structure.msh, PETSc.ScalarType((0.0, 0.0, 0.0))
    )
    bc = []

    total_num_panels = params.pv_array.stream_rows * params.pv_array.span_rows

    for num_panel in range(total_num_panels):
        for location in params.structure.bc_list:
            location_panel = f"{location}_{num_panel}"
            # f"front_{num_panel}" , f"back_{num_panel}":
            # for location in  [f"left_{num_panel}"]:# , f"right_{num_panel}":
            # for location in  f"left_{num_panel}":
            # for location in f"left_{num_panel}":
            # location = f"top_{num_panel}"
            dofs = get_facet_dofs_by_gmsh_tag(domain, functionspace, location_panel)
            bc.append(dolfinx.fem.dirichletbc(zero_vec, dofs, functionspace))

    def connection_point_up(x, nodes_to_pin_between):
        """
        This pins points along a line between the values in `nodes_to_pin_between`

        This function returns a boolean array which is true for nodes
        (0D elements) that fall along a line between the start and stop points expressed
        in `nodes_to_pin_between`. Each row of `nodes_to_pin_between` is a single line
        constraint, with the first three columns expressing the 3D starting point and the
        second three columns expressing the ending point. `nodes_to_pin_between` has
        dimension `number_of_fixation_lines x 6`. The decision about whether or not a node
        is colinear with the line defined by the starting and stopping points is made by
        calculating a cross product. If A is the starting point, B is the stopping point,
        and C is an arbitrary point we are testing for collinearity, then

            | AC x AB | = 0

        or the magnitude of segement AC (test vector) crossed with the segment AB
        (truth vector) being near zero means the three points are collinear.

        """

        eps = 1e-3

        for k, pts in enumerate(nodes_to_pin_between):
            # print(pts)

            test_vecs = x[:, :].T - pts[0:3]
            truth_vec = pts[3:6] - pts[0:3]

            cross_product = np.cross(test_vecs, truth_vec)
            cross_product_mag = np.linalg.norm(cross_product, axis=1)

            pinned_pts = cross_product_mag < eps
            # print(np.shape(cross_product), np.shape(cross_product_mag))

            if k == 0:
                total_pinned_pts = np.copy(pinned_pts)
            else:
                total_pinned_pts = np.logical_or(pinned_pts, total_pinned_pts)

        return total_pinned_pts

    def connection_point_up_helper(nodes_to_pin_between):
        """
        This returns a function with a single input, x, as expected by `dolfinx.mesh.locate_entities`

        This is a wrapper around the function `connection_point_up`. It allows us to pass extra arguments
        to that function (besides just `x`) to define how the calculation should be carried out. The return
        value of this helper function is a function handle with a single input, `x`, which is what
        `dolfinx.mesh.locate_entities` expects.

        That is:

            fn_handle(x) = fn(x, extra_arg_1, extra_arg_2)

        where the extra arguments can be passed in here.

        """
        fn_handle = lambda x: connection_point_up(x, nodes_to_pin_between)

        return fn_handle

    # Start pinning along the lines expressed byt numpy_pt_total_array
    # First, determine the total number of rows (number of lines to pin)
    num_nodes = np.shape(domain.numpy_pt_total_array)[0]

    # Determine how many pinning lines exist per each panel (e.g., 24 lines distributed on 8 panels means 3 lines per panel)
    nodes_per_panel = int(num_nodes / total_num_panels)

    # The torque tube entry (oriented spanwise along the middle, divides panel into upstream and downstream rectangular halves)
    # is always the first entry, e.g., [0, ..., ..., 3, ..., ..., 6, ..., ...], [0, 3, 6] are the torque tubes
    tube_nodes_idx = np.arange(0, num_nodes, nodes_per_panel, dtype=np.int64)

    if params.structure.tube_connection == True:
        # If making torque tube connections, pass only those pinning lines to the BC identification function
        tube_nodes = domain.numpy_pt_total_array[tube_nodes_idx, :]

        facet_uppoint = dolfinx.mesh.locate_entities(
            domain.structure.msh, 1, connection_point_up_helper(tube_nodes)
        )
        dofs_disp = dolfinx.fem.locate_dofs_topological(
            functionspace, 1, [facet_uppoint]
        )

        bc.append(dolfinx.fem.dirichletbc(zero_vec, dofs_disp, functionspace))

    if params.structure.motor_connection == True:
        # If making motor mount connections, pass only those pinning lines to the BC identification function
        # this is done by making a copy of the numpy_pt_total_array with the torque tube lines *deleted*
        # not done in place, so numpy_pt_total_array remains unaltered.
        motor_nodes = np.delete(domain.numpy_pt_total_array, tube_nodes_idx, axis=0)

        facet_uppoint = dolfinx.mesh.locate_entities(
            domain.structure.msh, 1, connection_point_up_helper(motor_nodes)
        )
        dofs_disp = dolfinx.fem.locate_dofs_topological(
            functionspace, 1, [facet_uppoint]
        )

        bc.append(dolfinx.fem.dirichletbc(zero_vec, dofs_disp, functionspace))

    return bc

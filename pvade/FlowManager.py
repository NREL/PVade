"""Summary
"""
import dolfinx
import ufl

from petsc4py import PETSc
from mpi4py import MPI

import numpy as np
import scipy.interpolate as interp


class Flow:
    """This class solves the CFD problem"""

    def __init__(self, domain):
        """Initialize the fluid solver

        This method initialize the Flow object, namely, it creates all the
        necessary function spaces on the mesh, initializes key counting and
        boolean variables and records certain characteristic quantities like
        the minimum cell size and the number of degrees of freedom attributed
        to both the pressure and velocity function spaces.

        Args:
            domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object

        """
        # Store the comm and mpi info for convenience
        self.comm = domain.comm
        self.rank = domain.rank
        self.num_procs = domain.num_procs

        # Pressure (Scalar)
        P1 = ufl.FiniteElement("Lagrange", domain.msh.ufl_cell(), 1)
        self.Q = dolfinx.fem.FunctionSpace(domain.msh, P1)

        # Velocity (Vector)
        P2 = ufl.VectorElement("Lagrange", domain.msh.ufl_cell(), 2)
        self.V = dolfinx.fem.FunctionSpace(domain.msh, P2)

        # Stress (Tensor)
        P3 = ufl.TensorElement("Lagrange", domain.msh.ufl_cell(), 2)
        self.T = dolfinx.fem.FunctionSpace(domain.msh, P3)

        P4 = ufl.FiniteElement("DG", domain.msh.ufl_cell(), 0)

        self.DG = dolfinx.fem.FunctionSpace(domain.msh, P4)

        self.first_call_to_solver = True
        self.first_call_to_surface_pressure = True

        # Store the dimension of the problem for convenience
        self.ndim = domain.msh.topology.dim

        # find hmin in mesh
        num_cells = domain.msh.topology.index_map(self.ndim).size_local
        h = dolfinx.cpp.mesh.h(domain.msh, self.ndim, range(num_cells))

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
            print(f"hmin = {self.hmin}")
            print(f"Total num dofs = {self.num_Q_dofs + self.num_V_dofs}")

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
        self.facet_dim = self.ndim - 1

        self.zero_scalar = dolfinx.fem.Constant(domain.msh, PETSc.ScalarType(0.0))
        self.zero_vec = dolfinx.fem.Constant(
            domain.msh, PETSc.ScalarType((0.0, 0.0, 0.0))
        )

        self._locate_boundary_entities(domain, params)

        # self._locate_boundary_dofs()
        self._locate_boundary_dofs_tags(domain)

        self._build_velocity_boundary_conditions(domain, params)

        self._build_pressure_boundary_conditions(domain, params)

    def _locate_boundary_entities(self, domain, params):
        """Find facet entities on boundaries

        This function builds a complete list of the facets on the x_min,
        x_max, y_min, y_max, z_min, and z_max walls plus all internal
        surfaces. It makes use of the ``x_min_facets =
        dolfinx.mesh.locate_entities_boundary()`` function from dolfinx.

        Args:
            domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object

        """

        def x_min_wall(x):
            """Identify entities on the x_min wall

            Args:
                x (np.ndarray): An array of coordinates

            Returns:
                bool: An array mask, true for points on x_min wall
            """
            return np.isclose(x[0], params.domain.x_min)

        def x_max_wall(x):
            """Identify entities on the x_max wall

            Args:
                x (np.ndarray): An array of coordinates

            Returns:
                bool: An array mask, true for points on x_max wall
            """
            return np.isclose(x[0], params.domain.x_max)

        def y_min_wall(x):
            """Identify entities on the y_min wall

            Args:
                x (np.ndarray): An array of coordinates

            Returns:
                bool: An array mask, true for points on y_min wall
            """
            return np.isclose(x[1], params.domain.y_min)

        def y_max_wall(x):
            """Identify entities on the y_max wall

            Args:
                x (np.ndarray): An array of coordinates

            Returns:
                bool: An array mask, true for points on y_max wall
            """
            return np.isclose(x[1], params.domain.y_max)

        if self.ndim == 3:

            def z_min_wall(x):
                """Identify entities on the z_min wall

                Args:
                    x (np.ndarray): An array of coordinates

                Returns:
                    bool: An array mask, true for points on z_min wall
                """
                return np.isclose(x[2], params.domain.z_min)

            def z_max_wall(x):
                """Identify entities on the z_max wall

                Args:
                    x (np.ndarray): An array of coordinates

                Returns:
                    bool: An array mask, true for points on z_max wall
                """
                return np.isclose(x[2], params.domain.z_max)

        def internal_surface(x):
            """Identify entities on the internal surfaces

            Args:
                x (np.ndarray): An array of coordinates

            Returns:
                bool: An array mask, true for points on internal surfaces
            """
            x_mid = np.logical_and(
                params.domain.x_min < x[0], x[0] < params.domain.x_max
            )
            y_mid = np.logical_and(
                params.domain.y_min < x[1], x[1] < params.domain.y_max
            )
            if self.ndim == 3:
                z_mid = np.logical_and(
                    params.domain.z_min < x[2], x[2] < params.domain.z_max
                )
                return np.logical_and(x_mid, np.logical_and(y_mid, z_mid))
            elif self.ndim == 2:
                return np.logical_and(x_mid, y_mid)

        self.x_min_facets = dolfinx.mesh.locate_entities_boundary(
            domain.msh, self.facet_dim, x_min_wall
        )
        self.x_max_facets = dolfinx.mesh.locate_entities_boundary(
            domain.msh, self.facet_dim, x_max_wall
        )
        self.y_min_facets = dolfinx.mesh.locate_entities_boundary(
            domain.msh, self.facet_dim, y_min_wall
        )
        self.y_max_facets = dolfinx.mesh.locate_entities_boundary(
            domain.msh, self.facet_dim, y_max_wall
        )
        if self.ndim == 3:
            self.z_min_facets = dolfinx.mesh.locate_entities_boundary(
                domain.msh, self.facet_dim, z_min_wall
            )
            self.z_max_facets = dolfinx.mesh.locate_entities_boundary(
                domain.msh, self.facet_dim, z_max_wall
            )

        self.internal_surface_facets = dolfinx.mesh.locate_entities_boundary(
            domain.msh, self.facet_dim, internal_surface
        )

    def _locate_boundary_dofs(self):
        """Associate degrees of freedom with boundary facets

        This function takes the previously-constructed facets from
        ``_locate_boundary_entities`` and finds the corresponding degrees of
        freedom for use in the actual boundary condition specification.

        Args:
            None

        """
        self.x_min_V_dofs = dolfinx.fem.locate_dofs_topological(
            self.V, self.facet_dim, self.x_min_facets
        )
        self.x_max_V_dofs = dolfinx.fem.locate_dofs_topological(
            self.V, self.facet_dim, self.x_max_facets
        )
        self.y_min_V_dofs = dolfinx.fem.locate_dofs_topological(
            self.V, self.facet_dim, self.y_min_facets
        )
        self.y_max_V_dofs = dolfinx.fem.locate_dofs_topological(
            self.V, self.facet_dim, self.y_max_facets
        )
        self.z_min_V_dofs = dolfinx.fem.locate_dofs_topological(
            self.V, self.facet_dim, self.z_min_facets
        )
        self.z_max_V_dofs = dolfinx.fem.locate_dofs_topological(
            self.V, self.facet_dim, self.z_max_facets
        )
        self.internal_surface_V_dofs = dolfinx.fem.locate_dofs_topological(
            self.V, self.facet_dim, self.internal_surface_facets
        )

        self.x_min_Q_dofs = dolfinx.fem.locate_dofs_topological(
            self.Q, self.facet_dim, self.x_min_facets
        )
        self.x_max_Q_dofs = dolfinx.fem.locate_dofs_topological(
            self.Q, self.facet_dim, self.x_max_facets
        )
        self.y_min_Q_dofs = dolfinx.fem.locate_dofs_topological(
            self.Q, self.facet_dim, self.y_min_facets
        )
        self.y_max_Q_dofs = dolfinx.fem.locate_dofs_topological(
            self.Q, self.facet_dim, self.y_max_facets
        )
        self.z_min_Q_dofs = dolfinx.fem.locate_dofs_topological(
            self.Q, self.facet_dim, self.z_min_facets
        )
        self.z_max_Q_dofs = dolfinx.fem.locate_dofs_topological(
            self.Q, self.facet_dim, self.z_max_facets
        )
        self.internal_surface_Q_dofs = dolfinx.fem.locate_dofs_topological(
            self.Q, self.facet_dim, self.internal_surface_facets
        )

    def _locate_boundary_dofs_tags(self, domain):
        """Associate degrees of freedom with marker functions

        This function uses the marker information in the gmsh specification to
        find the corresponding degrees of freedom for use in the actual
        boundary condition specification. Note that this method does not
        require access to the facet information computed with
        ``_locate_boundary_entities``.

        Args:
            domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object

        """
        self.x_min_V_dofs = dolfinx.fem.locate_dofs_topological(
            self.V, self.facet_dim, domain.ft.find(domain.x_min_marker)
        )

        self.x_max_V_dofs = dolfinx.fem.locate_dofs_topological(
            self.V, self.facet_dim, domain.ft.find(domain.x_max_marker)
        )

        self.y_min_V_dofs = dolfinx.fem.locate_dofs_topological(
            self.V, self.facet_dim, domain.ft.find(domain.y_min_marker)
        )

        self.y_max_V_dofs = dolfinx.fem.locate_dofs_topological(
            self.V, self.facet_dim, domain.ft.find(domain.y_max_marker)
        )
        if self.ndim == 3:
            self.z_min_V_dofs = dolfinx.fem.locate_dofs_topological(
                self.V, self.facet_dim, domain.ft.find(domain.z_min_marker)
            )

            self.z_max_V_dofs = dolfinx.fem.locate_dofs_topological(
                self.V, self.facet_dim, domain.ft.find(domain.z_max_marker)
            )

        self.internal_surface_V_dofs = dolfinx.fem.locate_dofs_topological(
            self.V, self.facet_dim, domain.ft.find(domain.internal_surface_marker)
        )

        self.x_min_Q_dofs = dolfinx.fem.locate_dofs_topological(
            self.Q, self.facet_dim, domain.ft.find(domain.x_min_marker)
        )

        self.x_max_Q_dofs = dolfinx.fem.locate_dofs_topological(
            self.Q, self.facet_dim, domain.ft.find(domain.x_max_marker)
        )

        self.y_min_Q_dofs = dolfinx.fem.locate_dofs_topological(
            self.Q, self.facet_dim, domain.ft.find(domain.y_min_marker)
        )

        self.y_max_Q_dofs = dolfinx.fem.locate_dofs_topological(
            self.Q, self.facet_dim, domain.ft.find(domain.y_max_marker)
        )

        if self.ndim == 3:
            self.z_min_Q_dofs = dolfinx.fem.locate_dofs_topological(
                self.Q, self.facet_dim, domain.ft.find(domain.z_min_marker)
            )

            self.z_max_Q_dofs = dolfinx.fem.locate_dofs_topological(
                self.Q, self.facet_dim, domain.ft.find(domain.z_max_marker)
            )

        self.internal_surface_Q_dofs = dolfinx.fem.locate_dofs_topological(
            self.Q, self.facet_dim, domain.ft.find(domain.internal_surface_marker)
        )

    def _applybc(self, value, domain, V, marker):
        """Apply a single boundary condition

        This function builds a single Dirichlet boundary condition given the value, gmsh marker, and function space.

        Args:
            value (float, dolfinx.fem.Function): Scalar or function set on the dof
            domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object
            V (:obj:`dolfinx.dolfinx.fem.FunctionSpace`): The function space on which the boundary condition will be acting
            marker (int): boundary tag created in gmsh

        Returns:
            :obj:`dolfinx.fem.dolfinx.fem.dirichletbc`: Dirichlet boundary conditions

        """
        dofs = dolfinx.fem.locate_dofs_topological(
            V, self.facet_dim, domain.ft.find(marker)
        )

        # print(value, dofs, V)

        bc = dolfinx.fem.dirichletbc(value, dofs, V)

        return bc

    def _build_velocity_boundary_conditions(self, domain, params):
        """Build all boundary conditions on velocity

        This method builds all the boundary conditions associated with velocity and stores in a list, ``bcu``.

        Args:
            domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object

        """
        # Define velocity boundary conditions
        self.bcu = []

        # x_min and x_max wall BCs
        # self.bcu.append(dolfinx.fem.dirichletbc(self.zero_vec, self.x_min_V_dofs, self.V))
        # self.bcu.append(dolfinx.fem.dirichletbc(self.zero_vec, self.x_max_V_dofs, self.V))

        # y_min and y_max wall BCs
        # self.bcu.append(dolfinx.fem.dirichletbc(self.zero_vec, self.y_min_V_dofs, self.V))
        # self.bcu.append(dolfinx.fem.dirichletbc(self.zero_vec, self.y_max_V_dofs, self.V))
        # self.bcu.append(dolfinx.fem.dirichletbc(self.zero_vec, self.z_min_V_dofs, self.V))
        # self.bcu.append(dolfinx.fem.dirichletbc(self.zero_vec, self.z_max_V_dofs, self.V))

        def _apply_type_bc(bc_name, functionspace, marker, bcname):
            """Summary

            Args:
                bc_name (TYPE): Description
                functionspace (TYPE): Description
                marker (TYPE): Description
                bcname (TYPE): Description

            Returns:
                TYPE: Description
            """
            if bc_name == "free":
                if domain.rank == 0:
                    print("free bc_name on ", bcname)
            elif bc_name == "noslip":
                bc = self._applybc(self.zero_vec, domain, functionspace, marker)
                if domain.rank == 0:
                    print("noslip bc_name on ", bcname)
            elif bc_name == "slip":
                if "y" in bcname:
                    temp_func = functionspace.sub(1)
                elif "z" in bcname:
                    temp_func = functionspace.sub(2)
                bc = self._applybc(self.zero_scalar, domain, temp_func, marker)
                if domain.rank == 0:
                    print(
                        "Slip bc_name on ",
                        bcname,
                        "boundary normal velocity in ",
                        bcname,
                        "are set to 0.",
                    )
            else:
                if domain.rank == 0:
                    print("bc_name not recognized")

                exit()

            return bc

        if self.ndim == 3:
            # iterate over all noudaries
            for bclocation in (
                "bc_ywall_max",
                "bc_ywall_min",
                "bc_zwall_max",
                "bc_zwall_min",
            ):
                if bclocation == "bc_ywall_max":
                    temp_marker = domain.y_max_marker
                    temp_bcname = "y max"
                    bc_value = params.fluid.bc_ywall_max
                elif bclocation == "bc_ywall_min":
                    temp_marker = domain.y_min_marker
                    temp_bcname = "y min"
                    bc_value = params.fluid.bc_ywall_min
                elif bclocation == "bc_zwall_max":
                    temp_marker = domain.z_max_marker
                    temp_bcname = "z max"
                    bc_value = params.fluid.bc_zwall_max
                elif bclocation == "bc_zwall_min":
                    temp_marker = domain.z_min_marker
                    temp_bcname = "z min"
                    bc_value = params.fluid.bc_zwall_min

                self.bcu.append(
                    _apply_type_bc(bc_value, self.V, temp_marker, temp_bcname)
                )

        elif self.ndim == 2:
            # iterate over all noudaries
            for bclocation in "bc_ywall_min", "bc_ywall_max":
                if bclocation == "bc_ywall_max":
                    temp_marker = domain.y_max_marker
                    temp_bcname = "y max"
                    bc_value = params.fluid.bc_ywall_max
                elif bclocation == "bc_ywall_min":
                    temp_marker = domain.y_min_marker
                    temp_bcname = "y min"
                    bc_value = params.fluid.bc_ywall_min

                self.bcu.append(
                    _apply_type_bc(bc_value, self.V, temp_marker, temp_bcname)
                )

        # self.bcu.append(self._noslip(self.zero_scalar,domain,self.V.sub(1),domain.geometry.y_min_marker))
        # self.bcu.append(self._noslip(self.zero_scalar,domain,self.V.sub(1),domain.geometry.y_max_marker))

        # # z_min and z_max wall BCs
        # self.bcu.append(dolfinx.fem.dirichletbc(self.zero_vec,domain,self.V,domain.geometry.z_min_marker))
        # self.bcu.append(dolfinx.fem.dirichletbc(self.zero_vec,domain,self.V,domain.geometry.z_max_marker))

        # interior surface (obstacle)
        self.bcu.append(
            dolfinx.fem.dirichletbc(self.zero_vec, self.internal_surface_V_dofs, self.V)
        )

        def inflow_profile_expression(x):
            """Define an inflow expression for use as boundary condition

            Args:
                x (np.ndarray): Array of coordinates

            Returns:
                np.ndarray: Value of velocity at each coordinate in input array
            """
            inflow_values = np.zeros(
                (domain.msh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType
            )

            H = 0.41
            inflow_dy = H - x[1]
            inflow_dz = H - x[2]

            u_hub = params.fluid.u_ref
            z_hub = params.pv_array.elevation

            if params.general.example == "cylinder3d":
                inflow_values[0] = (
                    16.0
                    * params.fluid.u_ref
                    * x[1]
                    * x[2]
                    * inflow_dy
                    * inflow_dz
                    / H**4
                )
            elif params.general.example == "cylinder2d":
                inflow_values[0] = (
                    4
                    * (params.fluid.u_ref)
                    * np.sin(np.pi / 8)
                    * x[1]
                    * (0.41 - x[1])
                    / (0.41**2)
                    # 16.0 * params.fluid.u_ref * x[1]  * inflow_dy / H**4
                )
            elif params.general.example == "panels":
                inflow_values[0] = (
                    (params.fluid.u_ref)
                    * np.log(((x[2]) - d0) / z0)
                    / (np.log((z_hub - d0) / z0))
                    # 4 * params.fluid.u_ref * np.sin(x[2]* np.pi/params.domain.z_max) * x[2] * (params.domain.z_max - x[2])/(params.domain.z_max**2)
                )
            elif params.general.example == "panels2d":
                inflow_values[0] = (
                    (params.fluid.u_ref)
                    * np.log(((x[1]) - d0) / z0)
                    / (np.log((z_hub - d0) / z0))
                    # 4 * params.fluid.u_ref * np.sin(x[2]* np.pi/params.domain.z_max) * x[2] * (params.domain.z_max - x[2])/(params.domain.z_max**2)
                )

            return inflow_values

        self.inflow_profile = dolfinx.fem.Function(self.V)

        if params.general.example in ["cylinder3d", "cylinder2d"]:
            self.inflow_profile.interpolate(inflow_profile_expression)

        else:
            z0 = 0.05
            d0 = 0.5
            if self.ndim == 3:
                upper_cells = dolfinx.mesh.locate_entities(
                    domain.msh, self.ndim, lambda x: x[2] > d0 + z0
                )
                lower_cells = dolfinx.mesh.locate_entities(
                    domain.msh, self.ndim, lambda x: x[2] <= d0 + z0
                )
            elif self.ndim == 2:
                upper_cells = dolfinx.mesh.locate_entities(
                    domain.msh, self.ndim, lambda x: x[1] > d0 + z0
                )
                lower_cells = dolfinx.mesh.locate_entities(
                    domain.msh, self.ndim, lambda x: x[1] <= d0 + z0
                )

            self.inflow_profile.interpolate(
                lambda x: np.zeros(
                    (domain.msh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType
                )
            )
            self.inflow_profile.interpolate(inflow_profile_expression, upper_cells)

        self.bcu.append(dolfinx.fem.dirichletbc(self.inflow_profile, self.x_min_V_dofs))

    def _build_pressure_boundary_conditions(self, domain, params):
        """Build all boundary conditions on pressure

        This method builds all the boundary conditions associated with pressure and stores in a list, ``bcp``.

        Args:
            domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object
        """
        # Define pressure boundary conditions
        self.bcp = []

        self.bcp.append(
            dolfinx.fem.dirichletbc(self.zero_scalar, self.x_max_Q_dofs, self.Q)
        )

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
        self.dpdx = dolfinx.fem.Constant(domain.msh, (0.0, 0.0, 0.0))
        self.dt_c = dolfinx.fem.Constant(domain.msh, (params.solver.dt))
        nu = dolfinx.fem.Constant(domain.msh, (params.fluid.nu))

        # Define trial and test functions for velocity
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        # Define trial and test functions for pressure
        self.p = ufl.TrialFunction(self.Q)
        self.q = ufl.TestFunction(self.Q)

        # Define functions for velocity solutions
        self.u_k = dolfinx.fem.Function(self.V, name="velocity")
        self.u_k1 = dolfinx.fem.Function(self.V, name="velocity")
        self.u_k2 = dolfinx.fem.Function(self.V)

        # Define functions for pressure solutions
        self.p_k = dolfinx.fem.Function(self.Q, name="pressure")
        self.p_k1 = dolfinx.fem.Function(self.Q, name="pressure")

        initialize_flow = True

        if initialize_flow:
            self.u_k1.x.array[:] = self.inflow_profile.x.array
            self.u_k2.x.array[:] = self.inflow_profile.x.array
            self.u_k.x.array[:] = self.inflow_profile.x.array

        # Define expressions used in variational forms
        # Crank-Nicolson velocity
        U_CN = 0.5 * (self.u + self.u_k1)

        # Adams-Bashforth velocity
        U_AB = 1.5 * self.u_k1 - 0.5 * self.u_k2

        use_eddy_viscosity = params.fluid.use_eddy_viscosity

        if use_eddy_viscosity:
            # By default, don't use any eddy viscosity
            filter_scale = ufl.CellVolume(domain.msh) ** (1.0 / domain.msh.topology.dim)

            # Strain rate tensor, 0.5*(du_i/dx_j + du_j/dx_i)
            Sij = ufl.sym(ufl.nabla_grad(U_AB))

            # ufl.sqrt(Sij*Sij)
            strainMag = (2.0 * ufl.inner(Sij, Sij)) ** 0.5

            # Smagorinsky dolfinx.fem.constant, typically close to 0.17
            Cs = 0.17

            # Eddy viscosity
            self.nu_T = Cs**2 * filter_scale**2 * strainMag

        else:
            self.nu_T = dolfinx.fem.Constant(domain.msh, 0.0)

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
        def sigma(u, p, nu):
            """Convenience expression for fluid stress, sigma

            Args:
                u (dolfinx.fem.Function): Velocity
                p (dolfinx.fem.Function): Pressure
                nu (float, dolfinx.fem.Function): Viscosity

            Returns:
                ufl.dolfinx.fem.form: Stress in fluid, $2\nu \epsilon (u)$
            """
            return 2 * nu * epsilon(u) - p * ufl.Identity(len(u))

        fractional_step_scheme = "IPCS"
        U = 0.5 * (self.u_k1 + self.u)
        n = ufl.FacetNormal(domain.msh)
        f = dolfinx.fem.Constant(
            domain.msh, (PETSc.ScalarType(0), PETSc.ScalarType(0), PETSc.ScalarType(0))
        )
        # Define variational problem for step 1: tentative velocity
        self.F1 = (
            (1.0 / self.dt_c) * ufl.inner(self.u - self.u_k1, self.v) * ufl.dx
            + ufl.inner(ufl.dot(U_AB, ufl.nabla_grad(U_CN)), self.v) * ufl.dx
            + (nu + self.nu_T) * ufl.inner(ufl.grad(U_CN), ufl.grad(self.v)) * ufl.dx
            + ufl.inner(ufl.grad(self.p_k1), self.v) * ufl.dx
            - ufl.inner(self.dpdx, self.v) * ufl.dx
        )

        self.a1 = dolfinx.fem.form(ufl.lhs(self.F1))
        self.L1 = dolfinx.fem.form(ufl.rhs(self.F1))

        # Define variational problem for step 2: pressure correction
        self.a2 = dolfinx.fem.form(
            ufl.dot(ufl.nabla_grad(self.p), ufl.nabla_grad(self.q)) * ufl.dx
        )
        self.L2 = dolfinx.fem.form(
            ufl.dot(ufl.nabla_grad(self.p_k1), ufl.nabla_grad(self.q)) * ufl.dx
            - (1.0 / self.dt_c) * ufl.div(self.u_k) * self.q * ufl.dx
        )

        # Define variational problem for step 3: velocity update
        self.a3 = dolfinx.fem.form(ufl.dot(self.u, self.v) * ufl.dx)
        self.L3 = dolfinx.fem.form(
            ufl.dot(self.u_k, self.v) * ufl.dx
            - self.dt_c * ufl.dot(ufl.nabla_grad(self.p_k - self.p_k1), self.v) * ufl.dx
        )

        # Define a function and the dolfinx.fem.form of the stress
        self.panel_stress = dolfinx.fem.Function(self.T)
        self.stress = sigma(self.u_k, self.p_k, nu + self.nu_T)

        # Define mesh normals
        # self.n = FacetNormal(domain.mesh)

        # Compute traction vector
        # self.traction = ufl.dot(self.stress, self.n)

        # Create a dolfinx.fem.form for projecting stress onto a tensor function space
        # e.g., panel_stress.assign(project(stress, T))
        # self.a4 = ufl.inner(ufl.TrialFunction(self.T), ufl.TestFunction(self.T))*ufl.dx
        # self.L4 = ufl.inner(self.stress, ufl.TestFunction(self.T))*ufl.dx

        # Create a dolfinx.fem.form for projecting CFL calculation onto DG function space
        cell_diam = ufl.CellDiameter(domain.msh)
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
            domain.msh, self.ndim, lambda x: x[0] > params.domain.x_max - 1
        )
        self.flux_plane = dolfinx.fem.Function(self.V)
        self.flux_plane.interpolate(
            lambda x: (np.ones(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1])),
            outlet_cells,
        )

        # self.flux_plane = Expression('x[0] < cutoff ? 0.0 : 1.0', cutoff=domain.x_range[1]-1.0, degree=1)
        self.flux_dx = ufl.Measure("dx", domain=domain.msh)  # not needed ?

        # self.vol = dolfinx.fem.petsc.assemble_matrix(self.flux_plane*self.flux_dx)
        # form1 = dolfinx.fem.form(self.flux_plane*self.flux_dx)
        # self.vol = dolfinx.fem.petsc.assemble_vector(form1)
        # self.J_initial = float(fem.petsc.assemble_matrix(self.flux_plane*self.u_k[0]*self.flux_dx)/self.vol)

        # if mpi_info['rank'] == 0:
        #     print('J_initial = %f' % (self.J_initial))

        # self.J_history = [self.J_initial]
        self.dpdx_history = [0.0]

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
        self.A5 = dolfinx.fem.petsc.assemble_matrix(self.a5)

        self.A1.assemble()
        self.A2.assemble()
        self.A3.assemble()
        self.A5.assemble()

        self.b1 = dolfinx.fem.petsc.assemble_vector(self.L1)
        self.b2 = dolfinx.fem.petsc.assemble_vector(self.L2)
        self.b3 = dolfinx.fem.petsc.assemble_vector(self.L3)
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
        self.solver_3.getPC().setType(params.solver.solver2_pc)
        self.solver_3.setFromOptions()

        self.solver_5 = PETSc.KSP().create(self.comm)
        self.solver_5.setOperators(self.A5)
        self.solver_5.setType("cg")
        self.solver_5.getPC().setType("jacobi")
        self.solver_5.setFromOptions()

    def solve(self, params):
        """Solve for a single timestep advancement

        Here we perform the three-step solution process (tentative velocity,
        pressure correction, velocity update) to advance the fluid simulation
        a single timestep. Additionally, we calculate the new CFL number
        associated with the latest velocity solution.

        Args:
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object
        """
        if self.first_call_to_solver:
            if self.rank == 0:
                print("Starting Fluid Solution")

            self._assemble_system(params)

        # Calculate the tentative velocity
        self._solver_step_1(params)

        # Calculate the change in pressure to enforce incompressibility
        self._solver_step_2(params)

        # Update the velocity according to the pressure field
        self._solver_step_3(params)

        # Compute the CFL number
        self.compute_cfl()

        # Update new -> old variables
        self.u_k2.x.array[:] = self.u_k1.x.array
        self.u_k1.x.array[:] = self.u_k.x.array
        self.p_k1.x.array[:] = self.p_k.x.array

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
        with self.b2.localForm() as loc:
            loc.set(0)

        self.b2 = dolfinx.fem.petsc.assemble_vector(self.b2, self.L2)

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
        with self.b3.localForm() as loc:
            loc.set(0)

        self.b3 = dolfinx.fem.petsc.assemble_vector(self.b3, self.L3)

        self.b3.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        )

        self.solver_3.solve(self.b3, self.u_k.vector)
        self.u_k.x.scatter_forward()

    def compute_cfl(self):
        """Solve for the CFL number

        Using the velocity, timestep size, and cell sizes, we calculate a CFL
        number at every mesh cell. From that, we select the single highest
        value of CFL number and record it for the purposes of monitoring
        simulation stability.

        Args:
            None
        """
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

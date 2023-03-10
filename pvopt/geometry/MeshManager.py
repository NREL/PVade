from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import VectorFunctionSpace, FunctionSpace
from dolfinx.cpp import mesh as cppmesh

from mpi4py import MPI
import gmsh
import numpy as np
import os
import time
import ufl
import dolfinx
from petsc4py import PETSc

import matplotlib.pyplot as plt

import meshio

# from pvopt.geometry.panels.DomainCreation   import *
class Domain:
    """This class creates the computational domain"""

    def __init__(self, params):
        """The class is initialised here

        Args:
            params (input parameters): input parameters available in the input file
        """
        self.first_move_mesh = True

        # Get MPI communicators
        self.comm = params.comm
        self.rank = params.rank
        self.num_procs = params.num_procs

        # Store a full copy of params on this object
        self.params = params

        problem = self.params.general.example

        if problem == "panels":
            from pvopt.geometry.panels.DomainCreation import DomainCreation

        elif problem == "cylinder3d":
            from pvopt.geometry.cylinder3d.DomainCreation import DomainCreation

        elif problem == "cylinder2d":
            from pvopt.geometry.cylinder2d.DomainCreation import DomainCreation

        self.geometry = DomainCreation(self.params)

    def build(self):

        # Only rank 0 builds the geometry and meshes the domain
        if self.rank == 0:
            self.geometry.build()
            self.geometry.mark_surfaces()
            self.geometry.set_length_scales()

            if self.params.fluid.periodic:
                self._enforce_periodicity(self.geometry.gmsh_model)

            self._generate_mesh(self.geometry.gmsh_model)

        # All ranks receive their portion of the mesh from rank 0 (like an MPI scatter)
        self.mesh, self.mt, self.ft = gmshio.model_to_mesh(
            self.geometry.gmsh_model, self.comm, 0
        )

        self.ndim = self.mesh.topology.dim

        # Specify names for the mesh elements
        self.mesh.name = self.params.general.example
        self.mt.name = f"{self.mesh.name}_cells"
        self.ft.name = f"{self.mesh.name}_facets"

    def read(self):
        """Read the mesh from external file located in output/mesh"""
        if self.rank == 0:
            print("Reading the mesh from file ...")
        with dolfinx.io.XDMFFile(
            MPI.COMM_WORLD, self.params.general.output_dir_mesh + "/mesh.xdmf", "r"
        ) as xdmf:
            self.mesh = xdmf.read_mesh(name="Grid")

        self.mesh.topology.create_connectivity(
            self.mesh.topology.dim - 1, self.mesh.topology.dim
        )
        with XDMFFile(
            MPI.COMM_WORLD, self.params.general.output_dir_mesh + "/mesh_mf.xdmf", "r"
        ) as infile:
            self.ft = infile.read_meshtags(self.mesh, "Grid")
        if self.rank == 0:
            print("Done.")

    def _enforce_periodicity(self, gmsh_model):

        # TODO: Make this a generic mapping depending on which walls are marked for peridic BCs
        # TODO: Copy code to enforce periodicity from old generate_and_convert_3d_meshes.py

        # Mark the front/back walls for periodic mapping
        front_back_translation = [
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            self.y_span,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
        ]

        gmsh_model.mesh.setPeriodic(
            2, self.dom_tags["back"], self.dom_tags["front"], front_back_translation
        )

        # Mark the left/right walls for periodic mapping
        left_right_translation = [
            1,
            0,
            0,
            self.x_span,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
        ]

        gmsh_model.mesh.setPeriodic(
            2, self.dom_tags["right"], self.dom_tags["left"], left_right_translation
        )

    def _generate_mesh(self, gmsh_model):
        print("Starting mesh generation... ", end="")

        # Generate the mesh
        tic = time.time()
        # gmsh.option.setNumber("Mesh.Algorithm", 8)
        # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        # # gmsh.option.setNumber("Mesh.RecombineAll", 1)
        # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        # gmsh_model.mesh.generate(3)
        # gmsh_model.mesh.setOrder(2)
        # gmsh_model.mesh.optimize("Netgen")
        gmsh_model.mesh.generate(3)
        toc = time.time()
        print("Finished.")
        print(f"Total meshing time = {toc-tic:.1f} s")

    def move_mesh(self, tt):


        if self.first_move_mesh:
            # Build a function space for the rotation (a vector of degree 1)
            ve1 = ufl.VectorElement("Lagrange", self.mesh.ufl_cell(), 1)
            self.V1 = dolfinx.fem.FunctionSpace(self.mesh, ve1)

            self.mesh_motion = dolfinx.fem.Function(self.V1, name="mesh_displacement")
            self.mesh_motion_bc = dolfinx.fem.Function(self.V1, name="mesh_displacement_bc")
            self.total_mesh_displacement = dolfinx.fem.Function(self.V1, name="total_mesh_disp")

            # Build a function space for the distance (a scalar of degree 1)
            fe1 = ufl.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
            self.Q1 = dolfinx.fem.FunctionSpace(self.mesh, fe1)

            self.distance = dolfinx.fem.Function(self.Q1)

            def _all_panel_surfaces(x):
                eps = 1.0e-5

                x_mid = np.logical_and(self.params.domain.x_min + eps < x[0], x[0] < self.params.domain.x_max - eps)
                y_mid = np.logical_and(self.params.domain.y_min + eps < x[1], x[1] < self.params.domain.y_max - eps)
                z_mid = np.logical_and(self.params.domain.z_min + eps < x[2], x[2] < self.params.domain.z_max - eps)

                all_panel_bool = np.logical_and(x_mid,  np.logical_and(y_mid, z_mid))
                
                return all_panel_bool

            def _all_domain_edges(x):
                eps = 1.0e-5

                x_edge = np.logical_or(x[0] < self.params.domain.x_min + eps,  self.params.domain.x_max - eps < x[0])
                y_edge = np.logical_or(x[1] < self.params.domain.y_min + eps,  self.params.domain.y_max - eps < x[1])
                z_edge = np.logical_or(x[2] < self.params.domain.z_min + eps,  self.params.domain.z_max - eps < x[2])

                all_domain_bool = np.logical_or(x_edge, np.logical_or(y_edge, z_edge))
                
                return all_domain_bool

            self.facet_dim = self.ndim - 1

            all_surface_facets = dolfinx.mesh.locate_entities_boundary(
                self.mesh, self.facet_dim, _all_panel_surfaces
            )

            all_edge_facets = dolfinx.mesh.locate_entities_boundary(
                self.mesh, self.facet_dim, _all_domain_edges
            )

            self.all_surface_V_dofs = dolfinx.fem.locate_dofs_topological(
                self.V1, self.facet_dim, all_surface_facets
            )

            self.all_edge_V_dofs = dolfinx.fem.locate_dofs_topological(
                self.V1, self.facet_dim, all_edge_facets
            )


        def mesh_motion_expression(x, tt):
            motion_array = np.zeros((self.mesh.geometry.dim, x.shape[1]))
            
            for k in range(self.params.pv_array.num_rows):

                panel_x_center = (k*self.params.pv_array.spacing[0])
                panel_z_center = self.params.pv_array.elevation
                
                x_shift = np.copy(x)
                x_shift[0, :] -= panel_x_center
                x_shift[2, :] -= panel_z_center
                                
                num_cycles_to_complete = 1
                time_per_cycle = self.params.solver.t_final/num_cycles_to_complete

                amplitude_degrees = 15.0

                theta_old = np.radians(amplitude_degrees)*np.sin(2.0*np.pi*(tt-self.params.solver.dt)/time_per_cycle)
                theta_new = np.radians(amplitude_degrees)*np.sin(2.0*np.pi*tt/time_per_cycle)

                theta = theta_new - theta_old

                if k % 2 == 1:
                    theta *= -1.0

                # theta = np.radians(15.0)
                
                # Rz = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                #                [np.sin(theta),  np.cos(theta), 0.0],
                #                [          0.0,            0.0, 1.0]])

                Ry = np.array([[ np.cos(theta), 0.0, np.sin(theta)],
                               [          0.0,  1.0,           0.0],
                               [-np.sin(theta), 0.0, np.cos(theta)]])
                
                x_rot = np.dot(Ry, x_shift)
                
                x_delta = x_rot - x_shift
                
                dist = x_shift[0, :]*x_shift[0, :] + x_shift[2, :]*x_shift[2, :]

                mask = dist < (0.5*self.params.pv_array.spacing[0])**2

                motion_array[:, mask] = x_delta[:, mask]
                            
            return motion_array
                
        def mesh_motion_expression_helper(tt):
            
            return lambda x: mesh_motion_expression(x, tt)

        self.mesh_motion_bc.interpolate(mesh_motion_expression_helper(tt))



        zero_vec = dolfinx.fem.Constant(self.mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))

        self.bcx = []
        self.bcx.append(dolfinx.fem.dirichletbc(self.mesh_motion_bc, self.all_surface_V_dofs))
        self.bcx.append(dolfinx.fem.dirichletbc(zero_vec, self.all_edge_V_dofs, self.V1))

        if self.first_move_mesh:
            u = ufl.TrialFunction(self.V1)
            v = ufl.TestFunction(self.V1)

            self.a = dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
            self.L = dolfinx.fem.form(ufl.inner(zero_vec, v) * ufl.dx)

            self.A = dolfinx.fem.petsc.assemble_matrix(self.a, bcs=self.bcx)
            self.A.assemble()

            self.b = dolfinx.fem.petsc.assemble_vector(self.L)

            self.mesh_motion_solver = PETSc.KSP().create(self.comm)
            self.mesh_motion_solver.setOperators(self.A)
            self.mesh_motion_solver.setType("cg")
            self.mesh_motion_solver.getPC().setType("jacobi")
            self.mesh_motion_solver.setFromOptions()

        self.A.zeroEntries()
        self.A = dolfinx.fem.petsc.assemble_matrix(self.A, self.a, bcs=self.bcx)
        self.A.assemble()

        with self.b.localForm() as loc:
            loc.set(0)

        self.b = dolfinx.fem.petsc.assemble_vector(self.b, self.L)

        dolfinx.fem.petsc.apply_lifting(self.b, [self.a], [self.bcx])

        self.b.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        )

        dolfinx.fem.petsc.set_bc(self.b, self.bcx)

        self.mesh_motion_solver.solve(self.b, self.mesh_motion.vector)
        self.mesh_motion.x.scatter_forward()

        vals = self.mesh_motion.vector.array.reshape(-1, 3)
        nn = np.shape(vals)[0]

        test_new_ghost_method = True

        if test_new_ghost_method:
            with self.mesh_motion.vector.localForm() as vals_local:
                vals = vals_local.array
                vals = vals.reshape(-1, 3)

            self.mesh.geometry.x[:, :] += vals[:, :]

        else:
            self.mesh.geometry.x[:nn, :] += vals[:, :]



        # bc_name = "dummy.xdmf"

        # if self.first_move_mesh:
        #     with XDMFFile(self.comm, bc_name, "w") as xdmf_file:
        #         xdmf_file.write_mesh(self.mesh)
        #         xdmf_file.write_function(self.mesh_motion, tt)

        # else:
        #     with XDMFFile(self.comm, bc_name, "a") as xdmf_file:
        #         xdmf_file.write_function(self.mesh_motion, tt)


        # if self.first_move_mesh:
        #     self.total_mesh_displacement.vector.array[:] = self.mesh.geometry.x[:nn, :].flatten()

        self.total_mesh_displacement.vector.array[:] += self.mesh_motion.vector.array

        # mesh_velocity = 0
        # return mesh_velocity

        # plt.figure(figsize=(10, 4), dpi=200)
        # plt.scatter(self.mesh.geometry.x[:, 0], self.mesh.geometry.x[:, 2], s=0.5)
        # plt.gca().set_aspect(1.0)
        # plt.savefig(f'output/yo_{tt*1000:.0f}.png')
        # plt.close()

        self.first_move_mesh = False

        # return self.mesh


    def write_mesh_file(self):
        """
        TODO: when saving a mesh file using only dolfinx functions
        it's possible certain elements of the data aren't preserved
        and that the mesh won't be able to be properly read in later
        on. MAKE SURE YOU CAN SAVE A MESH USING ONLY DOLFINX FUNCTIONS
        AND THEN READ IN THAT SAME MESH WITHOUT A LOSS OF CAPABILITY.
        """

        if self.rank == 0:
            # Save the *.msh file and *.vtk file (the latter is for visualization only)
            print(
                "Writing Mesh to %s... " % (self.params.general.output_dir_mesh), end=""
            )

            if os.path.exists(self.params.general.output_dir_mesh) == False:
                os.mkdir(self.params.general.output_dir_mesh)
            gmsh.write("%s/mesh.msh" % (self.params.general.output_dir_mesh))
            gmsh.write("%s/mesh.vtk" % (self.params.general.output_dir_mesh))

            def create_mesh(mesh, clean_points, cell_type):
                cells = mesh.get_cells_type(cell_type)
                cell_data = mesh.get_cell_data("gmsh:physical", cell_type)

                out_mesh = meshio.Mesh(
                    points=clean_points,
                    cells={cell_type: cells},
                    cell_data={"name_to_read": [cell_data]},
                )
                return out_mesh

            mesh_from_file = meshio.read(
                f"{self.params.general.output_dir_mesh}/mesh.msh"
            )
            pts = mesh_from_file.points
            tetra_mesh = create_mesh(mesh_from_file, pts, "tetra")
            tri_mesh = create_mesh(mesh_from_file, pts, "triangle")

            meshio.write(f"{self.params.general.output_dir_mesh}/mesh.xdmf", tetra_mesh)
            meshio.write(
                f"{self.params.general.output_dir_mesh}/mesh_mf.xdmf", tri_mesh
            )
            print("Done.")

    def test_mesh_functionspace(self):

        P2 = ufl.VectorElement("Lagrange", self.mesh.ufl_cell(), 2)
        P1 = ufl.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
        V = FunctionSpace(self.mesh, P2)
        Q = FunctionSpace(self.mesh, P1)

        local_rangeV = V.dofmap.index_map.local_range
        dofsV = np.arange(*local_rangeV)

        local_rangeQ = Q.dofmap.index_map.local_range
        dofsQ = np.arange(*local_rangeQ)

        # coords = self.mesh.coordinates()

        nodes_dim = 0
        self.mesh.topology.create_connectivity(nodes_dim, 0)
        num_nodes_owned_by_proc = self.mesh.topology.index_map(nodes_dim).size_local
        geometry_entitites = cppmesh.entities_to_geometry(
            self.mesh,
            nodes_dim,
            np.arange(num_nodes_owned_by_proc, dtype=np.int32),
            False,
        )
        points = self.mesh.geometry.x

        coords = points[:]

        print(f"Rank {self.rank} owns {num_nodes_owned_by_proc} nodes\n{coords}")

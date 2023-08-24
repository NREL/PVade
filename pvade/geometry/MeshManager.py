import gmsh
import numpy as np
import os
import time
import ufl
import dolfinx
import meshio
import warnings

from importlib import import_module

# from pvade.geometry.panels.DomainCreation   import *
class FSIDomain:
    """
    This class creates the computational domain for 3D examples(3D panels, 3D cylinder)
    """

    def __init__(self, params):
        """The class is initialised here
            Depending on the example we are solving, we import the corresponding DomainCrearion file
            We define markers which will be used for boundary tag asignment

        Args:
            params (input parameters): input parameters available in the input file
        """

        # Get MPI communicators
        self.comm = params.comm
        self.rank = params.rank
        self.num_procs = params.num_procs

        self._get_domain_markers(params)

    def _get_domain_markers(self,params):
        self.domain_markers = {}

        # Fluid Facet Markers
        self.domain_markers["x_min"] = {"idx": 1, "entity": "facet", "gmsh_tags": []}
        self.domain_markers["x_max"] = {"idx": 2, "entity": "facet", "gmsh_tags": []}
        self.domain_markers["y_min"] = {"idx": 3, "entity": "facet", "gmsh_tags": []}
        self.domain_markers["y_max"] = {"idx": 4, "entity": "facet", "gmsh_tags": []}
        self.domain_markers["z_min"] = {"idx": 5, "entity": "facet", "gmsh_tags": []}
        self.domain_markers["z_max"] = {"idx": 6, "entity": "facet", "gmsh_tags": []}

        self.domain_markers["internal_surface"] = {
            "idx": 7,
            "entity": "facet",
            "gmsh_tags": [],
        }

        # Cell Markers
        self.domain_markers["fluid"] = {"idx": 8, "entity": "cell", "gmsh_tags": []}
        self.domain_markers["structure"] = {"idx": 9, "entity": "cell", "gmsh_tags": []}

        # Structure Facet Markers
        if params.general.geometry_module == "panels2d" or params.general.geometry_module == "panels3d":
            for panel_id in range(params.pv_array.stream_rows * params.pv_array.span_rows):
                marker = 9 + np.array([1,2,3,4,5,6])+6*(panel_id)
                panel_marker = 100*(panel_id+1) 
                self.domain_markers[f"bottom_{panel_id}"] = {"idx": marker[0], "entity": "facet", "gmsh_tags": []}
                self.domain_markers[f"top_{panel_id}"]    = {"idx": marker[1], "entity": "facet", "gmsh_tags": []}
                self.domain_markers[f"left_{panel_id}"]   = {"idx": marker[2], "entity": "facet", "gmsh_tags": []}
                self.domain_markers[f"right_{panel_id}"]  = {"idx": marker[3], "entity": "facet", "gmsh_tags": []}
                self.domain_markers[f"back_{panel_id}"]   = {"idx": marker[4], "entity": "facet", "gmsh_tags": []}
                self.domain_markers[f"front_{panel_id}"]  = {"idx": marker[5], "entity": "facet", "gmsh_tags": []}
                # Cell Markers 
                self.domain_markers[f"panel_{panel_id}"] = {"idx": panel_marker, "entity": "cell", "gmsh_tags": []}
        elif params.general.geometry_module == "cylinder3d" or params.general.geometry_module == "cylinder2d":
            self.domain_markers[f"cylinder"] = {"idx": 100, "entity": "cell", "gmsh_tags": []}
            self.domain_markers[f"cylinder_side"]  = {"idx": 10, "entity": "facet", "gmsh_tags": []}



    def build(self, params):
        """This function call builds the geometry, marks the boundaries and creates a mesh using Gmsh."""

        domain_creation_module = (
            f"pvade.geometry.{params.general.geometry_module}.DomainCreation"
        )
        try:
            # This is equivalent to "import pvade.geometry.{params.general.geometry_module}.DomainCreation as dcm"
            dcm = import_module(domain_creation_module)
        except:
            raise ValueError(f"Could not import {domain_creation_module}")

        self.geometry = dcm.DomainCreation(params)

        # Only rank 0 builds the geometry and meshes the domain
        if self.rank == 0:
            if params.general.geometry_module == "panels3d" and params.general.fluid_analysis == True and params.general.structural_analysis == True :
                self.geometry.build_FSI(params)
            elif params.general.geometry_module == "panels3d" and params.general.fluid_analysis == False and params.general.structural_analysis == True :
                self.geometry.build_structure(params)
            else :
                self.geometry.build_FSI(params)
            
            # Build the domain markers for each surface and cell
            if hasattr(self.geometry, "domain_markers"):
                # If the "build" process created domain markers, use those directly...
                self.domain_markers = self.geometry.domain_markers

            else:
                # otherwise, call the method mark_surfaces to identify them after the fact
                self.domain_markers = self.geometry.mark_surfaces(
                    params, self.domain_markers
                )

            self.geometry.set_length_scales(params, self.domain_markers)

            if params.fluid.periodic:
                self._enforce_periodicity()

            self._generate_mesh()

        # When finished, rank 0 needs to tell other ranks about how the domain_markers dictionary was created
        # and what values it holds. This is important now since the number of indices "idx" generated in the
        # geometry module differs from what's prescribed in the init of this class.
        # TODO: GET RID OF THE DEFAULT DICTIONARY INITIALIZATION AND LET EACH GEOMETRY MODULE
        # CREATE THEIR OWN AND JUST HAVE RANK 0 ALWAYS BROADCAST IT.
        self.domain_markers = self.comm.bcast(self.domain_markers, root=0)

        # All ranks receive their portion of the mesh from rank 0 (like an MPI scatter)
        self.msh, self.cell_tags, self.facet_tags = dolfinx.io.gmshio.model_to_mesh(
            self.geometry.gmsh_model, self.comm, 0, partitioner=dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
            )

        self.ndim = self.msh.topology.dim

        # Specify names for the mesh elements
        self.msh.name = "mesh_total"
        self.cell_tags.name = "cell_tags"
        self.facet_tags.name = "facet_tags"


        if params.general.geometry_module == "panels3d" and params.general.fluid_analysis == True and params.general.structural_analysis == True :
            self._create_submeshes_from_parent()
            self._transfer_mesh_tags_to_submeshes(params)

        self._save_submeshes_for_reload_hack(params)

    def _save_submeshes_for_reload_hack(self, params):

        self.write_mesh_files(params)
        self.read_mesh_files(params.general.output_dir_mesh, params)

        # if params.general.fluid_analysis == True and params.general.structural_analysis == True :
        #     sub_domain_list = ["fluid", "structure"]
        #     for sub_domain_name in sub_domain_list:
        #         mesh_filename = os.path.join(params.general.output_dir_mesh, f"temp_{sub_domain_name}.xdmf")
        #         sub_domain = getattr(self, sub_domain_name)
        #         sub_domain.msh.name = "temp_mesh"

        #         # Write this submesh to immediately read it
        #         with dolfinx.io.XDMFFile(self.comm, mesh_filename, "w") as xdmf:
        #             xdmf.write_mesh(sub_domain.msh)
        #             xdmf.write_meshtags(sub_domain.cell_tags)
        #             sub_domain.msh.topology.create_connectivity(self.ndim-1, self.ndim)
        #             xdmf.write_meshtags(sub_domain.facet_tags)

        #         # Read the just-written mesh
        #         with dolfinx.io.XDMFFile(self.comm, mesh_filename, "r") as xdmf:
        #             submesh = xdmf.read_mesh(name=sub_domain.msh.name)
        #             cell_tags = xdmf.read_meshtags(submesh, name="cell_tags")
        #             ndim = submesh.topology.dim
        #             submesh.topology.create_connectivity(ndim - 1, ndim)
        #             facet_tags = xdmf.read_meshtags(submesh, name="facet_tags")

        #         sub_domain.msh = submesh
        #         sub_domain.facet_tags = facet_tags
        #         sub_domain.cell_tags = cell_tags

        # elif params.general.geometry_module == "panels3d" and params.general.fluid_analysis == False and params.general.structural_analysis == True :
        #         sub_domain_name = "structure"

        #         mesh_filename = os.path.join(params.general.output_dir_mesh, f"temp_{sub_domain_name}.xdmf")
        #         # sub_domain = getattr(self, sub_domain_list[0])
        #         # sub_domain.msh.name = "temp_mesh"

        #         # self.msh.name  = "temp_mesh"

        #         # Write this submesh to immediately read it
        #         with dolfinx.io.XDMFFile(self.comm, mesh_filename, "w") as xdmf:
        #             xdmf.write_mesh(self.msh)
        #             xdmf.write_meshtags(self.cell_tags)
        #             self.msh.topology.create_connectivity(self.ndim-1, self.ndim)
        #             xdmf.write_meshtags(self.facet_tags)

        #         # Read the just-written mesh
        #         with dolfinx.io.XDMFFile(self.comm, mesh_filename, "r") as xdmf:
        #             mesh = xdmf.read_mesh(name=self.msh.name)
        #             cell_tags = xdmf.read_meshtags(mesh, name="cell_tags")
        #             ndim = mesh.topology.dim
        #             mesh.topology.create_connectivity(ndim - 1, ndim)
        #             facet_tags = xdmf.read_meshtags(mesh, name="facet_tags")


        #         class FSISubDomain:
        #             pass

        #         sub_domain = FSISubDomain()

        #         sub_domain.msh = mesh
        #         sub_domain.facet_tags = facet_tags
        #         sub_domain.cell_tags = cell_tags

        #         setattr(self, sub_domain_name, sub_domain)


    def _create_submeshes_from_parent(self):
        """Create submeshes from a parent mesh by cell tags.

        This function uses the cell tags to identify meshes that
        are part of the structure or fluid domain and saves them
        as separate mesh entities inside a new object such that meshes
        are accessed like domain.fluid.msh or domain.structure.msh.

        """

        for sub_domain_name in ["fluid", "structure"]:
            if self.rank == 0:
                print(f"Creating {sub_domain_name} submesh")

            # Get the idx associated with either "fluid" or "structure"
            marker_id = self.domain_markers[sub_domain_name]["idx"]

            # Find all cells where cell tag = marker_id
            submesh_cells = self.cell_tags.find(marker_id)

            # Use those found cells to construct a new mesh
            submesh, entity_map, vertex_map, geom_map = dolfinx.mesh.create_submesh(
                self.msh, self.ndim, submesh_cells
            )

            class FSISubDomain:
                pass

            submesh.topology.create_connectivity(3, 2)

            sub_domain = FSISubDomain()

            sub_domain.msh = submesh
            sub_domain.entity_map = entity_map
            sub_domain.vertex_map = vertex_map
            sub_domain.geom_map = geom_map

            setattr(self, sub_domain_name, sub_domain)

    def _transfer_mesh_tags_to_submeshes(self, params):
        facet_dim = self.ndim - 1

        f_map = self.msh.topology.index_map(facet_dim)

        # Get the total number of facets in the parent mesh
        num_facets = f_map.size_local + f_map.num_ghosts
        all_values = np.zeros(num_facets, dtype=np.int32)

        # Assign non-zero facet tags using the facet tag indices
        all_values[self.facet_tags.indices] = self.facet_tags.values

        cell_to_facet = self.msh.topology.connectivity(self.ndim, facet_dim)

        for sub_domain_name in ["fluid", "structure"]:
            # Get the sub-domain object
            sub_domain = getattr(self, sub_domain_name)

            # Initialize facets and create connectivity between cells and facets
            # sub_domain.msh.topology.create_connectivity_all()
            sub_domain.msh.topology.create_entities(facet_dim)
            sub_f_map = sub_domain.msh.topology.index_map(facet_dim)
            sub_domain.msh.topology.create_connectivity(self.ndim, facet_dim)

            sub_cell_to_facet = sub_domain.msh.topology.connectivity(
                self.ndim, facet_dim
            )

            sub_num_facets = sub_f_map.size_local + sub_f_map.num_ghosts
            sub_values = np.empty(sub_num_facets, dtype=np.int32)

            for k, entity in enumerate(sub_domain.entity_map):
                child_facets = sub_cell_to_facet.links(k)
                parent_facets = cell_to_facet.links(entity)

                for child, parent in zip(child_facets, parent_facets):
                    sub_values[child] = all_values[parent]

            sub_cell_map = sub_domain.msh.topology.index_map(self.ndim)
            sub_num_cells = sub_cell_map.size_local + sub_cell_map.num_ghosts

            sub_domain.cell_tags = dolfinx.mesh.meshtags(
                sub_domain.msh,
                sub_domain.msh.topology.dim,
                np.arange(sub_num_cells, dtype=np.int32),
                np.ones(sub_num_cells, dtype=np.int32),
            )
            sub_domain.cell_tags.name = "cell_tags"

            sub_domain.facet_tags = dolfinx.mesh.meshtags(
                sub_domain.msh,
                sub_domain.msh.topology.dim - 1,
                np.arange(sub_num_facets, dtype=np.int32),
                sub_values,
            )
            sub_domain.facet_tags.name = "facet_tags"

            # sub_domain.cell_tags = dolfinx.mesh.meshtags(
            #     sub_domain.msh,
            #     sub_domain.msh.topology.dim,
            #     np.arange(sub_num_facets, dtype=np.int32),
            #     np.ones(sub_num_facets),
            # )

            # IS THIS REDUNDANT? SEEMS LIKE WE DO IT ABOVE
            # sub_cell_map = sub_domain.msh.topology.index_map(self.ndim) 
            # sub_num_cells = sub_cell_map.size_local + sub_cell_map.num_ghosts 
            # sub_domain.cell_tags = dolfinx.mesh.meshtags( sub_domain.msh, sub_domain.msh.topology.dim, np.arange(sub_num_cells, dtype=np.int32), np.ones(sub_num_cells, dtype=np.int32), ) 
            # sub_domain.cell_tags.name = "cell_tags" 
            # sub_domain.facet_tags = dolfinx.mesh.meshtags( sub_domain.msh, sub_domain.msh.topology.dim - 1, np.arange(sub_num_facets, dtype=np.int32), sub_values, ) 
            # sub_domain.facet_tags.name = "facet_tags"
            
    def _enforce_periodicity(self):
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

        self.geometry.gmsh_model.mesh.setPeriodic(
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

        self.geometry.gmsh_model.mesh.setPeriodic(
            2, self.dom_tags["right"], self.dom_tags["left"], left_right_translation
        )

    def _generate_mesh(self):
        """This function call generates the mesh."""
        print("Starting mesh generation... ", end="")

        # Generate the mesh
        tic = time.time()

        if self.geometry.ndim == 3:
            self._generate_mesh_3d()
        elif self.geometry.ndim == 2:
            self._generate_mesh_2d()

        toc = time.time()

        if self.rank == 0:
            print("Finished.")
            print(f"Total meshing time = {toc-tic:.1f} s")

    def _generate_mesh_3d(self):
        # Mesh.Algorithm 2D mesh algorithm
        # (1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay for Quads, 9: Packing of Parallelograms)
        # Default value: 6
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        # 3D mesh algorithm
        # (1: Delaunay, 3: Initial mesh only, 4: Frontal, 7: MMG3D, 9: R-tree, 10: HXT)
        # Default value: 1
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)

        # Mesh recombination algorithm
        # (0: simple, 1: blossom, 2: simple full-quad, 3: blos- som full-quad)
        # Default value: 1
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)

        # Apply recombination algorithm to all surfaces, ignoring per-surface spec Default value: 0
        # gmsh.option.setNumber("Mesh.RecombineAll", 1)

        self.geometry.gmsh_model.mesh.generate(3)
        self.geometry.gmsh_model.mesh.setOrder(2)

        self.geometry.gmsh_model.mesh.optimize("Relocate3D")
        self.geometry.gmsh_model.mesh.generate(3)

    def _generate_mesh_2d(self):
        # Mesh.Algorithm 2D mesh algorithm
        # (1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay for Quads, 9: Packing of Parallelograms)
        # Default value: 6
        gmsh.option.setNumber("Mesh.Algorithm", 2)

        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        # gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

        self.geometry.gmsh_model.mesh.generate(3)
        self.geometry.gmsh_model.mesh.setOrder(2)

        self.geometry.gmsh_model.mesh.optimize("Netgen")
        self.geometry.gmsh_model.mesh.generate(3)

    def write_mesh_files(self, params):

        # Attempt to save both the fluid and structure subdomains
        sub_domain_list = ["fluid", "structure"]

        for sub_domain_name in sub_domain_list:
            try:
                if self.rank == 0:
                    print(f"Beginning write of {sub_domain_name} mesh.")

                # Get the fluid or structure object from self
                sub_domain = getattr(self, sub_domain_name)

            except:
                if self.rank == 0:
                    print(f"Could not find subdomain {sub_domain_name}, not writing this mesh.") 

            else:
                # Write this subdomain mesh to a file
                mesh_name = f"{sub_domain_name}_mesh.xdmf"
                mesh_filename = os.path.join(params.general.output_dir_mesh, mesh_name)

                # Write this submesh
                with dolfinx.io.XDMFFile(self.comm, mesh_filename, "w") as xdmf:
                    sub_domain.msh.name = mesh_name
                    xdmf.write_mesh(sub_domain.msh)
                    xdmf.write_meshtags(sub_domain.cell_tags)
                    sub_domain.msh.topology.create_connectivity(self.ndim-1, self.ndim)
                    xdmf.write_meshtags(sub_domain.facet_tags)

                # Write a gmsh copy too
                gmsh_mesh_name = f"{sub_domain_name}_mesh.msh"
                gmsh_mesh_filename = os.path.join(params.general.output_dir_mesh, gmsh_mesh_name)
                gmsh.write(gmsh_mesh_filename)

                if self.rank == 0:
                    print(f"Finished writing {sub_domain_name} mesh.")

    def read_mesh_files(self, read_mesh_dir, params):
        """Read the mesh from an external file.
        The User can load an existing mesh file (mesh.xdmf)
        and use it to solve the CFD/CSD problem
        """

        sub_domain_list = ["fluid", "structure"]

        for sub_domain_name in sub_domain_list:
            mesh_name = f"{sub_domain_name}_mesh.xdmf"
            mesh_filename = os.path.join(read_mesh_dir, mesh_name)

            try:
                if self.rank == 0:
                    print("Reading {sub_domain_name} mesh.")

                # Read the subdomain mesh
                with dolfinx.io.XDMFFile(self.comm, mesh_filename, "r") as xdmf:
                    submesh = xdmf.read_mesh(name=mesh_name)
                    cell_tags = xdmf.read_meshtags(submesh, name="cell_tags")
                    ndim = submesh.topology.dim
                    submesh.topology.create_connectivity(ndim - 1, ndim)
                    facet_tags = xdmf.read_meshtags(submesh, name="facet_tags")

            except:
                if self.rank == 0:
                    print(f"Could not find subdomain {sub_domain_name} mesh file, not reading this mesh.") 

            else:
                class FSISubDomain:
                    pass

                submesh.topology.create_connectivity(3, 2)

                sub_domain = FSISubDomain()

                sub_domain.msh = submesh
                sub_domain.cell_tags = cell_tags
                sub_domain.facet_tags = facet_tags

                # These elements do not need to be created when reading a mesh
                # they are only used in the transfer of facet tags, and since
                # those can be read directly from a file, we don't need these
                sub_domain.entity_map = None
                sub_domain.vertex_map = None
                sub_domain.geom_map = None

                setattr(self, sub_domain_name, sub_domain)

                if self.rank == 0:
                    print("Finished read {sub_domain_name} mesh.")

    def test_mesh_functionspace(self):
        P2 = ufl.VectorElement("Lagrange", self.msh.ufl_cell(), 2)
        P1 = ufl.FiniteElement("Lagrange", self.msh.ufl_cell(), 1)
        V = dolfinx.fem.FunctionSpace(self.msh, P2)
        Q = dolfinx.fem.FunctionSpace(self.msh, P1)

        local_rangeV = V.dofmap.index_map.local_range
        dofsV = np.arange(*local_rangeV)

        local_rangeQ = Q.dofmap.index_map.local_range
        dofsQ = np.arange(*local_rangeQ)

        nodes_dim = 0
        self.msh.topology.create_connectivity(nodes_dim, 0)
        num_nodes_owned_by_proc = self.msh.topology.index_map(nodes_dim).size_local
        geometry_entitites = dolfinx.cpp.mesh.entities_to_geometry(
            self.msh,
            nodes_dim,
            np.arange(num_nodes_owned_by_proc, dtype=np.int32),
            False,
        )
        points = self.msh.geometry.x

        coords = points[:]

        print(f"Rank {self.rank} owns {num_nodes_owned_by_proc} nodes\n{coords}")


    def test_submesh_transfer(self, params):
        P2 = ufl.VectorElement("Lagrange", self.msh.ufl_cell(), 2)
        # P2 = ufl.FiniteElement("Lagrange", self.fluid.msh.ufl_cell(), 1)

        V_fluid = dolfinx.fem.FunctionSpace(self.fluid.msh, P2)
        V_struc = dolfinx.fem.FunctionSpace(self.structure.msh, P2)
        V_all = dolfinx.fem.FunctionSpace(self.msh, P2)

        u_all = dolfinx.fem.Function(V_all)
        u_fluid = dolfinx.fem.Function(V_fluid)
        u_struc = dolfinx.fem.Function(V_struc)

        u_fluid.x.array[:] = -5.0
        u_struc.x.array[:] = -5.0

        from petsc4py import PETSc

        def fluid_function_value_setter(x):

            fluid_function_vals = np.zeros((3, x.shape[1]), dtype=PETSc.ScalarType)

            fluid_function_vals[0, :] = x[0]
            fluid_function_vals[1, :] = x[1]
            fluid_function_vals[2, :] = x[2]

            return fluid_function_vals

        u_fluid.interpolate(fluid_function_value_setter)

        mesh_filename = os.path.join(params.general.output_dir_sol, "transfer_fluid.xdmf")
        with dolfinx.io.XDMFFile(self.comm, mesh_filename, "w") as fp:
            fp.write_mesh(self.fluid.msh)
            fp.write_function(u_fluid, 0)

        u_struc.interpolate(u_fluid)

        mesh_filename = os.path.join(params.general.output_dir_sol, "transfer_struc.xdmf")
        with dolfinx.io.XDMFFile(self.comm, mesh_filename, "w") as fp:
            fp.write_mesh(self.structure.msh)
            fp.write_function(u_struc, 0)

        u_fluid.interpolate(u_struc)

        mesh_filename = os.path.join(params.general.output_dir_sol, "transfer_fluid.xdmf")
        with dolfinx.io.XDMFFile(self.comm, mesh_filename, "a") as fp:
            # fp.write_mesh(self.fluid.msh)
            fp.write_function(u_fluid, 1)

        with dolfinx.io.VTKFile(self.comm, self.results_filename_vtk, "w") as file:
            file.write_mesh(domain.structure.msh)
            file.write_function(elasticity.uh, 0.0)

import gmsh
import numpy as np
import os
import time
import ufl
import dolfinx
import meshio

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

        self._get_domain_markers()

    def _get_domain_markers(self):
        self.domain_markers = {}

        # Facet Markers
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

    def build(self, params):
        """This function call builds the geometry, marks the boundaries and creates a mesh using Gmsh."""

        domain_creation_module = (
            f"pvade.geometry.{params.general.example}.DomainCreation"
        )
        try:
            # This is equivalent to "import pvade.geometry.{params.general.example}.DomainCreation as dcm"
            dcm = import_module(domain_creation_module)
        except:
            raise ValueError(f"Could not import {domain_creation_module}")

        self.geometry = dcm.DomainCreation(params)

        # Only rank 0 builds the geometry and meshes the domain
        if self.rank == 0:
            self.geometry.build(params)
            self.domain_markers = self.geometry.mark_surfaces(
                params, self.domain_markers
            )
            self.geometry.set_length_scales(params, self.domain_markers)

            if params.fluid.periodic:
                self._enforce_periodicity()

            self._generate_mesh()

        # All ranks receive their portion of the mesh from rank 0 (like an MPI scatter)
        self.msh, self.cell_tags, self.facet_tags = dolfinx.io.gmshio.model_to_mesh(
            self.geometry.gmsh_model, self.comm, 0
        )

        self.ndim = self.msh.topology.dim

        # Specify names for the mesh elements
        self.msh.name = "mesh_total"
        self.cell_tags.name = "cell_tags"
        self.facet_tags.name = "facet_tags"

        self._create_submeshes_from_parent()

        self._transfer_mesh_tags_to_submeshes(params)

    def _create_submeshes_from_parent(self):
        """Create submeshes from a parent mesh by cell tags.

        This function uses the cell tags to identify meshes that
        are part of the structure or fluid domain and saves them
        as separate mesh entities inside a new object such that meshes
        are accessed like domain.fluid.msh or domain.structure.msh.

        """

        for marker_key in ["fluid", "structure"]:
            if self.rank == 0:
                print(f"Creating {marker_key} submesh")

            # Get the idx associated with either "fluid" or "structure"
            marker_id = self.domain_markers[marker_key]["idx"]

            # Find all cells where cell tag = marker_id
            submesh_cells = self.cell_tags.find(marker_id)

            # Use those found cells to construct a new mesh
            submesh, entity_map, vertex_map, geom_map = dolfinx.mesh.create_submesh(
                self.msh, self.ndim, submesh_cells
            )

            class FSISubDomain:
                pass

            sub_domain = FSISubDomain()

            sub_domain.msh = submesh
            sub_domain.entity_map = entity_map
            sub_domain.vertex_map = vertex_map
            sub_domain.geom_map = geom_map

            setattr(self, marker_key, sub_domain)

    def _transfer_mesh_tags_to_submeshes(self, params):
        facet_dim = self.ndim - 1

        f_map = self.msh.topology.index_map(facet_dim)

        # Get the total number of facets in the parent mesh
        num_facets = f_map.size_local + f_map.num_ghosts
        all_values = np.zeros(num_facets, dtype=np.int32)

        # Assign non-zero facet tags using the facet tag indices
        all_values[self.facet_tags.indices] = self.facet_tags.values

        cell_to_facet = self.msh.topology.connectivity(self.ndim, facet_dim)

        for marker_key in ["fluid", "structure"]:
            # Get the sub-domain object
            sub_domain = getattr(self, marker_key)

            # Initialize facets and create connectivity between cells and facets
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

            sub_domain.facet_tags = dolfinx.mesh.meshtags(
                sub_domain.msh,
                sub_domain.msh.topology.dim - 1,
                np.arange(sub_num_facets, dtype=np.int32),
                sub_values,
            )

            sub_domain.cell_tags = dolfinx.mesh.meshtags(
                sub_domain.msh,
                sub_domain.msh.topology.dim,
                np.arange(sub_num_facets, dtype=np.int32),
                np.ones(sub_num_facets),
            )

            mesh_filename = os.path.join(
                params.general.output_dir_mesh, f"mesh_{marker_key}.xdmf"
            )

            with dolfinx.io.XDMFFile(self.comm, mesh_filename, "w") as fp:
                fp.write_mesh(sub_domain.msh)
                sub_domain.msh.topology.create_connectivity(facet_dim, self.ndim)
                fp.write_meshtags(sub_domain.facet_tags)

        for marker_key in ["fluid", "structure"]:
            # Get the sub-domain object
            sub_domain = getattr(self, marker_key)

            internal_facets_from_tag = sub_domain.facet_tags.find(3)

            tol = 0

            def internal_surface(x):
                x_mid = np.logical_and(
                    params.domain.x_min + tol < x[0], x[0] < params.domain.x_max - tol
                )
                y_mid = np.logical_and(
                    params.domain.y_min + tol < x[1], x[1] < params.domain.y_max - tol
                )
                if self.ndim == 3:
                    z_mid = np.logical_and(
                        params.domain.z_min + tol < x[2],
                        x[2] < params.domain.z_max - tol,
                    )

                    return np.logical_and(x_mid, np.logical_and(y_mid, z_mid))

                elif self.ndim == 2:
                    return np.logical_and(x_mid, y_mid)

            def x_min_wall(x):
                """Identify entities on the x_min wall

                Args:
                    x (np.ndarray): An array of coordinates

                Returns:
                    bool: An array mask, true for points on x_min wall
                """
                return np.isclose(x[1], params.domain.y_min)

            internal_facets = dolfinx.mesh.locate_entities_boundary(
                sub_domain.msh, self.ndim - 1, x_min_wall
            )

            print(
                f"from tag, nn = {len(internal_facets_from_tag)} vs from og, nn = {len(internal_facets)}"
            )

            assert np.allclose(internal_facets_from_tag, internal_facets)

    def read(self, read_path, params):
        """Read the mesh from an external file.
        The User can load an existing mesh file (mesh.xdmf)
        and use it to solve the CFD/CSD problem
        """
        if self.rank == 0:
            print("Reading the mesh from file ...")

        path_to_mesh = os.path.join(read_path, "mesh.xdmf")

        with dolfinx.io.XDMFFile(self.comm, path_to_mesh, "r") as xdmf:
            self.msh = xdmf.read_mesh(name="mesh_total")
            self.cell_tags = xdmf.read_meshtags(self.msh, name="cell_tags")

            self.ndim = self.msh.topology.dim
            self.msh.topology.create_connectivity(self.ndim - 1, self.ndim)

            self.facet_tags = xdmf.read_meshtags(self.msh, "facet_tags")

        self._create_submeshes_from_parent()

        self._transfer_mesh_tags_to_submeshes(params)

        if self.rank == 0:
            print("Done.")

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

    def write_mesh_file(self, params):
        """TODO: when saving a mesh file using only dolfinx functions
        it's possible certain elements of the data aren't preserved
        and that the mesh won't be able to be properly read in later
        on. MAKE SURE YOU CAN SAVE A MESH USING ONLY DOLFINX FUNCTIONS
        AND THEN READ IN THAT SAME MESH WITHOUT A LOSS OF CAPABILITY.
        """

        if self.rank == 0:
            # Save the *.msh file and *.vtk file (the latter is for visualization only)
            print(f"Writing Mesh to {params.general.output_dir_mesh}...")

            if os.path.exists(params.general.output_dir_mesh) == False:
                os.makedirs(params.general.output_dir_mesh)

            mesh_filename = os.path.join(params.general.output_dir_mesh, "mesh.xdmf")

            with dolfinx.io.XDMFFile(self.comm, mesh_filename, "w") as fp:
                fp.write_mesh(self.msh)
                fp.write_meshtags(self.cell_tags)
                fp.write_meshtags(self.facet_tags)

            # mesh_filename = os.path.join(params.general.output_dir_mesh, "mesh_structure.xdmf")

            # with dolfinx.io.XDMFFile(self.comm, mesh_filename, "w") as fp:
            #     fp.write_mesh(self.msh_structure)
            #     # fp.write_meshtags(self.cell_tags)
            #     # fp.write_meshtags(self.facet_tags)

            # mesh_filename = os.path.join(params.general.output_dir_mesh, "mesh_fluid.xdmf")

            # with dolfinx.io.XDMFFile(self.comm, mesh_filename, "w") as fp:
            #     fp.write_mesh(self.msh_fluid)
            #     # fp.write_meshtags(self.cell_tags)
            #     # fp.write_meshtags(self.facet_tags)

            print("Done.")

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

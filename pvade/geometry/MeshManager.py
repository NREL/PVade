import gmsh
import numpy as np
import os
import time
import ufl
import dolfinx

# import meshio
import yaml
from petsc4py import PETSc

from importlib import import_module

from numba import jit


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
        if params.general.fluid_analysis == True:
            self.first_move_mesh = True
        else:
            self.first_move_mesh = False

        self._get_domain_markers(params)

        self.numpy_pt_total_array = None

    def _get_domain_markers(self, params):
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
        if (
            params.general.geometry_module == "panels2d"
            or params.general.geometry_module == "panels3d"
            or params.general.geometry_module == "heliostats3d"
        ):
            for panel_id in range(
                params.pv_array.stream_rows * params.pv_array.span_rows
            ):
                marker = 9 + np.array([1, 2, 3, 4, 5, 6]) + 6 * (panel_id)
                panel_marker = 100 * (panel_id + 1)
                self.domain_markers[f"bottom_{panel_id}"] = {
                    "idx": marker[0],
                    "entity": "facet",
                    "gmsh_tags": [],
                }
                self.domain_markers[f"top_{panel_id}"] = {
                    "idx": marker[1],
                    "entity": "facet",
                    "gmsh_tags": [],
                }
                self.domain_markers[f"left_{panel_id}"] = {
                    "idx": marker[2],
                    "entity": "facet",
                    "gmsh_tags": [],
                }
                self.domain_markers[f"right_{panel_id}"] = {
                    "idx": marker[3],
                    "entity": "facet",
                    "gmsh_tags": [],
                }
                self.domain_markers[f"back_{panel_id}"] = {
                    "idx": marker[4],
                    "entity": "facet",
                    "gmsh_tags": [],
                }
                self.domain_markers[f"front_{panel_id}"] = {
                    "idx": marker[5],
                    "entity": "facet",
                    "gmsh_tags": [],
                }
                # Cell Markers
                self.domain_markers[f"panel_{panel_id}"] = {
                    "idx": panel_marker,
                    "entity": "cell",
                    "gmsh_tags": [],
                }
        elif (
            params.general.geometry_module == "cylinder3d"
            or params.general.geometry_module == "cylinder2d"
        ):
            self.domain_markers[f"cylinder"] = {
                "idx": 100,
                "entity": "cell",
                "gmsh_tags": [],
            }
            self.domain_markers[f"cylinder_side"] = {
                "idx": 10,
                "entity": "facet",
                "gmsh_tags": [],
            }

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
            if (
                (
                    params.general.geometry_module == "panels3d"
                    or params.general.geometry_module == "heliostats3d"
                )
                and params.general.fluid_analysis == True
                and params.general.structural_analysis == True
            ):
                self.geometry.build_FSI(params)
            elif (
                (
                    params.general.geometry_module == "panels3d"
                    or params.general.geometry_module == "heliostats3d"
                )
                and params.general.fluid_analysis == False
                and params.general.structural_analysis == True
            ):
                self.geometry.build_structure(params)
            else:
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

            # if params.general.fluid_analysis == True:
            self.numpy_pt_total_array = self.geometry.numpy_pt_total_array

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
        self.numpy_pt_total_array = self.comm.bcast(self.numpy_pt_total_array, root=0)

        # All ranks receive their portion of the mesh from rank 0 (like an MPI scatter)
        self.msh, self.cell_tags, self.facet_tags = dolfinx.io.gmshio.model_to_mesh(
            self.geometry.gmsh_model,
            self.comm,
            0,
            partitioner=dolfinx.mesh.create_cell_partitioner(
                dolfinx.mesh.GhostMode.shared_facet
            ),
        )

        self.ndim = self.msh.topology.dim

        self.msh.topology.create_connectivity(self.ndim, self.ndim - 1)

        # Specify names for the mesh elements
        self.msh.name = "mesh_total"
        self.cell_tags.name = "cell_tags"
        self.facet_tags.name = "facet_tags"

        if (
            params.general.geometry_module == "panels3d"
            or params.general.geometry_module == "heliostats3d"
            or params.general.geometry_module == "flag2d"
        ):
            self._create_submeshes_from_parent(params)
            self._transfer_mesh_tags_to_submeshes(params)

        self._save_submeshes_for_reload_hack(params)

        if params.general.fluid_analysis == True:
            # Create all forms that will eventually be used for mesh rotation/movement
            # Build a function space for the rotation (a vector of degree 1)
            vec_el_1 = ufl.VectorElement("Lagrange", self.fluid.msh.ufl_cell(), 1)
            self.V1 = dolfinx.fem.FunctionSpace(self.fluid.msh, vec_el_1)

            # vec_el_1_og = ufl.VectorElement("Lagrange", self.fluid_undeformed.msh.ufl_cell(), 1)
            self.V1_undeformed = dolfinx.fem.FunctionSpace(
                self.fluid_undeformed.msh, vec_el_1
            )

            self.fluid_mesh_displacement = dolfinx.fem.Function(
                self.V1, name="fluid_mesh_displacement"
            )
            self.fluid_mesh_displacement_bc = dolfinx.fem.Function(
                self.V1, name="fluid_mesh_displacement_bc"
            )
            self.fluid_mesh_displacement_bc_undeformed = dolfinx.fem.Function(
                self.V1_undeformed, name="fluid_mesh_displacement_bc"
            )
            self.total_mesh_displacement = dolfinx.fem.Function(
                self.V1, name="total_mesh_disp"
            )

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

    def _create_submeshes_from_parent(self, params):
        """Create submeshes from a parent mesh by cell tags.

        This function uses the cell tags to identify meshes that
        are part of the structure or fluid domain and saves them
        as separate mesh entities inside a new object such that meshes
        are accessed like domain.fluid.msh or domain.structure.msh.

        """
        submesh_list = []

        if params.general.fluid_analysis == True:
            submesh_list.append("fluid")
        if params.general.structural_analysis == True:
            submesh_list.append("structure")

        for sub_domain_name in submesh_list:
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

            submesh.topology.create_connectivity(self.ndim, self.ndim - 1)

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

        submesh_list = []

        if params.general.fluid_analysis == True:
            submesh_list.append("fluid")
        if params.general.structural_analysis == True:
            submesh_list.append("structure")

        for sub_domain_name in submesh_list:
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
        # gmsh.option.setNumber("Mesh.Algorithm", 2)

        # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        # # gmsh.option.setNumber("Mesh.RecombineAll", 1)
        # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

        # self.geometry.gmsh_model.mesh.generate(3)
        # self.geometry.gmsh_model.mesh.setOrder(2)

        # self.geometry.gmsh_model.mesh.optimize("Netgen")
        self.geometry.gmsh_model.mesh.generate(2)

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
                    print(
                        f"Could not find subdomain {sub_domain_name}, not writing this mesh."
                    )

            else:
                # Write this subdomain mesh to a file
                mesh_name = f"{sub_domain_name}_mesh.xdmf"
                mesh_filename = os.path.join(params.general.output_dir_mesh, mesh_name)

                # Write this submesh
                with dolfinx.io.XDMFFile(self.comm, mesh_filename, "w") as xdmf:
                    sub_domain.msh.name = mesh_name
                    xdmf.write_mesh(sub_domain.msh)
                    xdmf.write_meshtags(sub_domain.cell_tags)
                    sub_domain.msh.topology.create_connectivity(
                        self.ndim - 1, self.ndim
                    )
                    xdmf.write_meshtags(sub_domain.facet_tags)

                # Write a gmsh copy too
                gmsh_mesh_name = f"{sub_domain_name}_mesh.msh"
                gmsh_mesh_filename = os.path.join(
                    params.general.output_dir_mesh, gmsh_mesh_name
                )
                gmsh.write(gmsh_mesh_filename)

                if self.rank == 0:
                    print(f"Finished writing {sub_domain_name} mesh.")

        # Finally, dump a yaml file of the domain_markers
        # necessary for setting BCs in case this mesh directory is read for a new run
        yaml_name = "domain_markers.yaml"
        yaml_filename = os.path.join(params.general.output_dir_mesh, yaml_name)

        with open(yaml_filename, "w") as fp:
            yaml.dump(self.domain_markers, fp)

        # Save a version of the numpy_pt_array_total]
        csv_name = "numpy_fixation_points.csv"
        csv_filename = os.path.join(params.general.output_dir_mesh, csv_name)
        header = "start_x,start_y,start_z,end_x,end_y,end_z,"
        np.savetxt(
            csv_filename, self.numpy_pt_total_array, delimiter=",", header=header
        )

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
                    print(f"Reading {sub_domain_name} mesh.")

                # Read the subdomain mesh
                with dolfinx.io.XDMFFile(self.comm, mesh_filename, "r") as xdmf:
                    submesh = xdmf.read_mesh(name=mesh_name)
                    cell_tags = xdmf.read_meshtags(submesh, name="cell_tags")
                    ndim = submesh.topology.dim
                    submesh.topology.create_connectivity(ndim - 1, ndim)
                    facet_tags = xdmf.read_meshtags(submesh, name="facet_tags")

            except:
                if self.rank == 0:
                    print(
                        f"Could not find subdomain {sub_domain_name} mesh file, not reading this mesh."
                    )

            else:

                class FSISubDomain:
                    pass

                if not hasattr(self, "ndim"):
                    self.ndim = submesh.topology.dim
                else:
                    assert self.ndim == submesh.topology.dim

                submesh.topology.create_connectivity(self.ndim, self.ndim - 1)

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

                if sub_domain_name == "fluid":
                    domain_ufl = ufl.Mesh(
                        self.fluid.msh.ufl_domain().ufl_coordinate_element()
                    )
                    fluid_undeformed = dolfinx.mesh.Mesh(
                        self.comm,
                        self.fluid.msh.topology,
                        self.fluid.msh.geometry,
                        domain_ufl,
                    )

                    fluid_undeformed.topology.create_connectivity(
                        self.ndim, self.ndim - 1
                    )

                    sub_domain_undeformed = FSISubDomain()

                    sub_domain_undeformed.msh = fluid_undeformed
                    sub_domain_undeformed.cell_tags = cell_tags
                    sub_domain_undeformed.facet_tags = facet_tags

                    # These elements do not need to be created when reading a mesh
                    # they are only used in the transfer of facet tags, and since
                    # those can be read directly from a file, we don't need these
                    sub_domain_undeformed.entity_map = None
                    sub_domain_undeformed.vertex_map = None
                    sub_domain_undeformed.geom_map = None

                    setattr(self, "fluid_undeformed", sub_domain_undeformed)

                    # assert np.all(self.fluid.msh.geometry.x[:] == self.fluid_undeformed.msh.geometry.x[:])
                    # assert np.shape(self.fluid.msh.geometry.x[:]) == np.shape(self.fluid_undeformed.msh.geometry.x[:])

            if self.rank == 0:
                print(f"Finished read {sub_domain_name} mesh.")

        self.ndim = submesh.topology.dim

        yaml_name = "domain_markers.yaml"
        yaml_filename = os.path.join(read_mesh_dir, yaml_name)

        if params.general.fluid_analysis == True:
            with open(yaml_filename, "r") as fp:
                self.domain_markers = yaml.safe_load(fp)

        csv_name = "numpy_fixation_points.csv"
        csv_filename = os.path.join(read_mesh_dir, csv_name)
        self.numpy_pt_total_array = np.genfromtxt(
            csv_filename, delimiter=",", skip_header=1
        )

        if params.general.fluid_analysis == True:
            # Create all forms that will eventually be used for mesh rotation/movement
            # Build a function space for the rotation (a vector of degree 1)
            vec_el_1 = ufl.VectorElement("Lagrange", self.fluid.msh.ufl_cell(), 1)
            self.V1 = dolfinx.fem.FunctionSpace(self.fluid.msh, vec_el_1)

            self.V1_undeformed = dolfinx.fem.FunctionSpace(
                self.fluid_undeformed.msh, vec_el_1
            )

            self.fluid_mesh_displacement = dolfinx.fem.Function(
                self.V1, name="fluid_mesh_displacement"
            )
            self.fluid_mesh_displacement_bc = dolfinx.fem.Function(
                self.V1, name="fluid_mesh_displacement_bc"
            )
            self.fluid_mesh_displacement_bc_undeformed = dolfinx.fem.Function(
                self.V1_undeformed, name="fluid_mesh_displacement_bc"
            )
            self.total_mesh_displacement = dolfinx.fem.Function(
                self.V1, name="total_mesh_disp"
            )

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

        mesh_filename = os.path.join(
            params.general.output_dir_sol, "transfer_fluid.xdmf"
        )
        with dolfinx.io.XDMFFile(self.comm, mesh_filename, "w") as fp:
            fp.write_mesh(self.fluid.msh)
            fp.write_function(u_fluid, 0)

        u_struc.interpolate(u_fluid)

        mesh_filename = os.path.join(
            params.general.output_dir_sol, "transfer_struc.xdmf"
        )
        with dolfinx.io.XDMFFile(self.comm, mesh_filename, "w") as fp:
            fp.write_mesh(self.structure.msh)
            fp.write_function(u_struc, 0)

        u_fluid.interpolate(u_struc)

        mesh_filename = os.path.join(
            params.general.output_dir_sol, "transfer_fluid.xdmf"
        )
        with dolfinx.io.XDMFFile(self.comm, mesh_filename, "a") as fp:
            # fp.write_mesh(self.fluid.msh)
            fp.write_function(u_fluid, 1)

        with dolfinx.io.VTKFile(self.comm, self.results_filename_vtk, "w") as file:
            file.write_mesh(domain.structure.msh)
            file.write_function(elasticity.uh, 0.0)

    def _calc_distance_to_panel_surface(self, params, min_dist_cutoff=1.0e-6):
        # Get the coordinates of each point from the mesh objects
        fluid_pts = self.fluid.msh.geometry.x
        local_structure_pts = self.structure.msh.geometry.x

        # These are *local* to the process, and we need the *global* structure coordinates
        # get the size of this chunk
        local_num_pts = np.shape(local_structure_pts)[0]

        # Initialize an array to hold the chunk sizes on each process
        global_num_pts_list = np.zeros(self.num_procs, dtype=np.int64)

        # Let each rank know how many nodes of the structure are held by all other ranks
        self.comm.Allgather(
            np.array(local_num_pts, dtype=np.int64), global_num_pts_list
        )

        # Sum them to allocate an array to hold all the coordinates
        global_num_pts = int(np.sum(global_num_pts_list))
        global_structure_pts = np.zeros((global_num_pts, 3), dtype=np.float64)

        # Gather all the coordinates from each process into a global array representing all the points of the structure
        # After this step, local_structure_pts holds *every* node of the structure mesh
        # Note that we do not use self.ndim as the number of columns since even 2D meshes have a z column
        self.comm.Allgatherv(
            local_structure_pts.astype(np.float64),
            (global_structure_pts, 3 * global_num_pts_list),
        )

        @jit(nopython=True)
        def find_shortest_distances(fluid_pts, structure_pts):
            vec = np.zeros(np.shape(fluid_pts)[0])

            for k, pt in enumerate(fluid_pts):
                delta_x = pt - structure_pts
                dist_2 = np.sum(delta_x**2, axis=1)
                # dist = np.linalg.norm(delta_x, axis=1)
                min_dist = np.sqrt(np.amin(dist_2))
                if min_dist < min_dist_cutoff:
                    min_dist = min_dist_cutoff
                vec[k] = min_dist

            return vec

        vec = find_shortest_distances(fluid_pts, global_structure_pts)

        # Build a function space for the distance (a scalar of degree 1)
        scalar_el_1 = ufl.FiniteElement("Lagrange", self.fluid.msh.ufl_cell(), 1)
        self.Q1 = dolfinx.fem.FunctionSpace(self.fluid.msh, scalar_el_1)

        self.distance = dolfinx.fem.Function(self.Q1)
        nn = np.shape(self.distance.vector.array)[0]

        self.distance.vector.array[:] = vec[0:nn]
        self.distance.x.scatter_forward()

        dist_filename = os.path.join(
            params.general.output_dir_mesh, "distance_field.xdmf"
        )
        with dolfinx.io.XDMFFile(self.comm, dist_filename, "w") as xdmf_file:
            xdmf_file.write_mesh(self.fluid.msh)
            xdmf_file.write_function(self.distance, 0.0)

    def _force_interface_node_matching(self):
        # Get the coordinates of each point from the mesh objects
        fluid_pts = self.fluid.msh.geometry.x
        fluid_boundary_pts = fluid_pts[self.all_interior_V_dofs, :]
        local_structure_pts = self.structure.msh.geometry.x

        # These are *local* to the process, and we need the *global* structure coordinates
        # get the size of this chunk
        local_num_pts = np.shape(local_structure_pts)[0]

        # Initialize an array to hold the chunk sizes on each process
        global_num_pts_list = np.zeros(self.num_procs, dtype=np.int64)

        # Let each rank know how many nodes of the structure are held by all other ranks
        self.comm.Allgather(
            np.array(local_num_pts, dtype=np.int64), global_num_pts_list
        )

        # Sum them to allocate an array to hold all the coordinates
        global_num_pts = int(np.sum(global_num_pts_list))
        global_structure_pts = np.zeros((global_num_pts, 3), dtype=np.float64)

        # Gather all the coordinates from each process into a global array representing all the points of the structure
        # After this step, local_structure_pts holds *every* node of the structure mesh
        self.comm.Allgatherv(
            local_structure_pts.astype(np.float64),
            (global_structure_pts, 3 * global_num_pts_list),
        )

        @jit(nopython=True)
        def find_closest_structure_idx(fluid_pts, structure_pts):
            idx_vec = np.zeros(np.shape(fluid_pts)[0], dtype=np.int64)

            for k, pt in enumerate(fluid_pts):
                delta_x = pt - structure_pts
                dist_2 = np.sum(delta_x**2, axis=1)

                # Find what row this minimum occurs on, that
                # is the row from structure_pts to copy
                min_dist_id = np.argmin(dist_2)

                idx_vec[k] = min_dist_id

            return idx_vec

        if not hasattr(self, "idx_vec"):
            self.idx_vec = find_closest_structure_idx(
                fluid_boundary_pts, global_structure_pts
            )

        # Apply the result in the following way:
        # fluid_mesh_point[interior_surface_pt_idx, :] = structure_mesh_point[min_dist_idx, :]
        # Where the interior surface point index comes from the boundary condition functions
        # And the minimum distance idx is the one identified in find_closest_structure_idx

        correction = (
            global_structure_pts[self.idx_vec, :]
            - self.fluid.msh.geometry.x[self.all_interior_V_dofs, :]
        )

        self.fluid.msh.geometry.x[self.all_interior_V_dofs, :] = global_structure_pts[
            self.idx_vec, :
        ]

        for k, dof in enumerate(self.all_interior_V_dofs):
            if 3 * dof > np.shape(self.fluid_mesh_displacement.vector.array[:])[0] - 1:
                break
            else:
                self.fluid_mesh_displacement.vector.array[3 * dof] += correction[k, 0]
                self.fluid_mesh_displacement.vector.array[3 * dof + 1] += correction[
                    k, 1
                ]
                self.fluid_mesh_displacement.vector.array[3 * dof + 2] += correction[
                    k, 2
                ]

        self.fluid_mesh_displacement.x.scatter_forward()

    # def custom_interpolate(self, elasticity):
    #
    # NOTE: this is possibly no longer necessary since forcing
    # point matching at the interface means we *should* be able to
    # use the built in interpolate function versus
    # writing our own nearest neighbor function tranfer each time.
    #
    #     # Get the coordinates of each point from the mesh objects
    #     fluid_pts = self.V1.tabulate_dof_coordinates()
    #     local_structure_pts = elasticity.V.tabulate_dof_coordinates()
    #     local_u_delta_vals = elasticity.u_delta.vector.array[:]

    #     # These are *local* to the process, and we need the *global* vector of values
    #     local_num_vals = np.shape(local_u_delta_vals)[0]

    #     local_nn = int(local_num_vals/3)

    #     # Initialize an array to hold the chunk sizes on each process
    #     global_num_vals_list = np.zeros(self.num_procs, dtype=np.int64)

    #     # Let each rank know how many values of the vector are held by all other ranks
    #     self.comm.Allgather(
    #         np.array(local_num_vals, dtype=np.int64), global_num_vals_list
    #     )

    #     # Sum them to allocate an array to hold all the coordinates
    #     global_num_vals = int(np.sum(global_num_vals_list))
    #     global_vals = np.zeros(global_num_vals, dtype=np.float64)

    #     global_nn = int(global_num_vals/3)
    #     global_structure_pts = np.zeros((global_nn, self.ndim), dtype=np.float64)

    #     # Gather all the coordinates from each process into a global array representing all the points of the structure
    #     # After this step, local_structure_pts holds *every* node of the structure mesh
    #     self.comm.Allgatherv(
    #         local_u_delta_vals.astype(np.float64),
    #         (global_vals, global_num_vals_list),
    #     )

    #     self.comm.Allgatherv(
    #         local_structure_pts[:local_nn, :].astype(np.float64),
    #         (global_structure_pts, global_num_vals_list),
    #     )

    #     @jit(nopython=True)
    #     def nearest_neighbor_interp(fluid_pts, structure_pts, global_vals):
    #         vec = np.zeros(3*np.shape(fluid_pts)[0])

    #         for k, pt in enumerate(fluid_pts):
    #             delta_x = pt - structure_pts
    #             dist_2 = np.sum(delta_x**2, axis=1)
    #             # dist = np.linalg.norm(delta_x, axis=1)
    #             min_id = np.argmin(dist_2)

    #             vec[3*k] = global_vals[3*min_id]
    #             vec[3*k+1] = global_vals[3*min_id+1]
    #             vec[3*k+2] = global_vals[3*min_id+2]

    #         return vec

    #     vec = nearest_neighbor_interp(fluid_pts, global_structure_pts, global_vals)

    #     return vec

    def move_mesh(self, elasticity, params):
        if self.first_move_mesh:
            # Save the un-moved coordinates for future reference
            # self.fluid.msh.initial_position = self.fluid.msh.geometry.x[:, :]
            # self.structure.msh.initial_position = self.structure.msh.geometry.x[:, :]

            self._calc_distance_to_panel_surface(params)

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

                all_interior_surfaces = np.logical_and(
                    x_mid, np.logical_and(y_mid, z_mid)
                )

                return all_interior_surfaces

            def _all_exterior_surfaces(x):
                eps = 1.0e-5

                x_edge = np.logical_or(
                    x[0] < params.domain.x_min + eps, params.domain.x_max - eps < x[0]
                )
                y_edge = np.logical_or(
                    x[1] < params.domain.y_min + eps, params.domain.y_max - eps < x[1]
                )
                z_edge = np.logical_or(
                    x[2] < params.domain.z_min + eps, params.domain.z_max - eps < x[2]
                )

                all_exterior_surfaces = np.logical_or(
                    x_edge, np.logical_or(y_edge, z_edge)
                )

                return all_exterior_surfaces

            def _all_xmin_xmax_surfaces(x):
                eps = 1.0e-5

                x_edge = np.logical_or(
                    x[0] < params.domain.x_min + eps, params.domain.x_max - eps < x[0]
                )

                return x_edge

            def _all_ymin_ymax_surfaces(x):
                eps = 1.0e-5

                y_edge = np.logical_or(
                    x[1] < params.domain.y_min + eps, params.domain.y_max - eps < x[1]
                )

                return y_edge

            self.facet_dim = self.ndim - 1

            all_interior_facets = dolfinx.mesh.locate_entities_boundary(
                self.fluid.msh, self.facet_dim, _all_interior_surfaces
            )

            all_exterior_facets = dolfinx.mesh.locate_entities_boundary(
                self.fluid.msh, self.facet_dim, _all_exterior_surfaces
            )

            all_xmin_xmax_facets = dolfinx.mesh.locate_entities_boundary(
                self.fluid.msh, self.facet_dim, _all_xmin_xmax_surfaces
            )

            all_ymin_ymax_facets = dolfinx.mesh.locate_entities_boundary(
                self.fluid.msh, self.facet_dim, _all_ymin_ymax_surfaces
            )

            self.all_interior_V_dofs = dolfinx.fem.locate_dofs_topological(
                self.V1, self.facet_dim, all_interior_facets
            )

            self.all_exterior_V_dofs = dolfinx.fem.locate_dofs_topological(
                self.V1, self.facet_dim, all_exterior_facets
            )

            self.all_xmin_xmax_V_dofs = dolfinx.fem.locate_dofs_topological(
                self.V1.sub(0), self.facet_dim, all_xmin_xmax_facets
            )

            self.all_ymin_ymax_V_dofs = dolfinx.fem.locate_dofs_topological(
                self.V1.sub(1), self.facet_dim, all_ymin_ymax_facets
            )

        # Interpolate the elasticity displacement (lives on the structure mesh)
        # field onto a function that lives on the fluid mesh
        use_built_in_interpolate = True

        if use_built_in_interpolate:
            self.fluid_mesh_displacement_bc_undeformed.interpolate(elasticity.u_delta)
            self.fluid_mesh_displacement_bc_undeformed.x.scatter_forward()
            self.fluid_mesh_displacement_bc.x.array[:] = (
                self.fluid_mesh_displacement_bc_undeformed.x.array[:]
            )
            self.fluid_mesh_displacement_bc.x.scatter_forward()

        else:
            fluid_mesh_displacement_bc_vec = self.custom_interpolate(elasticity)
            nn_bc_vec = np.shape(self.fluid_mesh_displacement_bc.vector.array[:])[0]
            self.fluid_mesh_displacement_bc.vector.array[:] = (
                fluid_mesh_displacement_bc_vec[:nn_bc_vec]
            )
            self.fluid_mesh_displacement_bc.x.scatter_forward()

            # print(self.fluid_mesh_displacement_bc.vector.array[self.all_interior_V_dofs])

        # Set the boundary condition for the walls of the computational domain
        zero_vec = dolfinx.fem.Constant(
            self.fluid.msh, PETSc.ScalarType((0.0, 0.0, 0.0))
        )

        zero_scalar = dolfinx.fem.Constant(self.fluid.msh, PETSc.ScalarType((0.0)))

        self.bcx = []

        self.bcx.append(
            dolfinx.fem.dirichletbc(
                self.fluid_mesh_displacement_bc, self.all_interior_V_dofs
            )
        )

        if params.domain.free_slip_along_walls:
            # print("uh_max", np.amax(elasticity.uh.x.array[:]))
            self.bcx.append(
                dolfinx.fem.dirichletbc(
                    zero_scalar, self.all_xmin_xmax_V_dofs, self.V1.sub(0)
                )
            )
            # print("uh_max", np.amax(elasticity.uh.x.array[:]))
            self.bcx.append(
                dolfinx.fem.dirichletbc(
                    zero_scalar, self.all_ymin_ymax_V_dofs, self.V1.sub(1)
                )
            )

        else:
            # print("uh_max", np.amax(elasticity.uh.x.array[:]))
            self.bcx.append(
                dolfinx.fem.dirichletbc(zero_vec, self.all_exterior_V_dofs, self.V1)
            )

        if self.first_move_mesh:
            u = ufl.TrialFunction(self.V1)
            v = ufl.TestFunction(self.V1)

            # TODO: use the distance in the diffusion calculation
            # self.a = dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
            self.a = dolfinx.fem.form(
                1.0
                / self.distance
                * ufl.inner(ufl.grad(u), ufl.grad(v))
                * ufl.dx
                # ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            )
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

        self.mesh_motion_solver.solve(self.b, self.fluid_mesh_displacement.vector)
        self.fluid_mesh_displacement.x.scatter_forward()

        # vals = self.fluid_mesh_displacement.vector.array.reshape(-1, 3)
        # nn = np.shape(vals)[0]

        # Obtain the vector of values for the mesh motion in a way that
        # keeps the ghost values (needed for the mesh update)
        with self.fluid_mesh_displacement.vector.localForm() as vals_local:
            vals = vals_local.array
            vals = vals.reshape(-1, 3)

        # Move the mesh by those values: new = original + displacement
        # self.fluid.msh.geometry.x[:, :] = self.fluid.msh.initial_position[:, :] + vals[:, :]
        self.fluid.msh.geometry.x[:, :] += vals[:, :]

        # # Obtain the vector of values for the mesh motion in a way that
        # # keeps the ghost values (needed for the mesh update)
        # with elasticity.u_delta.vector.localForm() as vals_local:
        #     vals = vals_local.array
        #     vals = vals.reshape(-1, 3)

        # # Move the mesh by those values: new = original + displacement
        # # self.structure.msh.geometry.x[:, :] = self.structure.msh.initial_position[:, :] + vals[:, :]
        # self.structure.msh.geometry.x[:, :] += vals[:, :]

        # self._force_interface_node_matching()

        # Save this mesh motion as the total mesh displacement
        self.total_mesh_displacement.vector.array[
            :
        ] += self.fluid_mesh_displacement.vector.array

        self.total_mesh_displacement.x.scatter_forward()

        self.first_move_mesh = False

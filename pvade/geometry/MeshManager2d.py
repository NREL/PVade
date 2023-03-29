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
import meshio

# from pvopt.geometry.panels.DomainCreation   import *
class FSIDomain:
    """This class creates the computational domain for 2D computational domains (2D panels, 2D cylinder)"""

    def __init__(self, params):
        """The class is initialised here

        Args:
            params (input parameters): input parameters available in the input file
        """

        # Get MPI communicators
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()

        # Store a full copy of params on this object
        self.params = params

        
        # problem = self.params.general.example
        
        # if problem == "panels2d":
        #     from pvade.geometry.panels2d.DomainCreation   import DomainCreation 
        # elif problem == "cylinder2d":
        #     from pvade.geometry.cylinder2d.DomainCreation   import DomainCreation
        

        # define markers for boundaries 
    
        self.x_min_marker = 1
        self.y_min_marker = 2
        self.x_max_marker = 4
        self.y_max_marker = 5
        self.internal_surface_marker = 7
        self.fluid_marker = 8


    def build(self):
        """This function call builds the geometry using Gmsh"""
        self.mesh_comm = MPI.COMM_WORLD
        self.model_rank = 0
        self.gdim = 3

        problem = self.params.general.example


        if problem == "panels2d":
            from pvade.geometry.panels2d.DomainCreation   import DomainCreation 
        elif problem == "cylinder2d":
            from pvade.geometry.cylinder2d.DomainCreation   import DomainCreation
            


        geometry = DomainCreation(self.params)

        # Only rank 0 builds the geometry and meshes the domain
        # if self.rank == 0:
        self.pv_model = geometry.build()

        self._mark_surfaces()
        self._set_length_scales_mod()  # TODO: this should probably be a method of the domain creation class

        if self.params.fluid.periodic:
            self._enforce_periodicity()

        self._generate_mesh()

        # All ranks receive their portion of the mesh from rank 0 (like an MPI scatter)
        self.msh, self.mt, self.ft = gmshio.model_to_mesh(self.pv_model, self.comm, 0)

        self.ndim = self.msh.topology.dim

        # Specify names for the mesh elements
        self.msh.name = self.params.general.example
        self.mt.name = f"{self.msh.name}_cells"
        self.ft.name = f"{self.msh.name}_facets"

            
    def read(self,path):
        """Read the mesh from external file located in output/mesh
        """
        if self.rank  == 0:
            print("Reading the mesh from file ...")
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, path+"/mesh.xdmf", "r") as xdmf:
            self.msh = xdmf.read_mesh(name="Grid")

        self.msh.topology.create_connectivity(self.msh.topology.dim-1, self.msh.topology.dim)
        with XDMFFile(MPI.COMM_WORLD,path+"/mesh_mf.xdmf",'r') as infile:

            self.ft = infile.read_meshtags(self.msh, "Grid")
        if self.rank == 0:
            print("Done.")

    def _mark_surfaces(self):
        """Creates boundary tags using gmsh"""
        # Loop through all surfaces to find periodic tags
        surf_ids = self.pv_model.occ.getEntities(1)

        self.dom_tags = {}

        for surf in surf_ids:
            tag = surf[1]

            com = self.pv_model.occ.getCenterOfMass(1, tag)

            if np.isclose(com[0], self.params.domain.x_min):
                self.dom_tags["left"] = [tag]

            elif np.allclose(com[0], self.params.domain.x_max):
                self.dom_tags["right"] = [tag]

            elif np.allclose(com[1], self.params.domain.y_min):
                self.dom_tags["bottom"] = [tag]

            elif np.allclose(com[1], self.params.domain.y_max):
                self.dom_tags["top"] = [tag]

            else:
                if "panel_surface" in self.dom_tags:
                    self.dom_tags["panel_surface"].append(tag)
                else:
                    self.dom_tags["panel_surface"] = [tag]

        self.pv_model.addPhysicalGroup(2, [1], 11)
        self.pv_model.setPhysicalName(2, 11, "fluid")

        self.pv_model.addPhysicalGroup(1, self.dom_tags["left"], self.x_min_marker)
        self.pv_model.setPhysicalName(1, self.x_min_marker, "left")
        self.pv_model.addPhysicalGroup(1, self.dom_tags["right"], self.x_max_marker)
        self.pv_model.setPhysicalName(1, self.x_max_marker, "right")
        self.pv_model.addPhysicalGroup(1, self.dom_tags["bottom"], self.y_min_marker)
        self.pv_model.setPhysicalName(1, self.y_min_marker, "bottom")
        self.pv_model.addPhysicalGroup(1, self.dom_tags["top"], self.y_max_marker)
        self.pv_model.setPhysicalName(1, self.y_max_marker, "top")

        self.pv_model.addPhysicalGroup(
            1, self.dom_tags["panel_surface"], self.internal_surface_marker
        )
        self.pv_model.setPhysicalName(1, self.internal_surface_marker, "panel_surface")

    def _set_length_scales_mod(self):
        res_min = self.params.domain.l_char
        if self.mesh_comm.rank == self.model_rank:
            # Define a distance field from the immersed panels
            self.pv_model.mesh.field.add("Distance", 1)
            self.pv_model.mesh.field.setNumbers(
                1, "FacesList", self.dom_tags["panel_surface"]
            )
            self.pv_model.mesh.field.add("Threshold", 2)
            self.pv_model.mesh.field.setNumber(2, "IField", 1)
            if "cylinder2d" in self.params.general.example:
                self.cyld_radius = self.params.domain.cyld_radius
                self.pv_model.mesh.field.setNumber(
                    2, "LcMin", self.params.domain.l_char
                )
                self.pv_model.mesh.field.setNumber(
                    2, "LcMax", 3.0 * self.params.domain.l_char
                )
                self.pv_model.mesh.field.setNumber(2, "DistMin", 2.0 * self.cyld_radius)
                self.pv_model.mesh.field.setNumber(
                    2, "DistMax", 10.0 * self.cyld_radius
                )

            else:
                self.pv_model.mesh.field.setNumber(
                    2, "LcMin", 0.2 * self.params.domain.l_char
                )
                self.pv_model.mesh.field.setNumber(
                    2, "LcMax", 3.0 * self.params.domain.l_char
                )
                self.pv_model.mesh.field.setNumber(
                    2, "DistMin", 0.5 * self.params.domain.l_char
                )
                self.pv_model.mesh.field.setNumber(
                    2, "DistMax", 1.0 * self.params.domain.l_char
                )

                # self.pv_model.mesh.field.setNumber(4, "LcMin", self.params.pv_array.panel_thickness)
                # self.pv_model.mesh.field.setNumber(4, "LcMax", 3.0 * self.params.pv_array.panel_thickness)
                # self.pv_model.mesh.field.setNumber(4, "DistMin", 2* self.params.pv_array.panel_length)
                # self.pv_model.mesh.field.setNumber(4, "DistMax", 10.* self.params.pv_array.panel_length)

            self.pv_model.mesh.field.add("Distance", 3)
            self.pv_model.mesh.field.setNumbers(3, "FacesList", self.dom_tags["top"])
            self.pv_model.mesh.field.setNumbers(3, "FacesList", self.dom_tags["bottom"])

            self.pv_model.mesh.field.add("Threshold", 4)
            self.pv_model.mesh.field.setNumber(4, "IField", 3)
            self.pv_model.mesh.field.setNumber(
                4, "LcMin", 5.0 * self.params.domain.l_char
            )
            self.pv_model.mesh.field.setNumber(
                4, "LcMax", 10.0 * self.params.domain.l_char
            )
            self.pv_model.mesh.field.setNumber(4, "DistMin", 0.1)
            self.pv_model.mesh.field.setNumber(4, "DistMax", 0.5)

            self.pv_model.mesh.field.add("Distance", 5)
            self.pv_model.mesh.field.setNumbers(5, "FacesList", self.dom_tags["left"])
            self.pv_model.mesh.field.setNumbers(5, "FacesList", self.dom_tags["right"])

            self.pv_model.mesh.field.add("Threshold", 6)
            self.pv_model.mesh.field.setNumber(6, "IField", 5)
            self.pv_model.mesh.field.setNumber(
                6, "LcMin", 2.0 * self.params.domain.l_char
            )
            self.pv_model.mesh.field.setNumber(
                6, "LcMax", 5.0 * self.params.domain.l_char
            )
            self.pv_model.mesh.field.setNumber(6, "DistMin", 0.1)
            self.pv_model.mesh.field.setNumber(6, "DistMax", 0.5)

            self.pv_model.mesh.field.add("Min", 7)
            self.pv_model.mesh.field.setNumbers(7, "FieldsList", [2, 4, 6])
            self.pv_model.mesh.field.setAsBackgroundMesh(7)

        if self.mesh_comm.rank == self.model_rank:
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.setOrder(2)
            gmsh.model.mesh.optimize("Netgen")

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

        self.pv_model.mesh.setPeriodic(
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

        self.pv_model.mesh.setPeriodic(
            2, self.dom_tags["right"], self.dom_tags["left"], left_right_translation
        )

    def _generate_mesh(self):
        if self.rank == 0:
            print("Starting mesh generation... ", end="")

            # Generate the mesh
            tic = time.time()

            #Mesh.Algorithm 2D mesh algorithm 
            # (1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay for Quads, 9: Packing of Parallelograms)
            # Default value: 6
            gmsh.option.setNumber("Mesh.Algorithm", 2)


            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            # gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
            self.pv_model.mesh.generate(3)
            self.pv_model.mesh.setOrder(2)
            self.pv_model.mesh.optimize("Netgen")
            self.pv_model.mesh.generate(3)
            toc = time.time()
            if self.rank == 0:
                print("Finished.")
                print(f"Total meshing time = {toc-tic:.1f} s")

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
                os.makedirs(self.params.general.output_dir_mesh)
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
            tetra_mesh = create_mesh(mesh_from_file, pts, "quad")
            tri_mesh = create_mesh(mesh_from_file, pts, "line")

            meshio.write(f"{self.params.general.output_dir_mesh}/mesh.xdmf", tetra_mesh)
            meshio.write(
                f"{self.params.general.output_dir_mesh}/mesh_mf.xdmf", tri_mesh
            )
            print("Done.")

    def test_mesh_functionspace(self):

        P2 = ufl.VectorElement("Lagrange", self.msh.ufl_cell(), 2)
        P1 = ufl.FiniteElement("Lagrange", self.msh.ufl_cell(), 1)
        V = FunctionSpace(self.msh, P2)
        Q = FunctionSpace(self.msh, P1)

        local_rangeV = V.dofmap.index_map.local_range
        dofsV = np.arange(*local_rangeV)

        local_rangeQ = Q.dofmap.index_map.local_range
        dofsQ = np.arange(*local_rangeQ)

        # coords = self.mesh.coordinates()

        nodes_dim = 0
        self.msh.topology.create_connectivity(nodes_dim, 0)
        num_nodes_owned_by_proc = self.msh.topology.index_map(nodes_dim).size_local
        geometry_entitites = cppmesh.entities_to_geometry(
            self.msh,
            nodes_dim,
            np.arange(num_nodes_owned_by_proc, dtype=np.int32),
            False,
        )
        points = self.msh.geometry.x

        coords = points[:]

        print(f"Rank {self.rank} owns {num_nodes_owned_by_proc} nodes\n{coords}")

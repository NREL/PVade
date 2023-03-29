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

# from pvade.geometry.panels.DomainCreation   import *
class FSIDomain:
    """This class creates the computational domain 
    """
    def __init__(self, params):
        """The class is initialised here

        Args:
            params (input parameters): input parameters available in the input file
        """

        # Get MPI communicators
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()

        # # Store a full copy of params on this object
        self.params = params

        # define markers for boundaries 
        
        self.x_min_marker = 1 
        self.y_min_marker = 2
        self.z_min_marker = 3
        self.x_max_marker = 4
        self.y_max_marker = 5
        self.z_max_marker = 6
        self.internal_surface_marker = 7
        self.fluid_marker = 8
        

    def build(self):
        """This function call builds the geometry using Gmsh
        """
        self.mesh_comm = MPI.COMM_WORLD
        self.model_rank = 0
        self.gdim = 3

 
        problem = self.params.general.example

        if problem == "panels":
            from pvade.geometry.panels.DomainCreation   import DomainCreation
        elif problem == "cylinder3d":
            from pvade.geometry.cylinder3d.DomainCreation   import DomainCreation
        
            

        geometry = DomainCreation(self.params)
        print(geometry)
        # Only rank 0 builds the geometry and meshes the domain
        # if self.rank == 0:
        self.pv_model = geometry.build()
        
        def print_model_info(model):
            print("Model Info:")
            for dim in range(3+1):
                ents = model.occ.getEntities(dim)
                print(f"| Dim {dim}: {ents}")
                
        print_model_info(self.pv_model)       
        
        self._mark_surfaces()
        self.pv_model = geometry._set_length_scales(self.pv_model,self.dom_tags) 

        # self._set_length_scales() # TODO: this should probably be a method of the domain creation class

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
        if self.rank  == 0:
            print("Done.")


    def _mark_surfaces(self):
        """Creates boundary tags using gmsh 
        """
        # Loop through all surfaces to find periodic tags
        surf_ids = self.pv_model.occ.getEntities(2)

        self.dom_tags = {}

        for surf in surf_ids:
            tag = surf[1]

            com = self.pv_model.occ.getCenterOfMass(2, tag)

            if np.isclose(com[0], self.params.domain.x_min):
                self.dom_tags["left"] = [tag]

            elif np.allclose(com[0], self.params.domain.x_max):
                self.dom_tags["right"] = [tag]

            elif np.allclose(com[1], self.params.domain.y_min):
                self.dom_tags["front"] = [tag]

            elif np.allclose(com[1], self.params.domain.y_max):
                self.dom_tags["back"] = [tag]

            elif np.allclose(com[2], self.params.domain.z_min):
                self.dom_tags["bottom"] = [tag]

            elif np.allclose(com[2], self.params.domain.z_max):
                self.dom_tags["top"] = [tag]

            else:
                if "panel_surface" in self.dom_tags:
                    self.dom_tags["panel_surface"].append(tag)
                else:
                    self.dom_tags["panel_surface"] = [tag]


        self.pv_model.addPhysicalGroup(3, [1], 11)
        self.pv_model.setPhysicalName(3, 11, "fluid")

        self.pv_model.addPhysicalGroup(2, self.dom_tags["left"], self.x_min_marker)
        self.pv_model.setPhysicalName(2, self.x_min_marker, "left")
        self.pv_model.addPhysicalGroup(2, self.dom_tags["right"], self.x_max_marker)
        self.pv_model.setPhysicalName(2, self.x_max_marker, "right")
        self.pv_model.addPhysicalGroup(2, self.dom_tags["front"], self.y_min_marker)
        self.pv_model.setPhysicalName(2, self.y_min_marker, "front")
        self.pv_model.addPhysicalGroup(2, self.dom_tags["back"], self.y_max_marker)
        self.pv_model.setPhysicalName(2, self.y_max_marker, "back")
        self.pv_model.addPhysicalGroup(2, self.dom_tags["bottom"], self.z_min_marker)
        self.pv_model.setPhysicalName(2, self.z_min_marker, "bottom")
        self.pv_model.addPhysicalGroup(2, self.dom_tags["top"], self.z_max_marker)
        self.pv_model.setPhysicalName(2, self.z_max_marker, "top")
        self.pv_model.addPhysicalGroup(2, self.dom_tags["panel_surface"], self.internal_surface_marker)
        self.pv_model.setPhysicalName(2, self.internal_surface_marker, "panel_surface")

    # def _set_length_scales(self):
    #     res_min = self.params.domain.l_char
    #     if self.mesh_comm.rank == self.model_rank:
    #         # Define a distance field from the immersed panels
    #         distance = self.pv_model.mesh.field.add("Distance", 1)
    #         self.pv_model.mesh.field.setNumbers(distance, "FacesList", self.dom_tags["panel_surface"])
            
    #         threshold = self.pv_model.mesh.field.add("Threshold")
    #         self.pv_model.mesh.field.setNumber(threshold, "IField", distance)


    #         factor = self.params.domain.l_char
    #         if 'cylinder3d' in self.params.general.example:
    #             self.cyld_radius = self.params.domain.cyld_radius
    #             resolution = factor * self.cyld_radius / 10
    #             self.pv_model.mesh.field.setNumber(threshold, "LcMin", resolution)
    #             self.pv_model.mesh.field.setNumber(threshold, "LcMax", 20 * resolution)
    #             self.pv_model.mesh.field.setNumber(threshold, "DistMin", .5 * self.cyld_radius)
    #             self.pv_model.mesh.field.setNumber(threshold, "DistMax", self.cyld_radius)

    #         else:
    #             resolution = factor * 10*self.params.pv_array.panel_thickness/2
    #             half_panel = self.params.pv_array.panel_length * np.cos(self.params.pv_array.tracker_angle)
    #             self.pv_model.mesh.field.setNumber(threshold, "LcMin", resolution*0.5)
    #             self.pv_model.mesh.field.setNumber(threshold, "LcMax", 5*resolution)
    #             self.pv_model.mesh.field.setNumber(threshold, "DistMin", self.params.pv_array.spacing[0])
    #             self.pv_model.mesh.field.setNumber(threshold, "DistMax", self.params.pv_array.spacing+half_panel)


    #         # Define a distance field from the immersed panels
    #         zmin_dist = self.pv_model.mesh.field.add("Distance")
    #         self.pv_model.mesh.field.setNumbers(zmin_dist, "FacesList", self.dom_tags["bottom"])

    #         zmin_thre = self.pv_model.mesh.field.add("Threshold")
    #         self.pv_model.mesh.field.setNumber(zmin_thre, "IField", zmin_dist)
    #         self.pv_model.mesh.field.setNumber(zmin_thre, "LcMin", 2*resolution)
    #         self.pv_model.mesh.field.setNumber(zmin_thre, "LcMax", 5*resolution)
    #         self.pv_model.mesh.field.setNumber(zmin_thre, "DistMin", 0.1)
    #         self.pv_model.mesh.field.setNumber(zmin_thre, "DistMax", 0.5)
            
    #         xy_dist = self.pv_model.mesh.field.add("Distance")
    #         self.pv_model.mesh.field.setNumbers(xy_dist, "FacesList", self.dom_tags["left"])
    #         self.pv_model.mesh.field.setNumbers(xy_dist, "FacesList", self.dom_tags["right"])
            
    #         xy_thre = self.pv_model.mesh.field.add("Threshold")
    #         self.pv_model.mesh.field.setNumber(xy_thre, "IField", xy_dist)
    #         self.pv_model.mesh.field.setNumber(xy_thre, "LcMin", 2 * resolution)
    #         self.pv_model.mesh.field.setNumber(xy_thre, "LcMax", 5* resolution)
    #         self.pv_model.mesh.field.setNumber(xy_thre, "DistMin", 0.1)
    #         self.pv_model.mesh.field.setNumber(xy_thre, "DistMax", 0.5)


    #         minimum = self.pv_model.mesh.field.add("Min")
    #         self.pv_model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, xy_thre, zmin_thre ])
    #         self.pv_model.mesh.field.setAsBackgroundMesh(minimum)

 

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
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            
            #3D mesh algorithm 
            # (1: Delaunay, 3: Initial mesh only, 4: Frontal, 7: MMG3D, 9: R-tree, 10: HXT)
            # Default value: 1
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)

            
            # Mesh recombination algorithm 
            # (0: simple, 1: blossom, 2: simple full-quad, 3: blos- som full-quad)
            # Default value: 1
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
            
            # Apply recombination algorithm to all surfaces, ignoring per-surface spec Default value: 0
            # gmsh.option.setNumber("Mesh.RecombineAll", 1)

            self.pv_model.mesh.generate(3)
            self.pv_model.mesh.setOrder(2)


            self.pv_model.mesh.optimize("Relocate3D")
            self.pv_model.mesh.generate(3)


            toc = time.time()
            if self.rank == 0:
                print("Finished.")
                print(f"Total meshing time = {toc-tic:.1f} s")


    def write_mesh_file(self):
        '''
        TODO: when saving a mesh file using only dolfinx functions
        it's possible certain elements of the data aren't preserved
        and that the mesh won't be able to be properly read in later
        on. MAKE SURE YOU CAN SAVE A MESH USING ONLY DOLFINX FUNCTIONS
        AND THEN READ IN THAT SAME MESH WITHOUT A LOSS OF CAPABILITY.
        '''

        if self.rank == 0:
            # Save the *.msh file and *.vtk file (the latter is for visualization only)
            print('Writing Mesh to %s... ' % (self.params.general.output_dir_mesh), end='')
            
            if os.path.exists(self.params.general.output_dir_mesh) == False: 
                os.makedirs(self.params.general.output_dir_mesh)
            gmsh.write('%s/mesh.msh' % (self.params.general.output_dir_mesh))
            gmsh.write('%s/mesh.vtk' % (self.params.general.output_dir_mesh))
            def create_mesh(mesh, clean_points, cell_type):
                cells = mesh.get_cells_type(cell_type)
                cell_data = mesh.get_cell_data("gmsh:physical", cell_type)

                out_mesh = meshio.Mesh(points=clean_points, cells={
                                    cell_type: cells}, cell_data={"name_to_read": [cell_data]})
                return out_mesh
                
            mesh_from_file = meshio.read(f'{self.params.general.output_dir_mesh}/mesh.msh')
            pts = mesh_from_file.points
            tetra_mesh = create_mesh(mesh_from_file, pts, "tetra")
            tri_mesh = create_mesh(mesh_from_file, pts, "triangle")

            meshio.write(f'{self.params.general.output_dir_mesh}/mesh.xdmf', tetra_mesh)
            meshio.write(f'{self.params.general.output_dir_mesh}/mesh_mf.xdmf', tri_mesh)
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

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
        self.comm = params.comm
        self.rank = params.rank
        self.num_procs = params.num_procs

        self.x_min_marker = 1
        self.x_max_marker = 2
        self.y_min_marker = 3
        self.y_max_marker = 4
        self.z_min_marker = 5
        self.z_max_marker = 6
        self.internal_surface_marker = 7
        self.fluid_marker = 8
        self.structure_marker = 8

    def build(self, params):
        """This function call builds the geometry using Gmsh"""

        problem = params.general.example

        if problem == "panels2d":
            from pvade.geometry.panels2d.DomainCreation   import DomainCreation 
        elif problem == "cylinder2d":
            from pvade.geometry.cylinder2d.DomainCreation   import DomainCreation
            
        self.geometry = DomainCreation(params)

        # Only rank 0 builds the geometry and meshes the domain
        if self.rank == 0:
            self.geometry.build()
            self.geometry.mark_surfaces()
            self.geometry.set_length_scales()

            if params.fluid.periodic:
                self._enforce_periodicity()

            self._generate_mesh()

        # All ranks receive their portion of the mesh from rank 0 (like an MPI scatter)
        self.msh, self.mt, self.ft = gmshio.model_to_mesh(self.geometry.gmsh_model, self.comm, 0)

        self.ndim = self.msh.topology.dim

        # Specify names for the mesh elements
        self.msh.name = params.general.example
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
        self.geometry.gmsh_model.mesh.generate(3)
        self.geometry.gmsh_model.mesh.setOrder(2)
        self.geometry.gmsh_model.mesh.optimize("Netgen")
        self.geometry.gmsh_model.mesh.generate(3)
        toc = time.time()
        if self.rank == 0:
            print("Finished.")
            print(f"Total meshing time = {toc-tic:.1f} s")


    def write_mesh_file(self, params):
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
                "Writing Mesh to %s... " % (params.general.output_dir_mesh), end=""
            )

            if os.path.exists(params.general.output_dir_mesh) == False:
                os.makedirs(params.general.output_dir_mesh)
            gmsh.write("%s/mesh.msh" % (params.general.output_dir_mesh))
            gmsh.write("%s/mesh.vtk" % (params.general.output_dir_mesh))

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
                f"{params.general.output_dir_mesh}/mesh.msh"
            )
            pts = mesh_from_file.points
            tetra_mesh = create_mesh(mesh_from_file, pts, "quad")
            tri_mesh = create_mesh(mesh_from_file, pts, "line")

            meshio.write(f"{params.general.output_dir_mesh}/mesh.xdmf", tetra_mesh)
            meshio.write(
                f"{params.general.output_dir_mesh}/mesh_mf.xdmf", tri_mesh
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

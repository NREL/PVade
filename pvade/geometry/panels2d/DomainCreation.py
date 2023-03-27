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


class DomainCreation:
    def __init__(self, params):
        """Initialize the DomainCreation object
         This initializes an object that creates the computational domain.

        Args:
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object
        """
        # Get MPI communicators
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()

        # Store a full copy of params on this object
        self.params = params

        self.x_min_marker = 1
        self.y_min_marker = 2
        self.z_min_marker = 3
        self.x_max_marker = 4
        self.y_max_marker = 5
        self.z_max_marker = 6
        self.internal_surface_marker = 7
        self.fluid_marker = 8

    def build(self):
        """This function creates the computational domain for a 2d simulation involving N panels.
            The panels are set at a distance apart, rotated at an angle theta and are elevated with a distance H from the ground.

        Returns:
            The function returns gmsh.model which contains the geometric description of the computational domain
        """
        self.mesh_comm = MPI.COMM_WORLD
        self.model_rank = 0
        self.gdim = 3

        # Initialize Gmsh options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        # All ranks create a Gmsh model object
        self.pv_model = gmsh.model()

        # Only rank 0 builds the geometry and meshes the domain
        # if self.rank == 0:

        self.pv_model.add("pv_domain")
        self.pv_model.setCurrent("pv_domain")

        # Compute and store some useful geometric quantities
        self.x_span = self.params.domain.x_max - self.params.domain.x_min
        self.y_span = self.params.domain.y_max - self.params.domain.y_min
        # self.z_span = self.params.domain.z_max - self.params.domain.z_min
        tracker_angle_rad = np.radians(self.params.pv_array.tracker_angle)

        # Create the fluid domain extent
        domain = self.pv_model.occ.addRectangle(
            self.params.domain.x_min,
            self.params.domain.y_min,
            0,  # self.params.domain.z_min,
            self.x_span,
            self.y_span
            # self.z_span,
        )

        for panel_id in range(self.params.pv_array.num_rows):
            panel_box = self.pv_model.occ.addRectangle(
                -0.5 * self.params.pv_array.panel_length,
                -0.5 * self.params.pv_array.panel_thickness,
                0,
                self.params.pv_array.panel_length,
                self.params.pv_array.panel_thickness,
            )

            # Rotate the panel currently centered at (0, 0, 0)
            self.pv_model.occ.rotate(
                [(2, panel_box)], 0, 0, 0, 0, 0, 1, tracker_angle_rad
            )

            # Translate the panel [panel_loc, 0, elev]
            self.pv_model.occ.translate(
                [(2, panel_box)],
                panel_id * self.params.pv_array.spacing[0],
                self.params.pv_array.elevation,
                0,
            )

            # Remove each panel from the overall domain
            self.pv_model.occ.cut([(2, domain)], [(2, panel_box)])

        self.pv_model.occ.synchronize()
        return self.pv_model

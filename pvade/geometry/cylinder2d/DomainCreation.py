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

# from pvopt.geometry.panels.domain_creation import *


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

    def build(self):
        """This function creates the computational domain for a flow around a 2D cylinder.

        Returns:
            The function returns gmsh.model which contains the geometric description of the computational domain
        """
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        # All ranks create a Gmsh model object
        self.pv_model = gmsh.model()
        c_x = c_y = 0.2
        r = self.params.domain.cyld_radius
        gdim = 2
        mesh_comm = MPI.COMM_WORLD
        model_rank = 0
        if mesh_comm.rank == model_rank:
            rectangle = self.pv_model.occ.addRectangle(
                self.params.domain.x_min,
                self.params.domain.y_min,
                0,
                self.params.domain.x_max,
                self.params.domain.y_max,
                tag=1,
            )
            obstacle = self.pv_model.occ.addDisk(c_x, c_y, 0, r, r)

        if mesh_comm.rank == model_rank:
            self.pv_model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
            self.pv_model.occ.synchronize()
        return self.pv_model

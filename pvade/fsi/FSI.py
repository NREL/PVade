# from dolfinx import *
"""Fluid-structure coupling utilities.

This module defines the :class:`FSI` class used to transfer fluid traction
loads to the structure solver.  The main coupling operation is implemented
in :meth:`FSI.fluid_struct`, where panel stress fields computed by the fluid
solver are relaxed and written into the structural solver state.
"""

# from dolfinx import *
import numpy as np
import time
import os
import shutil
from dolfinx.io import XDMFFile, VTKFile
from mpi4py import MPI
from pathlib import Path
import pytest
import dolfinx
from petsc4py import PETSc
import json

# from dolfinx.fem import create_nonmatching_meshes_interpolation_data

# hello


# test actions
class FSI:
    """Fluid-structure coupling manager.

    Stores MPI metadata and provides a method to transfer fluid stresses to
    the structural solver with optional under-relaxation.

    Attributes:
        comm (MPI Communicator): An MPI communicator used by all PVade objects
        ndim (int): The number of dimensions in the problem
        num_procs (int): The total number of processors being used to solve the problem
        rank (int): A unique ID for each process in the range `[0, 1, ..., num_procs-1]`

    """

    def __init__(self, domain, flow, structure, params):
        """Initialize the FSI coupling object.

        Stores MPI communicator metadata and the problem dimensionality used
        during coupling operations.

        Args:
            domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object
            flow (:obj:`pvade.fluid.FlowManager.Flow`): A Flow object
            structure (:obj:`pvade.structure.StructureMain.Structure`): A
                Structure object.
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object

        """

        self.comm = params.comm
        self.rank = params.rank
        self.num_procs = params.num_procs
        self.ndim = domain.fluid.msh.topology.dim

    def fluid_struct(self, domain, flow, structure, params):
        """Transfer fluid stress to the structure and apply relaxation.

        Performs one FSI coupling update:

        1. Copies the previous structural stress state to ``stress_old``.
        2. Interpolates the current fluid panel stress field onto the
           structure stress function.
        3. Applies linear under-relaxation using
           ``params.structure.beta_relaxation`` and the predicted stress.

        Args:
            domain (:obj:`pvade.geometry.MeshManager.FSIDomain`): Domain object
                containing fluid and structure meshes.
            flow (:obj:`pvade.fluid.FlowManager.Flow`): Flow object containing
                ``panel_stress_undeformed``.
            structure (:obj:`pvade.structure.StructureMain.Structure`):
                Structure object whose elasticity stress fields are updated.
            params (:obj:`pvade.IO.Parameters.SimParams`): Simulation parameters
                containing the stress relaxation factor.
        """
        # print("tst")

        structure.elasticity.stress_old.x.array[:] = structure.elasticity.stress.x.array
        structure.elasticity.stress_old.x.scatter_forward()

        structure.elasticity.stress.interpolate(flow.panel_stress_undeformed)
        structure.elasticity.stress.x.scatter_forward()

        beta = params.structure.beta_relaxation

        structure.elasticity.stress.x.array[:] = (
            beta * structure.elasticity.stress.x.array
            + (1.0 - beta) * structure.elasticity.stress_predicted.x.array
        )

        structure.elasticity.stress.x.scatter_forward()

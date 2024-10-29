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
    """Input/Output and file writing class

    This class manages the generation and updating of the saved xdmf files
    from each run along with command line outputs and log files.

    Attributes:
        comm (MPI Communicator): An MPI communicator used by all PVade objects
        ndim (int): The number of dimensions in the problem
        num_procs (int): The total number of processors being used to solve the problem
        rank (int): A unique ID for each process in the range `[0, 1, ..., num_procs-1]`
        results_filename_fluid (str): The full path to the directory in which all saved files will be written.

    """

    def __init__(self, domain, flow, structure, params):
        """Initialize the DataStream object

        This initializes an object that manages the I/O for all of PVade.

        Args:
            domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object
            flow (:obj:`pvade.fluid.FlowManager.Flow`): A Flow object
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object

        """

        self.comm = params.comm
        self.rank = params.rank
        self.num_procs = params.num_procs
        self.ndim = domain.fluid.msh.topology.dim

    def fluid_struct(self, domain, flow, structure, params):
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

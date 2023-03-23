# from dolfinx import *
import numpy as np
import time
import os
import shutil
from dolfinx.io import XDMFFile, VTKFile
from mpi4py import MPI
from pathlib import Path
import pytest


class DataStream:

    """Input/Output and file writing class

    This class manages the generation and updating of the saved xdmf files
    from each run along with command line outputs and log files.

    Attributes:
        comm (MPI Communicator): An MPI communicator used by all PVade objects
        ndim (int): The number of dimensions in the problem
        num_procs (int): The total number of processors being used to solve the problem
        rank (int): A unique ID for each process in the range `[0, 1, ..., num_procs-1]`
        results_filename (str): The full path to the directory in which all saved files will be written.

    """

    def __init__(self, domain, flow, params):
        """Initialize the DataStream object

        This initializes an object that manages the I/O for all of PVade.

        Args:
            domain (:obj:`pvade.geometry.MeshManager.Domain`): A Domain object
            flow (:obj:`pvade.FlowManager.Flow`): A Flow object
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object

        """
        self.comm = params.comm
        self.rank = params.rank
        self.num_procs = params.num_procs

        self.ndim = domain.msh.topology.dim

        self.results_filename = f"{params.general.output_dir_sol}/solution.xdmf"

        with XDMFFile(self.comm, self.results_filename, "w") as xdmf_file:
            tt = 0.0
            xdmf_file.write_mesh(domain.msh)
            xdmf_file.write_function(flow.u_k, 0.0)
            xdmf_file.write_function(flow.p_k, 0.0)

    def save_XDMF_files(self, flow, tt):
        """Write additional timestep to XDMF file

        This function appends the state of the flow at time `tt` to an existing XDMF file.

        Args:
            flow (:obj:`pvade.FlowManager.Flow`): A Flow object
            tt (float): The time at which the current solution exists

        """
        with XDMFFile(self.comm, self.results_filename, "a") as xdmf_file:
            xdmf_file.write_function(flow.u_k, tt)
            xdmf_file.write_function(flow.p_k, tt)

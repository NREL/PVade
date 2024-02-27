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
class DataStream:
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

    def __init__(self, domain, flow, elasticity, params):
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

        self.log_filename = f"{params.general.output_dir_sol}/log.txt"
        if self.rank == 0:
            with open(self.log_filename, "w") as fp:
                fp.write("Run Started.\n")

        # If doing a fluid simulation, start a fluid solution file
        if params.general.fluid_analysis == True:
            self.results_filename_fluid = (
                f"{params.general.output_dir_sol}/solution_fluid.xdmf"
            )

            with XDMFFile(self.comm, self.results_filename_fluid, "w") as xdmf_file:
                tt = 0.0
                xdmf_file.write_mesh(domain.fluid.msh)
                xdmf_file.write_function(flow.u_k, 0.0)
                xdmf_file.write_function(flow.p_k, 0.0)
                xdmf_file.write_function(flow.panel_stress, 0.0)
                xdmf_file.write_function(domain.total_mesh_displacement, 0.0)

        # If doing a structure simulation, start a structure solution file
        if params.general.structural_analysis == True:
            self.results_filename_structure = (
                f"{params.general.output_dir_sol}/solution_structure.xdmf"
            )

            with XDMFFile(self.comm, self.results_filename_structure, "w") as xdmf_file:
                tt = 0.0
                xdmf_file.write_mesh(domain.structure.msh)
                xdmf_file.write_function(elasticity.u, 0.0)
                xdmf_file.write_function(elasticity.stress, 0.0)
                xdmf_file.write_function(elasticity.v_old, 0.0)
                xdmf_file.write_function(elasticity.sigma_vm_h, 0.0)

        if self.comm.rank == 0 and self.comm.size > 1 and params.general.test == True:
            self.log_filename_structure = f"{params.general.output_dir_sol}/log_str.txt"

            with open(self.log_filename_structure, "w") as fp:
                fp.write(f"start , size of vec is {vec_check.size}\n")
                fp.close()

            for n in range(solution_vec.size):
                with open(self.log_filename_structure, "a") as fp:
                    fp.write(
                        f"row {n}, {vec_check[n]} compared to  {solution_vec[n]}, diff is {abs(vec_check[n] - solution_vec[n])}\n"
                    )
                    if abs(vec_check[n] - solution_vec[n]) > 1e-6:
                        fp.write("flag \n")
                        fp.close()
                        print(
                            f"solution vec incorrect in {n}th entry \n {vec_check[n]} compared to  {solution_vec[n]} np = {self.comm.size} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        )
                        exit()
            print(f"all values match with np = {self.comm.size}")

        # def store_vec_rank1(functionspace, vector):
        #     imap = vector.function_space.dofmap.index_map
        #     local_range = (
        #         np.asarray(imap.local_range, dtype=np.int32)
        #         * vector.function_space.dofmap.index_map_bs
        #     )
        #     size_global = imap.size_global * vector.function_space.dofmap.index_map_bs

        #     # Communicate ranges and local data
        #     ranges = MPI.COMM_WORLD.gather(local_range, root=0)
        #     data = MPI.COMM_WORLD.gather(vector.vector.array, root=0)
        #     # print(data)
        #     # Communicate local dof coordinates
        #     x = functionspace.tabulate_dof_coordinates()[: imap.size_local]
        #     x_glob = MPI.COMM_WORLD.gather(x, root=0)

        #     # Declare gathered parallel arrays
        #     global_array = np.zeros(size_global)
        #     global_x = np.zeros((size_global, 3))
        #     u_from_global = np.zeros(global_array.shape)

        #     x0 = functionspace.tabulate_dof_coordinates()

        #     if self.comm.rank == 0:
        #         # Create array with all values (received)
        #         for r, d in zip(ranges, data):
        #             global_array[r[0] : r[1]] = d

        #     return global_array

        # exit()

    def save_XDMF_files(self, fsi_object, domain, tt):
        """Write additional timestep to XDMF file

        This function appends the state of the flow at time `tt` to an existing XDMF file.

        Args:
            flow (:obj:`pvade.fluid.FlowManager.Flow`): A Flow object
            tt (float): The time at which the current solution exists

        """

        if fsi_object.name == "fluid":
            with XDMFFile(self.comm, self.results_filename_fluid, "a") as xdmf_file:
                xdmf_file.write_function(fsi_object.u_k, tt)
                xdmf_file.write_function(fsi_object.p_k, tt)
                xdmf_file.write_function(fsi_object.panel_stress, tt)
                xdmf_file.write_function(domain.total_mesh_displacement, tt)

        elif fsi_object.name == "structure":
            with XDMFFile(self.comm, self.results_filename_structure, "a") as xdmf_file:
                xdmf_file.write_function(fsi_object.u, tt)
                xdmf_file.write_function(fsi_object.stress, tt)
                xdmf_file.write_function(fsi_object.v_old, tt)
                xdmf_file.write_function(fsi_object.sigma_vm_h, tt)

        else:
            raise ValueError(
                f"Got found fsi object name = {fsi_object.name}, not recognized."
            )

    def print_and_log(self, string_to_print):
        if self.rank == 0:
            print(string_to_print)

            with open(self.log_filename, "a") as fp:
                fp.write(f"{string_to_print}\n")

    def fluid_struct(self, domain, flow, elasticity, params):
        # print("tst")

        elasticity.stress_old.x.array[:] = elasticity.stress.x.array
        elasticity.stress_old.x.scatter_forward()

        elasticity.stress.interpolate(flow.panel_stress_undeformed)
        elasticity.stress.x.scatter_forward()

        beta = params.structure.beta_relaxation

        elasticity.stress.x.array[:] = (
            beta * elasticity.stress.x.array
            + (1.0 - beta) * elasticity.stress_predicted.x.array
        )

        elasticity.stress.x.scatter_forward()

from dolfinx import *
import numpy as np
import time
import os
import shutil
from dolfinx.io import XDMFFile, VTKFile
from mpi4py import MPI
from pathlib import Path
import pytest


class DataStream:
    def __init__(self, domain, flow, params):

        self.comm = params.comm
        self.rank = params.rank
        self.num_procs = params.num_procs

        self.ndim = domain.msh.topology.dim

        self.results_filename = f"{params.general.output_dir_sol}/solution.xdmf"

        # if self.num_procs > 1:
        #     encoding = [XDMFFile.Encoding.HDF5]
        # else:
        #     encoding = [XDMFFile.Encoding.HDF5, XDMFFile.Encoding.ASCII]

        # # @pytest.mark.parametrize("encoding", encodings)

        with XDMFFile(self.comm, self.results_filename, "w") as xdmf_file:
            tt = 0.0
            xdmf_file.write_mesh(domain.mesh)
            xdmf_file.write_function(domain.total_mesh_displacement, tt)
            xdmf_file.write_function(flow.u_k, 0.0)
            xdmf_file.write_function(flow.p_k, 0.0)


    def save_XDMF_files(self, flow, tt):
        with XDMFFile(self.comm, self.results_filename, "a") as xdmf_file:
            xdmf_file.write_function(flow.u_k, tt)
            xdmf_file.write_function(flow.p_k, tt)

        self._write_single_time_xdmf(domain, flow, tt)

    def save_XDMF_files(self, domain, flow, tt):

        with XDMFFile(self.comm, self.results_filename, "a") as xdmf_file:
            xdmf_file.write_function(domain.total_mesh_displacement, tt)
            xdmf_file.write_function(flow.u_k, tt)
            xdmf_file.write_function(flow.p_k, tt)

        self._write_single_time_xdmf(domain, flow, tt)

    def _write_single_time_xdmf(self, domain, flow, tt):
        new_filename = self.results_filename.split(".xdmf")[0] + f"{int(tt*1000):06.0f}.xdmf"

        with XDMFFile(self.comm, new_filename, "w") as xdmf_file:
            xdmf_file.write_mesh(domain.mesh)
            xdmf_file.write_function(domain.total_mesh_displacement, tt)
            xdmf_file.write_function(flow.u_k, tt)
            xdmf_file.write_function(flow.p_k, tt)


    # def save_XDMF_files(self, flow, params, step, xdmf_file):
    #     current_time = step * params["dt"]

    #     xdmf_file.write(flow.u_k, current_time)
    #     xdmf_file.write(flow.p_k, current_time)

    #     try:
    #         nu_T_fn = project(flow.nu_T, flow.Q, solver_type="cg")
    #     except:
    #         print("Could not project nu_T form")
    #     else:
    #         nu_T_fn.rename("nu_T", "nu_T")
    #         xdmf_file.write(nu_T_fn, current_time)

    # def finalize(self, mpi_info):
    #     if mpi_info["rank"] == 0:
    #         print("Solver Finished.")
    #         self.toc = time.time()
    #         self.fp_log = open(self.fp_log_name, "a")
    #         self.fp_log.write("\nTotal Solver Time = %f (s)\n" % (self.toc - self.tic))
    #         self.fp_log.close()

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

#hello

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
        results_filename (str): The full path to the directory in which all saved files will be written.

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


        # fluid vars 
        self.ndim = domain.fluid.msh.topology.dim

        self.results_filename = f"{params.general.output_dir_sol}/solution.xdmf"
        self.log_filename = f"{params.general.output_dir_sol}/log.txt"

        with XDMFFile(self.comm, self.results_filename, "w") as xdmf_file:
            tt = 0.0
            xdmf_file.write_mesh(domain.fluid.msh)
            xdmf_file.write_function(flow.u_k, 0.0)
            xdmf_file.write_function(flow.p_k, 0.0)
            xdmf_file.write_function(flow.inflow_profile, 0.0)

        if self.rank == 0:
            with open(self.log_filename, "w") as fp:
                fp.write("Run Started.\n")

        # array = []
        # array.append(domain.fluid.facet_tags.find(domain.domain_markers["bottom_0"]["idx"]))
        # array.append(domain.structure.facet_tags.find(domain.domain_markers["top_0"]["idx"])[:])
        # array.append(domain.structure.facet_tags.find(domain.domain_markers["left_0"]["idx"])[:])
        # array.append(domain.structure.facet_tags.find(domain.domain_markers["right_0"]["idx"])[:])
        # array.append(domain.structure.facet_tags.find(domain.domain_markers["bottom_0"]["idx"])[:])


        # tree = dolfinx.geometry.BoundingBoxTree(domain.fluid.msh, 3)
        # num_entities_local = domain.fluid.msh.topology.index_map(3).size_local + domain.fluid.msh.topology.index_map(3).num_ghosts
        # entities = np.arange(num_entities_local, dtype=np.int32)
        # midpoint_tree = dolfinx.geometry.create_midpoint_tree(domain.fluid.msh, 3, entities)

        # def f_mesh_expr(x):
        #     cells = dolfinx.geometry.compute_closest_entity(tree, midpoint_tree, domain.fluid.msh, x.T)
        #     return flow.u_k.eval(x.T, cells).T
        # cells =  domain.fluid.cell_tags.find(domain.domain_markers["structure"]["idx"])
        # cells_num =  domain.fluid.cell_tags.find(1)
        # submesh_entities = dolfinx.mesh.locate_entities(domain.fluid.msh, dim=3, marker=1)

        # elasticity.uh.interpolate(f_mesh_expr, cells=submesh_entities)
        # flow.u_k.vector.ghostUpdate()

        def store_vec_rank1(functionspace, vector):
            imap = vector.function_space.dofmap.index_map
            local_range = np.asarray(imap.local_range, dtype=np.int32) * vector.function_space.dofmap.index_map_bs
            size_global = imap.size_global * vector.function_space.dofmap.index_map_bs

            # Communicate ranges and local data
            ranges = MPI.COMM_WORLD.gather(local_range, root=0)
            data = MPI.COMM_WORLD.gather(vector.vector.array, root=0)
            # print(data)
            # Communicate local dof coordinates
            x = functionspace.tabulate_dof_coordinates()[:imap.size_local]
            x_glob = MPI.COMM_WORLD.gather(x, root=0)

            # Declare gathered parallel arrays
            global_array = np.zeros(size_global)
            global_x = np.zeros((size_global, 3))
            u_from_global = np.zeros(global_array.shape)

            x0 = functionspace.tabulate_dof_coordinates()

            if self.comm.rank == 0:
                # Create array with all values (received)
                for r, d in zip(ranges, data):
                    global_array[r[0]:r[1]] = d

            return global_array  
        
        def mpi_print(s):
            print(f"Rank {self.rank}:\n {s} ")




        elasticity.uh.interpolate(flow.u_k)#, nmm_interpolation_data=create_nonmatching_meshes_interpolation_data(
        # elasticity.uh.function_space._cpp_object,
        # flow.u_k.function_space._cpp_object))
        elasticity.uh.x.scatter_forward()

        # elasticity.uh.interpolate(flow.u_k)

        elasticity.uh_exp.interpolate(flow.inflow_profile)


        mpi_print(f"size of uh {elasticity.uh.vector.size},size of uh_Exp {elasticity.uh_exp.vector.size}")

        
        # for m in range(flow.u_k.vector.size):
        #     if (store_vec_rank1(flow.V,flow.u_k)[m] - store_vec_rank1(flow.V,flow.inflow_profile)[m]) > 1.e-6:
        #         print("not the same vectors (fluid)" )

        
        for m in range(elasticity.uh.vector.size):
            if (store_vec_rank1(elasticity.V,elasticity.uh)[m] - store_vec_rank1(elasticity.V,elasticity.uh_exp)[m]) > 1.e-6:
                print("not the same vectors (struct)" )

        

        # with elasticity.uh.vector as v_local:
        #     # mpi_print(v_local.array)
        #     print("local size on rank", self.rank, ": ",  elasticity.uh.vector.local_size,\
        #           elasticity.uh.vector.array.size,elasticity.uh.vector.array_w.size,elasticity.uh.vector.array_r.size)
        #     # mpi_print(elasticity.uh.x.array[:])
     

        # elasticity.uh.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        # elasticity.uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.REVERSE)
        # elasticity.uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # 
        # elasticity.uh.x.scatter_forward()  
        #    
        # Get local ranges and global size of array
        # imap = elasticity.uh.function_space.dofmap.index_map
        # local_range = np.asarray(imap.local_range, dtype=np.int32) * \
        #     elasticity.uh.function_space.dofmap.index_map_bs
        # size_global = imap.size_global * elasticity.uh.function_space.dofmap.index_map_bs
        
   

        solution_vec = np.sort(np.array(store_vec_rank1(elasticity.V,elasticity.uh)))
        solution_vec = solution_vec.reshape((-1, 1))
        # print(f"sie of vec: {solution_vec.size}, Global array: \n{solution_vec}\n")    

        if self.comm.size == 1:
            np.savetxt('solution_1rank_output.txt', solution_vec, delimiter=',')
            # with open('solution_1rank_output.txt', 'w') as filehandle:
            #     json.dump(solution_vec.toList(), filehandle)
        else:
            vec_check = np.genfromtxt('solution_1rank_output.txt', delimiter=",")
            # my_file = open('solution_1rank_output.txt', 'r')
            # vec_check = my_file.read()


        # 1 rank mpi on terminal 
        if False:#self.comm.size == 1:
            vec_check = np.array([\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [0.        ],\
            [5.90707241],\
            [5.90707241],\
            [5.90707241],\
            [5.90707241],\
            [5.90707241],\
            [5.90707241],\
            [5.90707241],\
            [6.3707722 ],\
            [6.3707722 ],\
            [6.3707722 ],\
            [6.3707722 ],\
            [6.3707722 ],\
            [6.3707722 ],\
            [6.3707722 ],\
            [7.07294565],\
            [7.64901147],\
            [7.84437314],\
            [7.88178718],\
            [7.88178718],\
            [7.88178718],\
            [7.88797221],\
            [7.90053877],\
            [7.90053877],\
            [7.91868428],\
            [8.07891265],\
            [8.07891265],\
            [8.09588944],\
            [8.10751676],\
            [8.11320099],\
            [8.11320099],\
            [8.11320099],\
            [8.50174927],\
            [8.68683861],\
            [9.00455634],\
            [9.00455634],\
            [9.00455634],\
            [9.00455634],\
            [9.00455634],\
            [9.00455634],\
            [9.00455634],\
            [9.15877859],\
            [9.15877859],\
            [9.15877859],\
            [9.15877859],\
            [9.15877859],\
            [9.15877859],\
            [9.15877859]])
        #     # Create array with all coordinates
        #     for r, x_ in zip(ranges, x_glob):
        #         global_x[r[0]:r[1], :] = x_
        #     serial_to_global = []
        #     for coord in x0:
        #         serial_to_global.append(np.abs(global_x-coord).sum(axis=1).argmin())
        #     # Create sorted array from
        #     for serial, glob in enumerate(serial_to_global):
        #         u_from_global[serial] = global_array[glob]
        #     print(f"Gathered from parallel: \n{u_from_global}\n")

        # # Calculated process
        # u_cal = np.multiply(3, u_from_global)
        # print(f"Calculated global array: \n{u_cal}\n")
        

        # print(vec_check)
        
                        
                # assert (abs(vec_check[n] - solution_vec[n]) < 1e-6), f"solution vec incorrect in {n}th entry \n {vec_check[n]} compared to  {solution_vec[n]}"
        


        

       
        # with elasticity.uh.vector.localForm() as v_local:
        #     # mpi_print(elasticity.uh.vector.array)
        #     print("Global size on rank", self.rank, ": ",  elasticity.uh.vector.size)
        #     with open(self.log_filename_str, "w") as fp:
        #             fp.write(f"Global array: \n{solution_vec}\n")
        #             fp.close()
        #     mpi_print(elasticity.uh.x.array[:])     
        # elasticity.uh.x.scatter_forward()
        # flow.u_k.x.array[:] - elasticity.uh.x.array[:]
        # print((flow.u_k.x.array[:] == elasticity.uh.x.array[:]).all())

        # mesh test
        

        # mpi_print(f"Number of local cells: {domain.structure.msh.topology.index_map(3).size_local}")
        # mpi_print(f"Number of global cells: {domain.structure.msh.topology.index_map(3).size_global}")
        # mpi_print(f"Number of local vertices: {domain.structure.msh.topology.index_map(0).size_local}")
        # mpi_print("Cell (dim = 3) to vertex (dim = 0) connectivity")
        # mpi_print(domain.structure.msh.topology.connectivity(3, 0))

        # if self.comm.size > 1:
        #     mpi_print(f"Ghost cells (global numbering): {domain.structure.msh.topology.index_map(3).ghosts}")
        #     mpi_print(f"Ghost owner rank: {domain.structure.msh.topology.index_map(3).ghost_owner_rank()}")
        # structure 
        if elasticity.structural_analysis == True:
            self.ndim_str = domain.structure.msh.topology.dim

            self.results_filename_str = f"{params.general.output_dir_sol}/solution_str.xdmf"
            

            with XDMFFile(self.comm, self.results_filename_str, "w") as xdmf_file:
                tt = 0.0
                xdmf_file.write_mesh(domain.structure.msh)
                xdmf_file.write_function(elasticity.uh, 0.0)
                xdmf_file.write_function(elasticity.sigma_vm_h, 0.0)
            self.results_filename_vtk = f"{params.general.output_dir_sol}/solution_vtk.pvd"
            with VTKFile(self.comm, self.results_filename_vtk, "w") as file:
                file.write_mesh(domain.structure.msh)
                file.write_function(elasticity.uh, 0.0)
            # if self.rank == 0:
                # with open(self.log_filename_str, "w") as fp:
                    # fp.write("Run Started.\n")
        if self.comm.rank == 0 and self.comm.size > 1:
            self.log_filename_str = f"{params.general.output_dir_sol}/log_str.txt"
        
            with open(self.log_filename_str, "w") as fp:
                fp.write(f"start , size of vec is {vec_check.size}\n")
                fp.close()

            for n in range(solution_vec.size):
                with open(self.log_filename_str, "a") as fp:
                        fp.write(f"row {n}, {vec_check[n]} compared to  {solution_vec[n]}, diff is {abs(vec_check[n] - solution_vec[n])}\n")
                        if abs(vec_check[n] - solution_vec[n]) > 1e-6 :
                            fp.write("flag \n")
                            fp.close()
                            print(f"solution vec incorrect in {n}th entry \n {vec_check[n]} compared to  {solution_vec[n]} np = {self.comm.size} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            exit()
            print(f"all values match with np = {self.comm.size}")
        
        exit()
    def save_XDMF_files(self, flow, tt):
        """Write additional timestep to XDMF file

        This function appends the state of the flow at time `tt` to an existing XDMF file.

        Args:
            flow (:obj:`pvade.fluid.FlowManager.Flow`): A Flow object
            tt (float): The time at which the current solution exists

        """
        with XDMFFile(self.comm, self.results_filename, "a") as xdmf_file:
            xdmf_file.write_function(flow.u_k, tt)
            xdmf_file.write_function(flow.p_k, tt)

    def save_XDMF_files_str(self, elasticity, tt):
        """Write additional timestep to XDMF file

        This function appends the state of the flow at time `tt` to an existing XDMF file.

        Args:
            flow (:obj:`pvade.fluid.FlowManager.Flow`): A Flow object
            tt (float): The time at which the current solution exists

        """
        with XDMFFile(self.comm, self.results_filename_str, "a") as xdmf_file:
            xdmf_file.write_function(elasticity.uh, tt)
            xdmf_file.write_function(elasticity.sigma_vm_h, tt)
        

    def print_and_log(self, string_to_print):
        if self.rank == 0:
            print(string_to_print)

            with open(self.log_filename, "a") as fp:
                fp.write(f"{string_to_print}\n")

    # def fluid_struct(self, domain, flow, elasticity, params):
        
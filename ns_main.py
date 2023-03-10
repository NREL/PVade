from pvade.FlowManager import Flow
from pvade.DataStream import DataStream
from pvade.Parameters import SimParams

import cProfile
import sys
import numpy as np
from mpi4py import MPI
import time
from dolfinx.common import TimingType, list_timings
import tqdm.autonotebook


def main():
    # problems solver are:
    # cylinder3d

    # cylinder2d
    # panels
    # problem = "cylinder2d"

    # problem = "panels"
    problem = sys.argv[2]

    # use correct inpit file
    if problem == "cylinder3d":
        params = SimParams("inputs/3d_cyld.yaml")
        dim = 3
    elif problem == "cylinder2d":
        params = SimParams("inputs/2d_cyld.yaml")
        dim = 2
    elif problem == "panels":
        params = SimParams("inputs/sim_params_alt.yaml")
        dim = 3
    elif problem == "panels2d":
        params = SimParams("inputs/sim_params_alt_2D.yaml")
        dim = 2
    else:
        raise ValueError(f"Problem is not defined, please specify problem type using --problem $problem \n$problem: [cylinder3d,cylinder2d,panels,panels2d] ")
    
    if MPI.COMM_WORLD.rank == 0:
        print("Currently Solving:",problem)
        print("command used is: ",str(sys.argv))

    if dim == 3:
        from pvade.geometry.MeshManager3d import FSIDomain
    elif dim == 2:
        from pvade.geometry.MeshManager2d import FSIDomain
    else:
        print("dimension not defined")
        exit()



    # Initialize the domain and construct the initial mesh
    domain = FSIDomain(params)
    if params.general.create_mesh == True: 
        domain.build()
        domain.write_mesh_file()
    elif params.general.read_mesh == True:
        domain.read()
    else:
        raise ValueError(f"Error in creating/loading mesh, please correct inputs")

    # domain.test_mesh_functionspace()

    if params.general.mesh_only == True:
        # dolfinx.cpp.common.TimingType()
        list_timings(MPI.COMM_WORLD, [TimingType.wall])
        exit()

    # Check to ensure mesh node matching for periodic simulations
    # if domain.periodic_simulation:
    # domain.check_mesh_periodicity(mpi_info)

    # Initialize the function spaces for the flow
    flow = Flow(domain)

    # # # Specify the boundary conditions
    flow.build_boundary_conditions(domain, params)

    # # # Build the fluid forms
    flow.build_forms(domain, params)

    dataIO = DataStream(domain, flow, params)


    # with XDMFFile(domain.comm, results_filename, "w") as xdmf:
    #     xdmf.write_mesh(domain.msh)
    #     xdmf.write_function(flow.u_k, 0.0)
    #     xdmf.write_function(flow.p_k, 0.0)
    # # print(np.max(flow.u_k.x.array))
    t_steps = int(params.solver.t_final / params.solver.dt)
    # save_vtk_every_n = int(params.solver.save_vtk_every / params.solver.dt)
    tic = time.time()
    if domain.rank == 0:
        progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=params.solver.t_steps)
    for k in range(params.solver.t_steps):
        if domain.rank == 0:
            progress.update(1)

        # Solve the fluid problem at each timestep
        flow.solve(params)

        # adjust pressure to avoid dissipation of pressure profile
        # flow.adjust_dpdx_for_constant_flux(mpi_info)
        if (k + 1) % params.solver.save_xdmf_interval_n == 0:
            if domain.rank == 0:
                print(
                    f"Time {params.solver.dt*(k+1):.2f} of {params.solver.t_final:.2f}, CFL = {flow.cfl_max}"
                )

            dataIO.save_XDMF_files(flow, (k + 1) * params.solver.dt)
    toc = time.time()
    if domain.rank == 0:
        print(f"Total solve time = {toc-tic:.2f} s.")
        print(f"Average time per iteration = {(toc-tic)/t_steps:.2f} s.")
    list_timings(MPI.COMM_WORLD, [TimingType.wall])

def write_metrics():
    if MPI.COMM_WORLD.Get_rank() == 0:
        with open('profiling.txt', 'r') as output_file:
            #solver_line = [line for line in output_file if "(solve)" in line]
            solver_line = [line for line in output_file if "(solve)" in line]
            print(solver_line)
            solver_line = solver_line[0].split()
            print("solve: total time = ",solver_line[3], " time per call  = ",solver_line[4], " num calls = ",solver_line[0])

        with open('profiling.txt', 'r') as output_file_1:
            #solver_line = [line for line in output_file if "(solve)" in line]
            solver1_line = [line for line in output_file_1 if "_solver_step_1" in line]
            print(solver1_line)
            solver1_line = solver1_line[0].split()
            print("solver 1: total time = ",solver1_line[3], " time per call  = ",solver1_line[4], " num calls = ",solver1_line[0])

        with open('profiling.txt', 'r') as output_file:
            solver2_line = [line for line in output_file if "_solver_step_2" in line]
            print(solver2_line)
            solver2_line = solver2_line[0].split()
            print("solver 2: total time = ",solver2_line[3], " time per call  = ",solver2_line[4], " num calls = ",solver2_line[0])

        with open('profiling.txt', 'r') as output_file:
            solver3_line = [line for line in output_file if "_solver_step_3" in line]
            print(solver3_line)
            solver3_line = solver3_line[0].split()
            print("solver 3: total time = ",solver3_line[3], " time per call  = ",solver3_line[4], " num calls = ",solver3_line[0])

        with open('profiling.txt', 'r') as output_file:
            meshread_line = [line for line in output_file if "(read)" in line]
            if meshread_line:
                meshread_line =meshread_line[0].split()
                print("mesh read: total time = ",meshread_line[3], " time per call  = ",meshread_line[4], " num calls = ",meshread_line[0])

        with open('profiling.txt', 'r') as output_file:
            meshbuild_line = [line for line in output_file if "(build)" in line]
            if meshbuild_line:
                meshbuild_line = meshbuild_line[0].split()
                print("mesh build: total time = ",meshbuild_line[3], " time per call  = ",meshbuild_line[4], " num calls = ",meshbuild_line[0])
# Print profiling results
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("profiling.prof")

    with open("profiling.txt", "w") as output_file:
        sys.stdout = output_file
        profiler.print_stats(sort="cumtime")
        sys.stdout = sys.__stdout__


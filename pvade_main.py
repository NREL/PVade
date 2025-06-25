from pvade.fluid.FlowManager import Flow
from pvade.IO.DataStream import DataStream, start_print_and_log
from pvade.fsi.FSI import FSI
from pvade.IO.Parameters import SimParams
from pvade.IO.Utilities import get_input_file, write_metrics
from pvade.geometry.MeshManager import FSIDomain

from dolfinx.common import TimingType, list_timings
import cProfile
import sys
import tqdm.autonotebook
import numpy as np


from pvade.structure.StructureMain import Structure
import os
from mpi4py import MPI

def main(input_file=None):
    # Get the path to the input file from the command line
    if input_file is None:
        input_file = get_input_file()
    # input_file = "inputs/sim_params_alt.yaml"  # get_input_file()

    # Load the parameters object specified by the input file
    params = SimParams(input_file)

    # Save the parameters object to a logfile
    logfile_name = os.path.join(params.general.output_dir, "logfile.log")
    start_print_and_log(params.rank, logfile_name)


    fluid_analysis = params.general.fluid_analysis
    structural_analysis = params.general.structural_analysis
    thermal_analysis = params.general.thermal_analysis
    
    # Initialize the domain and construct the initial mesh
    domain = FSIDomain(params)
    if params.general.input_mesh_dir is not None:
        domain.read_mesh_files(params.general.input_mesh_dir, params)
    else:
        domain.build(params)

    # If we only want to create the mesh, we can stop here
    if params.general.mesh_only:
        list_timings(params.comm, [TimingType.wall])
        structure, flow = [], []
        return params, structure, flow
    

    if fluid_analysis:
        flow = Flow(domain, params)
        # # # Specify the boundary conditions
        flow.build_boundary_conditions(domain, params)
        # # # Build the fluid forms
        flow.build_forms(domain, params)
    else:
        flow = None

    if structural_analysis:
        structure = Structure(domain, params)
        structure.build_boundary_conditions(domain, params)
        # # # Build the fluid forms
        structure.build_forms(domain, params)
    else:
        structure = None

    if structural_analysis and fluid_analysis:
        domain.move_mesh(structure, params)

    dataIO = DataStream(domain, flow, structure, params)
    FSI_inter = FSI(domain, flow, structure, params)

    solve_structure_interval_n = int(params.structure.dt / params.solver.dt)

    for k in range(params.solver.t_steps):
        current_time = (k + 1) * params.solver.dt

        if (
            structural_analysis
            and (k + 1) % solve_structure_interval_n == 0
            and current_time > params.fluid.warm_up_time
        ):  # :# TODO: add condition to work with fluid time step
            if fluid_analysis:
                structure.elasticity.stress_predicted.x.array[:] = (
                    2.0 * structure.elasticity.stress.x.array
                    - structure.elasticity.stress_old.x.array
                )

            structure.solve(params, dataIO)

            # if fluid_analysis:
            #     dataIO.fluid_struct(domain, flow, elasticity, params)
            # adjust pressure to avoid dissipation of pressure profile
            # flow.adjust_dpdx_for_constant_flux(params)
            if fluid_analysis:
                # pass
                domain.move_mesh(structure, params)

        if fluid_analysis and not params.general.debug_mesh_motion_only:
            flow.solve(domain, params, current_time)

        if (
            structural_analysis
            and (k + 1) % solve_structure_interval_n == 0
            and current_time > params.fluid.warm_up_time
        ):  # :# TODO: add condition to work with fluid time step
            if fluid_analysis:
                FSI_inter.fluid_struct(domain, flow, structure, params)

        if (k + 1) % params.solver.save_xdmf_interval_n == 0:
            if fluid_analysis:
                if domain.rank == 0 and not params.general.debug_mesh_motion_only:
                    print(
                        f"Time {current_time:.2f} of {params.solver.t_final:.2f} (step {k+1} of {params.solver.t_steps}, {100.0*(k+1)/params.solver.t_steps:.1f}%)"
                    )
                    print(f"| CFL = {flow.cfl_max:.4f}")

                    if params.pv_array.num_panels == 1:
                        fx = flow.integrated_force_x[0]
                        fy = flow.integrated_force_y[0]

                        print(f"| f_x (drag) = {fx:.4f}")
                        print(f"| f_y (lift) = {fy:.4f}")
                    else:
                        if structure is not None:
                            # still print info, but just for first row
                            fx = flow.integrated_force_x[0]
                            fy = flow.integrated_force_y[0]

                            print(f"| f_x (drag) of 1st row = {fx:.4f}")
                            print(f"| f_y (lift) of 1st row = {fy:.4f}")
                    if thermal_analysis:
                        print(f"| T = {flow.theta_max:.4f}")

                dataIO.save_XDMF_files(flow, domain, current_time)

            if (
                structural_analysis
                and (k + 1) % solve_structure_interval_n == 0
                and current_time > params.fluid.warm_up_time
            ):

                vec = structure.elasticity.u.vector
                u_local = vec.array  # This gives a NumPy array on local rank

                if u_local.size > 0:
                    u_reshaped = u_local.reshape(-1, domain.ndim)
                    local_def_max = np.amax(np.sum(u_reshaped**2, axis=1))
                else:
                    local_def_max = -np.inf  # So it doesn't interfere in max
                print(f"Rank {domain.comm.rank}: u_vec.size = {u_local.size}, local_def_max = {local_def_max}")

                # local_def_max = np.amax(
                #     np.sum(
                #         structure.elasticity.u.vector.array.reshape(-1, domain.ndim)
                #         ** 2,
                #         axis=1,
                #     )
                # )
                # global_def_max_list = np.zeros(params.num_procs, dtype=np.float64)
                # # global_def_max_list = vec.comm.allreduce(local_def_max, op=MPI.MAX)
                # params.comm.Gather(local_def_max, global_def_max_list, root=0)

                # Wrap scalar in NumPy array (required for Gather)
                sendbuf = np.array([local_def_max], dtype=np.float64)

                # Preallocate only on root
                if params.comm.rank == 0:
                    global_def_max_list = np.empty(params.num_procs, dtype=np.float64)
                else:
                    global_def_max_list = None

                # Perform the gather
                params.comm.Gather(sendbuf, global_def_max_list, root=0)

                # Handle on root
                if params.comm.rank == 0:
                    print("Per-rank def max values:", global_def_max_list)
                    print("Global def max:", np.max(global_def_max_list))
                
                
                if domain.rank == 0:
                    # print("Structural time is : ", current_time)
                    # print("deformation norm =", {elasticity.unorm})
                    print(
                        f"| Max Deformation = {np.sqrt(np.amax(global_def_max_list)):.2e}"
                    )
                dataIO.save_XDMF_files(structure, domain, current_time)

    list_timings(params.comm, [TimingType.wall])

    return params, structure, flow


# Print profiling results
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    params, structure, flow = main()
    profiler.disable()

    if params.rank == 0:
        prof_filename = os.path.join(params.general.output_dir, "profiling.prof")
        profiler.dump_stats(prof_filename)

        prof_txt_filename = os.path.join(params.general.output_dir, "profiling.txt")

        with open(prof_txt_filename, "w") as output_file:
            sys.stdout = output_file
            profiler.print_stats(sort="cumtime")
            sys.stdout = sys.__stdout__

        if not params.general.mesh_only:
            write_metrics(flow, prof_filename=prof_txt_filename)

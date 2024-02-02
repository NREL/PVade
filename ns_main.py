from pvade.fluid.FlowManager import Flow
from pvade.DataStream import DataStream
from pvade.Parameters import SimParams
from pvade.Utilities import get_input_file, write_metrics
from pvade.geometry.MeshManager import FSIDomain

from dolfinx.common import TimingType, list_timings
import cProfile
import sys
import tqdm.autonotebook
import numpy as np


from pvade.structure.ElasticityManager import Elasticity


def main():
    # Get the path to the input file from the command line
    input_file = get_input_file()
    # input_file = "inputs/sim_params_alt.yaml"  # get_input_file()

    # Load the parameters object specified by the input file
    params = SimParams(input_file)

    fluid_analysis = params.general.fluid_analysis
    structural_analysis = params.general.structural_analysis

    # Initialize the domain and construct the initial mesh
    domain = FSIDomain(params)

    if params.general.input_mesh_dir is not None:
        domain.read_mesh_files(params.general.input_mesh_dir, params)
    else:
        domain.build(params)
        domain.write_mesh_files(params)

    if params.general.mesh_only == True:
        list_timings(params.comm, [TimingType.wall])
        elasticity, flow = [], []
        return params, elasticity, flow

    # Check to ensure mesh node matching for periodic simulations
    # if domain.periodic_simulation:
    # domain.check_mesh_periodicity(params)
    # sys.exit()

    flow = Flow(domain, fluid_analysis)
    elasticity = Elasticity(domain, structural_analysis, params)

    if fluid_analysis == True:
        flow = Flow(domain, fluid_analysis)
        # # # Specify the boundary conditions
        flow.build_boundary_conditions(domain, params)
        # # # Build the fluid forms
        flow.build_forms(domain, params)

    if structural_analysis == True:
        elasticity.build_boundary_conditions(domain, params)
        # # # Build the fluid forms
        elasticity.build_forms(domain, params)

    if structural_analysis == True and fluid_analysis == True:
        # pass
        domain.move_mesh(elasticity, params)

    dataIO = DataStream(domain, flow, elasticity, params)

    # if domain.rank == 0:
    #     progress = tqdm.autonotebook.tqdm(
    #         desc="Solving PDE", total=params.solver.t_steps
    #     )

    solve_structure_interval_n = int(params.structure.dt / params.solver.dt)

    for k in range(params.solver.t_steps):
        current_time = (k + 1) * params.solver.dt

        # if domain.rank == 0:
        #     progress.update(1)
        #     print()
        # Solve the fluid problem at each timestep

        # print("time step is : ", current_time)
        # print("reaminder from modulo ",current_time % params.structure.dt )
        if (
            structural_analysis == True
            and (k + 1) % solve_structure_interval_n == 0
            and current_time > params.fluid.warm_up_time
        ):  # :# TODO: add condition to work with fluid time step
            if fluid_analysis == True:
                elasticity.stress_predicted.x.array[:] = (
                    2.0 * elasticity.stress.x.array - elasticity.stress_old.x.array
                )

            elasticity.solve(params, dataIO)

            # if fluid_analysis == True:
            #     dataIO.fluid_struct(domain, flow, elasticity, params)
            # adjust pressure to avoid dissipation of pressure profile
            # flow.adjust_dpdx_for_constant_flux(params)
            if fluid_analysis == True:
                # pass
                domain.move_mesh(elasticity, params)

        if fluid_analysis == True:
            flow.solve(domain, params, current_time)

        if (
            structural_analysis == True
            and (k + 1) % solve_structure_interval_n == 0
            and current_time > params.fluid.warm_up_time
        ):  # :# TODO: add condition to work with fluid time step
            if fluid_analysis == True:
                dataIO.fluid_struct(domain, flow, elasticity, params)

        if (k + 1) % params.solver.save_xdmf_interval_n == 0:
            if fluid_analysis == True:
                if domain.rank == 0:
                    print(
                        f"Time {current_time:.2f} of {params.solver.t_final:.2f} (step {k+1} of {params.solver.t_steps})"
                    )
                    print(f"| CFL = {flow.cfl_max:.4f}")

                    if params.pv_array.num_panels == 1:
                        lift_coeff = flow.lift_coeff_list[0]
                        drag_coeff = flow.drag_coeff_list[0]

                        print(f"| Lift = {lift_coeff:.4f}")
                        print(f"| Drag = {drag_coeff:.4f}")

                dataIO.save_XDMF_files(flow, domain, current_time)

            if (
                structural_analysis == True
                and (k + 1) % solve_structure_interval_n == 0
                and current_time > params.fluid.warm_up_time
            ):

                local_def_max = np.amax(
                    np.sum(elasticity.u.vector.array.reshape(-1, 3) ** 2, axis=1)
                )
                global_def_max_list = np.zeros(params.num_procs, dtype=np.float64)
                params.comm.Gather(local_def_max, global_def_max_list, root=0)

                if domain.rank == 0:
                    # print("Structural time is : ", current_time)
                    # print("deformation norm =", {elasticity.unorm})
                    print(
                        f"| Max Deformation = {np.sqrt(np.amax(global_def_max_list)):.2e}"
                    )
                dataIO.save_XDMF_files(elasticity, domain, current_time)

    list_timings(params.comm, [TimingType.wall])

    return params, elasticity, flow


# Print profiling results
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    params, elasticity, flow = main()
    profiler.disable()

    if params.rank == 0:
        profiler.dump_stats("profiling.prof")

        with open("profiling.txt", "w") as output_file:
            sys.stdout = output_file
            profiler.print_stats(sort="cumtime")
            sys.stdout = sys.__stdout__

        if not params.general.mesh_only:
            write_metrics(flow, elasticity)

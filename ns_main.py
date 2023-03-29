from pvade.FlowManager import Flow
from pvade.DataStream import DataStream
from pvade.Parameters import SimParams
from pvade.Utilities import get_input_file, write_metrics

from dolfinx.common import TimingType, list_timings
import cProfile
import sys
import tqdm.autonotebook


def main():

    # Get the path to the input file from the command line
    # input_file = get_input_file()
    input_file = "inputs/2d_cyld.yaml"  #get_input_file()

    # Load the parameters object specified by the input file
    params = SimParams(input_file)

    if params.domain.dim == 3:
        from pvade.geometry.MeshManager3d import FSIDomain
    elif params.domain.dim == 2:
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

    if params.general.mesh_only == True:
        list_timings(params.comm, [TimingType.wall])
        exit()

    # Check to ensure mesh node matching for periodic simulations
    # if domain.periodic_simulation:
    # domain.check_mesh_periodicity(params)

    # Initialize the function spaces for the flow
    flow = Flow(domain)

    # # # Specify the boundary conditions
    flow.build_boundary_conditions(domain, params)

    # # # Build the fluid forms
    flow.build_forms(domain, params)

    dataIO = DataStream(domain, flow, params)

    if domain.rank == 0:
        progress = tqdm.autonotebook.tqdm(
            desc="Solving PDE", total=params.solver.t_steps
        )

    for k in range(params.solver.t_steps):
        if domain.rank == 0:
            progress.update(1)

        # Solve the fluid problem at each timestep
        flow.solve(params)

        # adjust pressure to avoid dissipation of pressure profile
        # flow.adjust_dpdx_for_constant_flux(params)
        if (k + 1) % params.solver.save_xdmf_interval_n == 0:
            if domain.rank == 0:
                print(
                    f"Time {params.solver.dt*(k+1):.2f} of {params.solver.t_final:.2f}, CFL = {flow.cfl_max}"
                )

            dataIO.save_XDMF_files(flow, (k + 1) * params.solver.dt)

    list_timings(params.comm, [TimingType.wall])

    return params


# Print profiling results
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    params = main()
    profiler.disable()

    if params.rank == 0:
        profiler.dump_stats("profiling.prof")

        with open("profiling.txt", "w") as output_file:
            sys.stdout = output_file
            profiler.print_stats(sort="cumtime")
            sys.stdout = sys.__stdout__

        write_metrics()

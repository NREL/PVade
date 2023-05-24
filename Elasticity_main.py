from pvade.fluid.FlowManager import Flow
from pvade.DataStream import DataStream
from pvade.Parameters import SimParams
from pvade.Utilities import get_input_file, write_metrics
from pvade.geometry.MeshManager import FSIDomain

from dolfinx.common import TimingType, list_timings
import cProfile
import sys
import tqdm.autonotebook


from pvade.structure.ElasticityManager import Elasticity

def main():
    # Get the path to the input file from the command line
    input_file = get_input_file()
    # input_file = "inputs/sim_params_alt.yaml"  # get_input_file()

    # Load the parameters object specified by the input file
    params = SimParams(input_file)

    # Initialize the domain and construct the initial mesh
    domain = FSIDomain(params)

    if params.general.input_mesh_file is not None:
        domain.read(params.general.input_mesh_file, params)
    else:
        domain.build(params)
        domain.write_mesh_file(params)

    if params.general.mesh_only == True:
        list_timings(params.comm, [TimingType.wall])
        return params

    # Check to ensure mesh node matching for periodic simulations
    # if domain.periodic_simulation:
    # domain.check_mesh_periodicity(params)

    # # Initialize the function spaces for the flow
    # flow = Flow(domain)

    # # # # Specify the boundary conditions
    # flow.build_boundary_conditions(domain, params)

    # # # # Build the fluid forms
    # flow.build_forms(domain, params)


    fluid_analysis = False
    flow = Flow(domain,fluid_analysis)


    # intitalize structure 
    # apply only for 3d panels 
    structural_analysis = True

    elasticity = Elasticity(domain,structural_analysis)

    if structural_analysis == True:
        elasticity.build_boundary_conditions(domain, params)
        # # # Build the fluid forms
        elasticity.build_forms(domain, params)


    dataIO = DataStream(domain, flow,elasticity, params)

    if domain.rank == 0:
        progress = tqdm.autonotebook.tqdm(
            desc="Solving PDE", total=params.solver.t_steps
        )

    
    if structural_analysis == True:
        elasticity.solve(params,dataIO)
    
    if structural_analysis == True:
        if domain.rank == 0:
            print("deformation norm =", {elasticity.unorm}) 
        dataIO.save_XDMF_files_str(elasticity, 1)
    

    list_timings(params.comm, [TimingType.wall])

    return params, elasticity, flow 


# Print profiling results
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    params, elasticity, flow  = main()
    profiler.disable()

    if params.rank == 0:
        profiler.dump_stats("profiling.prof")

        with open("profiling.txt", "w") as output_file:
            sys.stdout = output_file
            profiler.print_stats(sort="cumtime")
            sys.stdout = sys.__stdout__

        if not params.general.mesh_only:
            write_metrics(flow,elasticity)

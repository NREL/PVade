import pytest
import os

@pytest.mark.unit
def test_meshing_3dpanels():
    from pvade.DataStream import DataStream
    from pvade.Parameters import SimParams
    from pvade.Utilities import get_input_file, write_metrics

    # Get the path to the input file from the command line
    input_file = "inputs/sim_params_alt.yaml"  #get_input_file()

    # Load the parameters object specified by the input file
    params = SimParams(input_file)
    from pvade.geometry.MeshManager3d import FSIDomain

    # Initialize the domain and construct the initial mesh
    domain = FSIDomain(params)
    domain.build()

@pytest.mark.unit
def test_meshing_3dcylinder():
    from pvade.DataStream import DataStream
    from pvade.Parameters import SimParams
    from pvade.Utilities import get_input_file, write_metrics

    # get the path to the input file from the command line
    input_file = "inputs/3d_cyld.yaml"  #get_input_file()

    # load the parameters object specified by the input file
    params = SimParams(input_file)
    from pvade.geometry.MeshManager3d import FSIDomain

    # initialize the domain and construct the initial mesh
    domain = FSIDomain(params)
    domain.build()
import pytest
import os
import sys


@pytest.mark.unit
def test_meshing_cylinder2d():
    input_path = "pvade/tests/inputs_test/"
    from pvade.DataStream import DataStream
    from pvade.Parameters import SimParams
    from pvade.Utilities import get_input_file, write_metrics

    # Get the path to the input file from the command line
    input_file = input_path + "2d_cyld.yaml"  # get_input_file()

    # Load the parameters object specified by the input file
    params = SimParams(input_file)

    from pvade.geometry.MeshManager2d import FSIDomain

    # Initialize the domain and construct the initial mesh
    domain = FSIDomain(params)
    domain.build(params)


@pytest.mark.unit
def test_meshing_2dpanels():
    input_path = "pvade/tests/inputs_test/"
    from pvade.DataStream import DataStream
    from pvade.Parameters import SimParams
    from pvade.Utilities import get_input_file, write_metrics

    # Get the path to the input file from the command line
    input_file = input_path + "sim_params_alt_2D.yaml"  # get_input_file()

    # Load the parameters object specified by the input file
    params = SimParams(input_file)

    from pvade.geometry.MeshManager2d import FSIDomain

    # Initialize the domain and construct the initial mesh
    domain = FSIDomain(params)
    domain.build(params)


@pytest.mark.unit()
def test_meshing_cylinder3d():
    input_path = "pvade/tests/inputs_test/"
    from pvade.DataStream import DataStream
    from pvade.Parameters import SimParams
    from pvade.Utilities import get_input_file, write_metrics

    # get the path to the input file from the command line
    input_file = input_path + "3d_cyld.yaml"  # get_input_file()

    # load the parameters object specified by the input file
    params = SimParams(input_file)
    from pvade.geometry.MeshManager3d import FSIDomain

    # initialize the domain and construct the initial mesh
    domain = FSIDomain(params)
    domain.build(params)


@pytest.mark.unit
def test_meshing_3dpanels():
    input_path = "pvade/tests/inputs_test/"
    from pvade.DataStream import DataStream
    from pvade.Parameters import SimParams
    from pvade.Utilities import get_input_file, write_metrics

    # Get the path to the input file from the command line
    input_file = input_path + "sim_params_alt.yaml"  # get_input_file()

    # Load the parameters object specified by the input file
    params = SimParams(input_file)
    from pvade.geometry.MeshManager3d import FSIDomain

    # Initialize the domain and construct the initial mesh
    domain = FSIDomain(params)
    domain.build(params)

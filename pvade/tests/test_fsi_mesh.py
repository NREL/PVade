import pytest
import os
import sys

from pvade.geometry.MeshManager import FSIDomain
from pvade.IO.DataStream import DataStream
from pvade.IO.Parameters import SimParams
from pvade.IO.Utilities import get_input_file, write_metrics

input_path = "pvade/tests/input/yaml/"


# @pytest.mark.unit
# def test_meshing_cylinder2d():
#     # Get the path to the input file from the command line
#     input_file = os.path.join(input_path, "2d_cyld.yaml")  # get_input_file()

#     # Load the parameters object specified by the input file
#     params = SimParams(input_file)

#     # Initialize the domain and construct the initial mesh
#     domain = FSIDomain(params)

#     domain.build(params)
#     domain.write_mesh_files(params)


# @pytest.mark.unit
# def test_meshing_2dpanels():
#     # Get the path to the input file from the command line
#     input_file = os.path.join(input_path, "sim_params_2D.yaml")  # get_input_file()

#     # Load the parameters object specified by the input file
#     params = SimParams(input_file)

#     # Initialize the domain and construct the initial mesh
#     domain = FSIDomain(params)

#     domain.build(params)
#     domain.write_mesh_files(params)
#     domain.test_mesh_functionspace()


# @pytest.mark.unit()
# def test_meshing_cylinder3d():
#     # get the path to the input file from the command line
#     input_file = os.path.join(input_path, "3d_cyld.yaml")  # get_input_file()

#     # load the parameters object specified by the input file
#     params = SimParams(input_file)

#     # initialize the domain and construct the initial mesh
#     domain = FSIDomain(params)

#     domain.build(params)
#     domain.write_mesh_files(params)


@pytest.mark.unit
def test_meshing_3dpanels():
    # Get the path to the input file from the command line
    input_file = os.path.join(input_path, "sim_params.yaml")  # get_input_file()

    # Load the parameters object specified by the input file
    params = SimParams(input_file)

    # Initialize the domain and construct the initial mesh
    domain = FSIDomain(params)

    domain.build(params)
    domain.write_mesh_files(params)
    domain.test_mesh_functionspace()

import pytest
import os

from pvade.FlowManager import Flow
from pvade.DataStream import DataStream
from pvade.Parameters import SimParams
from pvade.Utilities import get_input_file, write_metrics
from pvade.geometry.MeshManager import FSIDomain

from dolfinx.common import TimingType, list_timings
import cProfile
import sys
import tqdm.autonotebook

input_path = "pvade/tests/inputs_test/"


@pytest.mark.unit
def test_flow_3dpanels():
    # Get the path to the input file from the command line
    input_file = input_path + "sim_params_alt.yaml"  # get_input_file()

    # Load the parameters object specified by the input file
    params = SimParams(input_file)

    # Initialize the domain and construct the initial mesh
    domain = FSIDomain(params)
    domain.read("pvade/tests/test_mesh/panels3d")
    # Initialize the function spaces for the flow
    flow = Flow(domain)
    # # # Specify the boundary conditions
    flow.build_boundary_conditions(domain, params)
    # # # Build the fluid forms
    flow.build_forms(domain, params)
    dataIO = DataStream(domain, flow, params)
    flow.solve(params)


@pytest.mark.unit
def test_flow_2dpanels():
    # Get the path to the input file from the command line
    input_file = input_path + "sim_params_alt_2D.yaml"  # get_input_file()

    # Load the parameters object specified by the input file
    params = SimParams(input_file)

    # Initialize the domain and construct the initial mesh
    domain = FSIDomain(params)
    domain.read("pvade/tests/test_mesh/panels2d")
    # Initialize the function spaces for the flow
    flow = Flow(domain)
    # # # Specify the boundary conditions
    flow.build_boundary_conditions(domain, params)
    # # # Build the fluid forms
    flow.build_forms(domain, params)
    dataIO = DataStream(domain, flow, params)
    flow.solve(params)


@pytest.mark.unit
def test_flow_2dcylinder():
    # Get the path to the input file from the command line
    input_file = input_path + "2d_cyld.yaml"  # get_input_file()

    # Load the parameters object specified by the input file
    params = SimParams(input_file)

    # Initialize the domain and construct the initial mesh
    domain = FSIDomain(params)
    domain.read("pvade/tests/test_mesh/cylinder2d")
    # Initialize the function spaces for the flow
    flow = Flow(domain)
    # # # Specify the boundary conditions
    flow.build_boundary_conditions(domain, params)
    # # # Build the fluid forms
    flow.build_forms(domain, params)
    dataIO = DataStream(domain, flow, params)
    flow.solve(params)


@pytest.mark.unit
def test_flow_3dcylinder():
    # Get the path to the input file from the command line
    input_file = input_path + "3d_cyld.yaml"  # get_input_file()

    # Load the parameters object specified by the input file
    params = SimParams(input_file)

    # Initialize the domain and construct the initial mesh
    domain = FSIDomain(params)
    domain.read("pvade/tests/test_mesh/cylinder3d")
    # Initialize the function spaces for the flow
    flow = Flow(domain)
    # # # Specify the boundary conditions
    flow.build_boundary_conditions(domain, params)
    # # # Build the fluid forms
    flow.build_forms(domain, params)
    dataIO = DataStream(domain, flow, params)
    flow.solve(params)

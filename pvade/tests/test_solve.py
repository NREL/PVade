import pytest
import os

import numpy as np

from pvade.fluid.FlowManager import Flow
from pvade.DataStream import DataStream
from pvade.Parameters import SimParams
from pvade.Utilities import get_input_file, write_metrics
from pvade.geometry.MeshManager import FSIDomain
from ns_main import main

from dolfinx.common import TimingType, list_timings
import cProfile
import sys
import tqdm.autonotebook

input_path = "pvade/tests/input/yaml/"

solve_iter = 10

rtol = 3.0e-5


@pytest.mark.unit
def test_flow_3dpanels():
    # Get the path to the input file from the command line
    input_file = os.path.join(input_path, "sim_params.yaml")  # get_input_file()

    # Load the parameters object specified by the input file
    params = SimParams(input_file)

    # Initialize the domain and construct the initial mesh
    domain = FSIDomain(params)
    domain.read_mesh_files("pvade/tests/input/mesh/panels3d/", params)

    print("fluid shape = ", np.shape(domain.fluid.msh.geometry.x))
    print("struct shape = ", np.shape(domain.structure.msh.geometry.x))

    fluid_analysis = params.general.fluid_analysis
    # Initialize the function spaces for the flow
    flow = Flow(domain, fluid_analysis)

    # # # Specify the boundary conditions
    flow.build_boundary_conditions(domain, params)

    # # # Build the fluid forms
    flow.build_forms(domain, params)

    os.makedirs(params.general.output_dir_sol, exist_ok=True)

    ## Initialize the function spaces for the flow
    # flow = Flow(domain)
    ## # # Specify the boundary conditions
    # flow.build_boundary_conditions(domain, params)
    ## # # Build the fluid forms
    # flow.build_forms(domain, params)
    ## dataIO = DataStream(domain, flow, None, params)

    for t_step in range(solve_iter):
        current_time = (t_step + 1) * params.solver.dt

        flow.solve(domain, params, current_time)

        max_velocity = np.amax(flow.u_k.x.array)
        max_pressure = np.amax(flow.p_k.x.array)

        print("max_velocity = ", max_velocity)
        print("max_pressure = ", max_pressure)
        assert not np.any(np.isnan(flow.p_k.x.array))

    max_velocity_truth = 18.198961285088807
    max_pressure_truth = 56.925361572846874
    assert np.isclose(max_velocity, max_velocity_truth, rtol=rtol)
    assert np.isclose(max_pressure, max_pressure_truth, rtol=rtol)


def test_fsi2():

    input_file = os.path.join(input_path, "flag2d.yaml")  # get_input_file()

    params, elasticity, flow = main(input_file=input_file)

    pos_filename = os.path.join(params.general.output_dir_sol, "accel_pos.csv")
    lift_and_drag_filename = os.path.join(
        params.general.output_dir_sol, "lift_and_drag.csv"
    )

    pos_data = np.genfromtxt(pos_filename, skip_header=1, delimiter=",")
    lift_and_drag_data = np.genfromtxt(
        lift_and_drag_filename, skip_header=1, delimiter=","
    )

    pos_data_truth = np.array(
        [
            [0.0, 0.0],
            [-3.8502815925065954e-11, 1.0341421206449845e-14],
            [-1.7962914965954527e-10, 3.364005545276934e-14],
            [-4.877831406032804e-10, 4.561057261690559e-15],
            [-1.093417209948098e-09, -9.421444469432739e-14],
            [-2.1644288415438676e-09, -3.0779751884798964e-13],
            [-3.836703208608397e-09, -8.696041330267227e-13],
            [-6.2793477523623264e-09, -1.980296787004134e-12],
            [-9.726446894462723e-09, -3.83508099942714e-12],
            [-1.4395258246335687e-08, -6.8340754022153335e-12],
            [-2.0493668049948853e-08, -1.1455034424591473e-11],
            [-2.828753315880372e-08, -1.815470768595669e-11],
            [-3.80528711617147e-08, -2.753212637479167e-11],
            [-5.003463682873219e-08, -4.035846641412406e-11],
            [-6.451100532751562e-08, -5.7414500752734744e-11],
            [-8.178940631395472e-08, -7.952270018564122e-11],
            [-1.0214150605583043e-07, -1.0770965542766492e-10],
            [-1.2584092897287308e-07, -1.4308033746302698e-10],
            [-1.5319786313207591e-07, -1.867057929933732e-10],
            [-1.8449727872042586e-07, -2.398048546780679e-10],
        ]
    )

    lift_and_drag_data_tuth = np.array(
        [
            [
                1.000000000e-03,
                1.447174203e-02,
                -1.429135451e-04,
                0.0,
                2.894348406e-04,
                -2.858270903e-06,
                0.0,
            ],
            [
                2.000000000e-03,
                4.478047234e-02,
                -5.910991028e-04,
                0.0,
                8.956094469e-04,
                -1.182198206e-05,
                0.0,
            ],
            [
                3.000000000e-03,
                7.662458401e-02,
                -1.166437342e-03,
                0.0,
                1.532491680e-03,
                -2.332874685e-05,
                0.0,
            ],
            [
                4.000000000e-03,
                1.090917271e-01,
                -1.549059217e-03,
                0.0,
                2.181834542e-03,
                -3.098118433e-05,
                0.0,
            ],
            [
                5.000000000e-03,
                1.422174280e-01,
                -1.844610069e-03,
                0.0,
                2.844348561e-03,
                -3.689220138e-05,
                0.0,
            ],
            [
                6.000000000e-03,
                1.756516101e-01,
                -2.302221183e-03,
                0.0,
                3.513032202e-03,
                -4.604442367e-05,
                0.0,
            ],
            [
                7.000000000e-03,
                2.095643216e-01,
                -2.702722109e-03,
                0.0,
                4.191286432e-03,
                -5.405444218e-05,
                0.0,
            ],
            [
                8.000000000e-03,
                2.441691147e-01,
                -2.893282934e-03,
                0.0,
                4.883382293e-03,
                -5.786565869e-05,
                0.0,
            ],
            [
                9.000000000e-03,
                2.791051252e-01,
                -3.173738055e-03,
                0.0,
                5.582102503e-03,
                -6.347476110e-05,
                0.0,
            ],
            [
                1.000000000e-02,
                3.142977415e-01,
                -3.521544987e-03,
                0.0,
                6.285954831e-03,
                -7.043089974e-05,
                0.0,
            ],
            [
                1.100000000e-02,
                3.501157256e-01,
                -3.632976303e-03,
                0.0,
                7.002314512e-03,
                -7.265952606e-05,
                0.0,
            ],
            [
                1.200000000e-02,
                3.863649691e-01,
                -3.714773367e-03,
                0.0,
                7.727299382e-03,
                -7.429546734e-05,
                0.0,
            ],
            [
                1.300000000e-02,
                4.227567674e-01,
                -3.939535627e-03,
                0.0,
                8.455135348e-03,
                -7.879071254e-05,
                0.0,
            ],
            [
                1.400000000e-02,
                4.596215584e-01,
                -4.000498313e-03,
                0.0,
                9.192431169e-03,
                -8.000996626e-05,
                0.0,
            ],
            [
                1.500000000e-02,
                4.970228501e-01,
                -3.910000927e-03,
                0.0,
                9.940457001e-03,
                -7.820001854e-05,
                0.0,
            ],
            [
                1.600000000e-02,
                5.345733314e-01,
                -3.965101880e-03,
                0.0,
                1.069146663e-02,
                -7.930203760e-05,
                0.0,
            ],
            [
                1.700000000e-02,
                5.724088202e-01,
                -3.989947060e-03,
                0.0,
                1.144817640e-02,
                -7.979894120e-05,
                0.0,
            ],
            [
                1.800000000e-02,
                6.108194444e-01,
                -3.795907648e-03,
                0.0,
                1.221638889e-02,
                -7.591815296e-05,
                0.0,
            ],
            [
                1.900000000e-02,
                6.494805024e-01,
                -3.677795842e-03,
                0.0,
                1.298961005e-02,
                -7.355591685e-05,
                0.0,
            ],
            [
                2.000000000e-02,
                6.882712185e-01,
                -3.662151033e-03,
                0.0,
                1.376542437e-02,
                -7.324302067e-05,
                0.0,
            ],
        ]
    )

    assert np.allclose(pos_data, pos_data_truth)
    print(lift_and_drag_data)
    # assert np.allclose(lift_and_drag_data, lift_and_drag_data_tuth)


# @pytest.mark.unit
# def test_flow_2dpanels():
#    # Get the path to the input file from the command line
#    input_file = os.path.join(input_path, "sim_params_2D.yaml")  # get_input_file()
#
#    # Load the parameters object specified by the input file
#    params = SimParams(input_file)
#
#    # Initialize the domain and construct the initial mesh
#    domain = FSIDomain(params)
#    domain.read("pvade/tests/test_mesh/panels2d/mesh.xdmf", params)
#    # Initialize the function spaces for the flow
#    flow = Flow(domain)
#    # # # Specify the boundary conditions
#    flow.build_boundary_conditions(domain, params)
#    # # # Build the fluid forms
#    flow.build_forms(domain, params)
#    # dataIO = DataStream(domain, flow, None, params)
#
#    for t_step in range(solve_iter):
#        flow.solve(params)
#
#        max_velocity = np.amax(flow.u_k.x.array)
#        max_pressure = np.amax(flow.p_k.x.array)
#
#        print("max_velocity = ", max_velocity)
#        print("max_pressure = ", max_pressure)
#
#    max_velocity_truth = 3.4734894184978726
#    max_pressure_truth = 1.698213865642233
#    assert np.isclose(max_velocity, max_velocity_truth, rtol=rtol)
#    assert np.isclose(max_pressure, max_pressure_truth, rtol=rtol)
#
#
# @pytest.mark.unit
# def test_flow_2dcylinder():
#    # Get the path to the input file from the command line
#    input_file = os.path.join(input_path, "2d_cyld.yaml")  # get_input_file()
#
#    # Load the parameters object specified by the input file
#    params = SimParams(input_file)
#
#    # Initialize the domain and construct the initial mesh
#    domain = FSIDomain(params)
#    domain.read("pvade/tests/test_mesh/cylinder2d/mesh/mesh.xdmf", params)
#    fluid_analysis = params.general.fluid_analysis
#    # Initialize the function spaces for the flow
#    flow = Flow(domain,fluid_analysis)
#
#    # # # Specify the boundary conditions
#    flow.build_boundary_conditions(domain, params)
#
#    # # # Build the fluid forms
#    flow.build_forms(domain, params)
#    # dataIO = DataStream(domain, flow, None, params)
#
#    for t_step in range(solve_iter):
#        flow.solve(params)
#
#        max_velocity = np.amax(flow.u_k.x.array)
#        max_pressure = np.amax(flow.p_k.x.array)
#
#        print("max_velocity = ", max_velocity)
#        print("max_pressure = ", max_pressure)
#
#    max_velocity_truth = 1.8113852701695827
#    max_pressure_truth = 1.3044593668958533
#    assert np.isclose(max_velocity, max_velocity_truth, rtol=rtol)
#    assert np.isclose(max_pressure, max_pressure_truth, rtol=rtol)
#
#
# @pytest.mark.unit
# def test_flow_3dcylinder():
#    # Get the path to the input file from the command line
#    input_file = os.path.join(input_path, "3d_cyld.yaml")  # get_input_file()
#
#    # Load the parameters object specified by the input file
#    params = SimParams(input_file)
#
#    # Initialize the domain and construct the initial mesh
#    domain = FSIDomain(params)
#    #domain.read("pvade/tests/test_mesh/cylinder3d/mesh.xdmf", params)
#    domain.build(params)
#    fluid_analysis = params.general.fluid_analysis
#    # Initialize the function spaces for the flow
#    flow = Flow(domain,fluid_analysis)
#
#    # # # Specify the boundary conditions
#    flow.build_boundary_conditions(domain, params)
#
#    # # # Build the fluid forms
#    flow.build_forms(domain, params)
#    # dataIO = DataStream(domain, flow, None, params)
#
#    for t_step in range(solve_iter):
#        flow.solve(params)
#
#        max_velocity = np.amax(flow.u_k.x.array)
#        max_pressure = np.amax(flow.p_k.x.array)
#
#        print("max_velocity = ", max_velocity)
#        print("max_pressure = ", max_pressure)
#
#    max_velocity_truth = 0.6242970092279582
#    max_pressure_truth = 0.30929163498498147
#    assert np.isclose(max_velocity, max_velocity_truth, rtol=rtol)
#    assert np.isclose(max_pressure, max_pressure_truth, rtol=rtol)

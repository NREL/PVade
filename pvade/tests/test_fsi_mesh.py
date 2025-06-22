import pytest
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

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


@pytest.mark.unit
@pytest.mark.parametrize(
    "wind_direction, num_stream_rows, num_span_rows",
    [
        (15.0, 2, 1),
        (105.0, 2, 1),
        (195.0, 2, 1),
        (285.0, 2, 1),
        (15.0, 3, 2),
        (105.0, 3, 2),
        (195.0, 3, 2),
        (285.0, 3, 2),
        (15.0, 4, 3),
        (105.0, 4, 3),
        (195.0, 4, 3),
        (285.0, 4, 3),
    ],
)
def test_meshing_3dpanels_rotations(wind_direction, num_stream_rows, num_span_rows):
    # Get the path to the input file from the command line
    input_file = os.path.join(input_path, "sim_params.yaml")  # get_input_file()

    # Load the parameters object specified by the input file
    params = SimParams(input_file)

    # Override default options to change wind direction to 250.0
    params.general.output_dir = "pvade/tests/output/panels3d_rotations"
    params.general.output_dir_mesh = "pvade/tests/output/panels3d_rotations/mesh"

    params.domain.x_min = -30.0
    params.domain.x_max = 50.0
    params.domain.y_min = -30.0
    params.domain.y_max = 30.0

    params.pv_array.stream_rows = num_stream_rows
    params.pv_array.span_rows = num_span_rows
    params.pv_array.span_spacing = 15.0
    params.pv_array.tracker_angle = list(
        np.linspace(-52.0, 52.0, num_stream_rows * num_span_rows)
    )

    params.fluid.wind_direction = wind_direction

    # Initialize the domain and construct the initial mesh
    domain = FSIDomain(params)

    domain.build(params)
    domain.write_mesh_files(params)

    coords = domain.structure.msh.geometry.x

    # These are the centroids of each table in the num_stream_rows x num_span_rows array
    xc = params.pv_array.stream_spacing * np.arange(params.pv_array.stream_rows)
    yc = params.pv_array.span_spacing * np.arange(params.pv_array.span_rows)

    # Arrays always start at xc = 0, but are centered in the y-direction
    # so now shift the mean to 0.0
    yc -= np.mean(yc)

    counter = 0

    for span_row in range(params.pv_array.span_rows):
        for stream_row in range(params.pv_array.stream_rows):

            span = params.pv_array.panel_span
            chord = params.pv_array.panel_chord
            thickness = params.pv_array.panel_thickness

            tracker_angle_rad = np.radians(params.pv_array.tracker_angle[counter])

            # Create the 4 corners of this table corresponding to the *top* surface (+0.5*thickness)
            top_surface_corners = np.array(
                [
                    [-0.5 * chord, -0.5 * span, 0.5 * thickness],
                    [0.5 * chord, -0.5 * span, 0.5 * thickness],
                    [0.5 * chord, 0.5 * span, 0.5 * thickness],
                    [-0.5 * chord, 0.5 * span, 0.5 * thickness],
                ]
            )

            # Create a rotation matrix for the tilt angle about the torque tube
            Ry = np.array(
                [
                    [np.cos(tracker_angle_rad), 0.0, np.sin(tracker_angle_rad)],
                    [0.0, 1.0, 0.0],
                    [
                        -np.sin(tracker_angle_rad),
                        0.0,
                        np.cos(tracker_angle_rad),
                    ],
                ]
            )

            # Rotate the panel currently centered at (0, 0, 0) about the y-axis (for tilt angle)
            top_surface_corners = np.dot(Ry, top_surface_corners.T).T

            # Shift the panels into the correct position for their centroid,
            # note that since the wind rotation is about the center of mass (xc_bar, 0)
            # we remove that component from xc before shifting. this is equivalent
            # to performing the rotation for wind angle about the point (xc_bar, yc_bar=0)
            # in the following step.
            mean_x_position = np.mean(xc)
            top_surface_corners[:, 0] += xc[stream_row] - mean_x_position
            top_surface_corners[:, 1] += yc[span_row]
            top_surface_corners[:, 2] += params.pv_array.elevation

            # Apply the rotation about the z-axis due to the non-270 wind direction
            array_rotation = (params.fluid.wind_direction + 90.0) % 360.0
            array_rotation_rad = np.radians(array_rotation)

            Rz = np.array(
                [
                    [
                        np.cos(array_rotation_rad),
                        -np.sin(array_rotation_rad),
                        0.0,
                    ],
                    [
                        np.sin(array_rotation_rad),
                        np.cos(array_rotation_rad),
                        0.0,
                    ],
                    [0.0, 0.0, 1.0],
                ]
            )

            top_surface_corners = np.dot(Rz, top_surface_corners.T).T

            # Shift the array by the amount omitted in the previous step
            # such that the center of mass is (xc_bar, 0)
            top_surface_corners[:, 0] += mean_x_position

            # Now, for each of the 4 corners defining the oriented top surface
            # we can assert that there is a point in the structural mesh
            # that exactly aligns with it (expressed by getting the distance from
            # the truth point to every point in the structrual mesh and asserting
            # that one of those distances is 0)
            for z, corner in enumerate(top_surface_corners):
                distance = np.linalg.norm(corner - coords, axis=1)

                min_dist = np.amin(distance)

                assert np.isclose(min_dist, 0.0)

                # print(f"Found corner {z+1} of 4 on panel {counter+1} of {num_stream_rows * num_span_rows} tilted at {params.pv_array.tracker_angle[counter]:.1f} deg")

            counter += 1

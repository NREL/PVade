import pytest
import dolfinx
import ufl
import numpy as np
import os
import matplotlib.pyplot as plt

from pvade.Parameters import SimParams
from pvade.geometry.MeshManager import FSIDomain

input_path = "pvade/tests/input/yaml/"


@pytest.mark.regression
def test_calc_distance_to_panel_surface():
    input_file = os.path.join(input_path, "embedded_box.yaml")  # get_input_file()

    params = SimParams(input_file)

    domain = FSIDomain(params)
    domain.build(params)

    # Call the function that calculates the distance to the panel surface
    min_dist_cutoff = 1.0e-8
    domain._calc_distance_to_panel_surface(params, min_dist_cutoff=min_dist_cutoff)

    # After this step, the domain should have a distance attribute
    assert hasattr(domain, "distance")

    # Get the coordinates of the function space and get the values of the distance function
    coords = domain.V1.tabulate_dof_coordinates()
    vals = domain.distance.vector.array[:]

    # Assert that all distance values are positive (minimum should be min_dist_cutoff)
    assert np.all(vals > 0)

    # The number of rows in coords should be equal to the number of distance values
    # but this won't be true in parallel since the mesh includes all ghost points
    # and the functions do not, so we assert >=
    assert np.shape(coords)[0] >= np.shape(vals)[0]

    # The minimum distance should be the one enforced by min_dist_cutoff
    min_dist = np.amin(vals)
    assert np.isclose(min_dist, min_dist_cutoff)

    # The maximum distance should be at the separation between corners
    # (i.e., the hypotenuse, or close to sqrt(0.5^2 + 0.5^2 + 0.5^2))
    max_dist = np.amax(vals)

    dx = params.domain.x_max - 0.5 * params.pv_array.panel_chord
    dy = params.domain.y_max - 0.5 * params.pv_array.panel_span
    dz = params.domain.z_max - 0.5 * params.pv_array.panel_thickness
    truth_max_dist = np.sqrt(dx * dx + dy * dy + dz * dz)

    assert np.isclose(max_dist, truth_max_dist)


@pytest.mark.regression
def test_move_mesh():
    input_file = os.path.join(input_path, "embedded_box.yaml")  # get_input_file()

    params = SimParams(input_file)

    domain = FSIDomain(params)
    # domain.build(params)
    # This will exist by evaluating the previous test, test_calc_distance_to_panel_surface,
    # first in the sequence, as the files are created during the write_to_read_hack step.
    domain.read_mesh_files(os.path.join(params.general.output_dir, "mesh"), params)

    # Create a dummy elasticity object that we can prescribe displacement values for
    class dummy_elasticity:
        def __init__(self, domain, x_shift, y_shift, z_shift):
            # Build a dummy displacement function to test mesh movement with
            P1 = ufl.VectorElement("Lagrange", domain.structure.msh.ufl_cell(), 1)
            V = dolfinx.fem.FunctionSpace(domain.structure.msh, P1)
            self.uh_delta = dolfinx.fem.Function(V)

            self.uh_delta.vector.array[0::3] = x_shift
            self.uh_delta.vector.array[1::3] = y_shift
            self.uh_delta.vector.array[2::3] = z_shift

    # Specify the distance to shift the structure in the x, y, and z direction
    x_shift = 0.05
    y_shift = 0.1
    z_shift = 0.2

    # Build the elasticity, prescribe those motions for uh_delta (used to move the mesh)
    elasticity = dummy_elasticity(domain, x_shift, y_shift, z_shift)

    # Make a copy of the locations before for testing
    fluid_coords_before = np.copy(domain.fluid.msh.geometry.x)
    structure_coords_before = np.copy(domain.structure.msh.geometry.x)

    # Assert the generated mesh for fluid and structure hits all the right min/max values
    assert np.isclose(np.amin(fluid_coords_before[:, 0]), params.domain.x_min)
    assert np.isclose(np.amin(fluid_coords_before[:, 1]), params.domain.y_min)
    assert np.isclose(np.amin(fluid_coords_before[:, 2]), params.domain.z_min)

    assert np.isclose(np.amax(fluid_coords_before[:, 0]), params.domain.x_max)
    assert np.isclose(np.amax(fluid_coords_before[:, 1]), params.domain.y_max)
    assert np.isclose(np.amax(fluid_coords_before[:, 2]), params.domain.z_max)

    assert np.isclose(
        np.amin(structure_coords_before[:, 0]), -0.5 * params.pv_array.panel_chord
    )
    assert np.isclose(
        np.amin(structure_coords_before[:, 1]), -0.5 * params.pv_array.panel_span
    )
    assert np.isclose(
        np.amin(structure_coords_before[:, 2]), -0.5 * params.pv_array.panel_thickness
    )

    assert np.isclose(
        np.amax(structure_coords_before[:, 0]), 0.5 * params.pv_array.panel_chord
    )
    assert np.isclose(
        np.amax(structure_coords_before[:, 1]), 0.5 * params.pv_array.panel_span
    )
    assert np.isclose(
        np.amax(structure_coords_before[:, 2]), 0.5 * params.pv_array.panel_thickness
    )

    # Move the mesh by the amount prescribed in uh_delta
    domain.move_mesh(elasticity, params, tt=0)

    # Get a copy of the new positions
    fluid_coords_after = np.copy(domain.fluid.msh.geometry.x[:])
    structure_coords_after = np.copy(domain.structure.msh.geometry.x[:])

    # Calculate the total displacement
    delta_fluid_coords = fluid_coords_after - fluid_coords_before
    delta_structure_coords = structure_coords_after - structure_coords_before

    # Assert that the fluid mesh moved, at most, the amount specified
    assert np.isclose(np.amax(delta_fluid_coords[:, 0]), x_shift)
    assert np.isclose(np.amax(delta_fluid_coords[:, 1]), y_shift)
    assert np.isclose(np.amax(delta_fluid_coords[:, 2]), z_shift)

    # Assert that the structure mesh moved all points uniformly
    assert np.allclose(delta_structure_coords[:, 0], x_shift)
    assert np.allclose(delta_structure_coords[:, 1], y_shift)
    assert np.allclose(delta_structure_coords[:, 2], z_shift)

    # Assert that the fluid still has the same bounding box
    # (i.e., no penetration of the moved nodes through the original domain)
    assert np.isclose(np.amin(fluid_coords_after[:, 0]), params.domain.x_min)
    assert np.isclose(np.amin(fluid_coords_after[:, 1]), params.domain.y_min)
    assert np.isclose(np.amin(fluid_coords_after[:, 2]), params.domain.z_min)

    for k in range(3):
        max_val = np.amax(fluid_coords_after[:, k])
        print(f"in dir {k}, max = {max_val}")

    assert np.isclose(np.amax(fluid_coords_after[:, 0]), params.domain.x_max)
    assert np.isclose(np.amax(fluid_coords_after[:, 1]), params.domain.y_max)
    assert np.isclose(np.amax(fluid_coords_after[:, 2]), params.domain.z_max)

    # fig, ax = plt.subplots(1, 2)
    # ax[0].scatter(fluid_coords_before[:, 0], fluid_coords_before[:, 1])
    # ax[0].scatter(structure_coords_before[:, 0], structure_coords_before[:, 1])
    # ax[1].scatter(fluid_coords_after[:, 0], fluid_coords_after[:, 1])
    # ax[1].scatter(structure_coords_after[:, 0], structure_coords_after[:, 1])
    # plt.show()

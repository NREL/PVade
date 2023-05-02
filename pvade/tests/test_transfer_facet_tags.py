import pytest
import dolfinx
import numpy as np

from pvade.Parameters import SimParams
from pvade.geometry.MeshManager import FSIDomain


@pytest.mark.unit
@pytest.mark.parametrize(
    "sub_domain_name, marker_name",
    [
        ("fluid", "x_min"),
        ("structure", "x_min"),
        ("fluid", "x_max"),
        ("structure", "x_max"),
        ("fluid", "y_min"),
        ("structure", "y_min"),
        ("fluid", "y_max"),
        ("structure", "y_max"),
        ("fluid", "z_min"),
        ("structure", "z_min"),
        ("fluid", "z_max"),
        ("structure", "z_max"),
        ("fluid", "internal_surface"),
        ("structure", "internal_surface"),
    ],
)
def test_transfer_facet_tags(sub_domain_name, marker_name):
    input_path = "pvade/tests/inputs_test/"

    # Get the path to the input file from the command line
    input_file = input_path + "sim_params_alt.yaml"  # get_input_file()

    # Load the parameters object specified by the input file
    params = SimParams(input_file)

    # Initialize the domain and construct the initial mesh
    domain = FSIDomain(params)
    domain.build(params)

    # Get the sub-domain object
    sub_domain = getattr(domain, sub_domain_name)

    idx = domain.domain_markers[marker_name]["idx"]

    facets_from_tag = sub_domain.facet_tags.find(idx)

    def x_min_wall(x):
        return np.isclose(x[0], params.domain.x_min)

    def x_max_wall(x):
        return np.isclose(x[0], params.domain.x_max)

    def y_min_wall(x):
        return np.isclose(x[1], params.domain.y_min)

    def y_max_wall(x):
        return np.isclose(x[1], params.domain.y_max)

    def z_min_wall(x):
        return np.isclose(x[2], params.domain.z_min)

    def z_max_wall(x):
        return np.isclose(x[2], params.domain.z_max)

    def internal_surface(x):
        tol = 1e-3

        x_mid = np.logical_and(
            params.domain.x_min + tol < x[0], x[0] < params.domain.x_max - tol
        )
        y_mid = np.logical_and(
            params.domain.y_min + tol < x[1], x[1] < params.domain.y_max - tol
        )
        z_mid = np.logical_and(
            params.domain.z_min + tol < x[2], x[2] < params.domain.z_max - tol
        )

        return np.logical_and(x_mid, np.logical_and(y_mid, z_mid))

    if marker_name == "x_min":
        facets_from_dolfinx = dolfinx.mesh.locate_entities_boundary(
            sub_domain.msh, domain.ndim - 1, x_min_wall
        )
    if marker_name == "x_max":
        facets_from_dolfinx = dolfinx.mesh.locate_entities_boundary(
            sub_domain.msh, domain.ndim - 1, x_max_wall
        )
    if marker_name == "y_min":
        facets_from_dolfinx = dolfinx.mesh.locate_entities_boundary(
            sub_domain.msh, domain.ndim - 1, y_min_wall
        )
    if marker_name == "y_max":
        facets_from_dolfinx = dolfinx.mesh.locate_entities_boundary(
            sub_domain.msh, domain.ndim - 1, y_max_wall
        )
    if marker_name == "z_min":
        facets_from_dolfinx = dolfinx.mesh.locate_entities_boundary(
            sub_domain.msh, domain.ndim - 1, z_min_wall
        )
    if marker_name == "z_max":
        facets_from_dolfinx = dolfinx.mesh.locate_entities_boundary(
            sub_domain.msh, domain.ndim - 1, z_max_wall
        )
    if marker_name == "internal_surface":
        facets_from_dolfinx = dolfinx.mesh.locate_entities_boundary(
            sub_domain.msh, domain.ndim - 1, internal_surface
        )

    assert len(facets_from_tag) == len(facets_from_dolfinx)
    assert np.allclose(facets_from_tag, facets_from_dolfinx)

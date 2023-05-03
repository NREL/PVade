import pytest
import os

from pvade.Parameters import SimParams

filename = "pvade/tests/test_inputs.yaml"


@pytest.fixture()
def create_params():
    # Create the file
    with open(filename, "w") as fp:
        fp.write("general:\n")
        fp.write("  geometry_module: test_name\n")
        fp.write("  output_dir: path/to/solution/dir\n")
        fp.write("domain:\n")
        fp.write("  x_min: 0.0\n")
        fp.write("  x_max: 2.5\n")
        fp.write("solver:\n")
        fp.write("  t_final: 1.0\n")
        fp.write("  dt: 0.1\n")
        fp.write("  save_xdmf_interval: 0.5\n")
        fp.write("  save_text_interval: 0.5\n")

    # Run the actual test
    yield

    # Teardown
    os.remove(filename)


@pytest.fixture()
def create_bad_params():
    # Create the file
    with open(filename, "w") as fp:
        fp.write("generalTYPO:\n")
        fp.write("  geometry_module: test_name\n")
        fp.write("  output_dir: path/to/solution/dir\n")

    # Run the actual test
    yield

    # Teardown
    os.remove(filename)


@pytest.mark.unit
def test_parameters(create_params):
    params = SimParams(filename)

    assert params.general.output_dir == "path/to/solution/dir"

    assert params.domain.x_min == pytest.approx(0.0)

    assert params.solver.t_steps == 10
    assert params.solver.save_xdmf_interval_n == 5


@pytest.mark.unit
def test_bad_parameters(create_bad_params):
    with pytest.raises(Exception):
        params = SimParams(filename)

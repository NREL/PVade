import pytest
import subprocess
import glob

# import sys

import os

rootdir = os.getcwd()
print("rootdir = ", rootdir)

# sys. get current absolute path and make that the prefix for pvade_main
# maybe specify input_mesh_dir from command line using root
# sys.path.append('../..')


def launch_sim(test_file):
    dt = 0.001  # 0.001
    tf = dt * 10  # ten timesteps

    command = (
        f"mpirun -n 2 python "
        + rootdir
        + "/pvade_main.py --input_file "
        + test_file["path_to_file"]
        + " --domain.l_char "
        + str(test_file["l_char"])
        + " --solver.dt "
        + str(dt)
        + " --solver.t_final "
        + str(tf)
    )
    print(command)
    # command = "echo hello"
    # command = "python ../../pvade_main.py --input_file ./input/yaml/flag2d.yaml"

    out = subprocess.run(command.split(), capture_output=True, text=True)
    print(out)
    # try:
    #     tmp = subprocess.check_call(command.split())
    #     return 1  # no errors
    # except:  # if any error
    #     return 0


list_of_test_files = []

list_of_test_files.append(
    {
        "path_to_file": rootdir + "/pvade/tests/input/yaml/sim_params.yaml",
        "l_char": 20.0,
    }
)

list_of_test_files.append(
    {
        "path_to_file": os.path.join(rootdir, "/pvade/tests/input/yaml/flag2d.yaml"),
        "l_char": 0.01,
    }
)


@pytest.mark.parametrize("test_file", list_of_test_files)
def test_launch_with_different_input_files(test_file):
    print("test_file = ", test_file)
    result = launch_sim(test_file)
    print(result)

    # assert result == 1


# test_launch_with_different_input_files(files_list)
# test_launch_with_different_input_files("../../input/flag2d.yaml")
# test_launch_with_different_input_files("../../input/2d_cyld.yaml")

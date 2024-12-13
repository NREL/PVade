import pytest
import subprocess
import glob
# import sys

import os
rootdir = os.getcwd()
print('rootdir = ', rootdir)

# sys. get current absolute path and make that the prefix for pvade_main
# maybe specify input_mesh_dir from command line using root
# sys.path.append('../..')

def launch_sim(input_file):
    dt = 0.001  # 0.001
    tf = dt * 10  # ten timesteps
    l_char = 0.01

    command = (
        f"python " 
        + rootdir + "/pvade_main.py --input_file "
        + input_file
        + " --domain.l_char "
        + str(l_char)
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


# files_list = glob.glob(rootdir+"/pvade/tests/input/yaml/*.yaml")
files_list = [rootdir+"/pvade/tests/input/yaml/flag2d.yaml", 
              rootdir+"/pvade/tests/input/yaml/2d_cyld.yaml"]
print(files_list)


@pytest.mark.parametrize("input_file", files_list)
def test_launch_with_different_input_files(input_file):
    print("input_file = ", input_file)
    result = launch_sim(input_file)
    print(result)

    # assert result == 1


# test_launch_with_different_input_files(files_list)
# test_launch_with_different_input_files("../../input/flag2d.yaml")
# test_launch_with_different_input_files("../../input/2d_cyld.yaml")

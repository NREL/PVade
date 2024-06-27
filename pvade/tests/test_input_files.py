import pytest
# import subprocess, sys
import os
import glob

def launch_sim(input_file):
    dt = 0.001 # 0.001
    tf = dt*10 # ten timesteps
    l_char = 0.01

    command = "mpirun -n 8 python ../../ns_main.py --input_file ../../input/" + input_file + " --domain.l_char " + str(l_char) + " --solver.dt " + str(dt) + " --solver.t_final " + str(tf)
    # command = "mpirun -n 8 python ../../ns_main.py --input_file " + input_file + " --domain.l_char " + str(l_char) + " --solver.dt " + str(dt) + " --solver.t_final " + str(tf)
    print(command)
    os.system(command)

# print(os.path.abspath(os.curdir))
# print(sys.path[1])
# files_list = os.listdir("../../input/")
files_list = glob.glob('../../input/*.yaml')
print(files_list)
# @pytest.mark.parametrize("input_file", ["flag2d.yaml","2d_cyld.yaml","3d_cyld.yaml",
#                                         "sim_params.yaml","sim_params_2D.yaml","single_row.yaml"])
@pytest.mark.parametrize("input_file", ["flag2d.yaml","flag2d.yaml"])
# @pytest.mark.parametrize("input_file", ["../../input/flag2d.yaml","../../input/2d_cyld.yaml"])
# @pytest.mark.parametrize("input_file", files_list)
def test_launch_with_different_input_files(input_file):
  print("input_file = ",input_file)
  launch_sim(input_file)

  # not sure the best way to test that this ran - with or without the next line shows "Pass" when there are errors, or just hangs if there is an error
  # assert True # just checks if it runs without errors

# os.system("pwd")
# test_launch_with_different_input_files(files_list)
# test_launch_with_different_input_files("flag2d.yaml")
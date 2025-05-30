import argparse


def get_input_file():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        metavar="",
        type=str,
        help="The full path to the input file, e.g., 'inputs/my_params.yaml'",
    )

    command_line_inputs, unknown = parser.parse_known_args()

    try:
        input_file_path = vars(command_line_inputs)["input_file"]
        assert input_file_path is not None

    except:
        raise ValueError("No input file specified, quitting.")

    return input_file_path


def write_metrics(flow, elasticity, prof_filename="profiling.txt"):
    with open(prof_filename, "r") as output_file:
        if flow.fluid_analysis == True:
            # solver_line = [line for line in output_file if "(solve)" in line]
            solver_line = [line for line in output_file if "(solve)" in line]
            print(solver_line)
            solver_line = solver_line[0].split()
            print(
                "solve: total time = ",
                solver_line[3],
                " time per call  = ",
                solver_line[4],
                " num calls = ",
                solver_line[0],
            )

    with open(prof_filename, "r") as output_file_1:
        if flow.fluid_analysis == True:
            # solver_line = [line for line in output_file if "(solve)" in line]
            solver1_line = [line for line in output_file_1 if "_solver_step_1" in line]
            print(solver1_line)
            solver1_line = solver1_line[0].split()
            print(
                "solver 1: total time = ",
                solver1_line[3],
                " time per call  = ",
                solver1_line[4],
                " num calls = ",
                solver1_line[0],
            )

    with open(prof_filename, "r") as output_file:
        if flow.fluid_analysis == True:
            solver2_line = [line for line in output_file if "_solver_step_2" in line]
            print(solver2_line)
            solver2_line = solver2_line[0].split()
            print(
                "solver 2: total time = ",
                solver2_line[3],
                " time per call  = ",
                solver2_line[4],
                " num calls = ",
                solver2_line[0],
            )

    with open(prof_filename, "r") as output_file:
        if flow.fluid_analysis == True:
            solver3_line = [line for line in output_file if "_solver_step_3" in line]
            print(solver3_line)
            solver3_line = solver3_line[0].split()
            print(
                "solver 3: total time = ",
                solver3_line[3],
                " time per call  = ",
                solver3_line[4],
                " num calls = ",
                solver3_line[0],
            )

    with open(prof_filename, "r") as output_file:
        if flow.fluid_analysis == True:
            meshread_line = [line for line in output_file if "(read)" in line]
            if meshread_line:
                meshread_line = meshread_line[0].split()
                print(
                    "mesh read: total time = ",
                    meshread_line[3],
                    " time per call  = ",
                    meshread_line[4],
                    " num calls = ",
                    meshread_line[0],
                )

    with open(prof_filename, "r") as output_file:
        if flow.fluid_analysis == True:
            meshbuild_line = [line for line in output_file if "(build)" in line]
            if meshbuild_line:
                meshbuild_line = meshbuild_line[0].split()
                print(
                    "mesh build: total time = ",
                    meshbuild_line[3],
                    " time per call  = ",
                    meshbuild_line[4],
                    " num calls = ",
                    meshbuild_line[0],
                )

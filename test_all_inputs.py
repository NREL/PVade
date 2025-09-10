import subprocess
from pathlib import Path
import pytest


@pytest.mark.parametrize("mesh_only", [True, False])
@pytest.mark.parametrize("nprocs", [1])
def test_pvade_run(input_file, mesh_only, nprocs):
    cmd = [
        "mpirun",
        "-n",
        str(nprocs),
        "python",
        "-u",
        "pvade_main.py",
        "--input",
        str(input_file.resolve()),
        "--solver.dt",
        "0.001",
        "--solver.t_final",
        "0.005",
        "--general.mesh_only",
        str(mesh_only).lower(),
    ]

    # Add special argument for duramat_case_study.yaml
    if input_file.name in [
        "duramat_case_study.yaml",
        "turbinflow_duramat_case_study.yaml",
    ]:
        cmd += ["--domain.l_char", "4"]

    print(
        f"\n=== Running: {input_file.name} | mesh_only={mesh_only} | nprocs={nprocs} ==="
    )
    print("Command:", " ".join(cmd))

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    print("OUTPUT LOG")
    for line in proc.stdout:
        print(line, end="")

    print("ERROR LOG")
    for line in proc.stderr:
        print(line, end="")

    proc.wait()

    assert proc.returncode == 0, (
        f"\nFAILED: {input_file.name} | mesh_only={mesh_only} | nprocs={nprocs}\n"
        f"Exit code: {proc.returncode}"
    )

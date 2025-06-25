import subprocess
from pathlib import Path
import pytest

@pytest.mark.parametrize("mesh_only", [True, False])
@pytest.mark.parametrize("nprocs", [1, 8])
def test_pvade_run(input_file, mesh_only, nprocs):
    cmd = [
        "mpirun", "-n", str(nprocs),
        "python", "pvade_main.py",
        "--input", str(input_file),
        "--solver.dt", "0.001",
        "--solver.t_final", "0.005",
        "--general.mesh_only", str(mesh_only).lower()
    ]

    print(f"\n=== Running: {input_file.name} | mesh_only = {mesh_only} | nprocs = {nprocs} ===\n")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    for line in proc.stdout:
        print(line, end="")

    proc.wait()
    assert proc.returncode == 0, f"{input_file.name} (mesh_only={mesh_only}, nprocs={nprocs}) failed"


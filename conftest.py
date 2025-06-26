from pathlib import Path

def pytest_addoption(parser):
    parser.addoption(
        "--input-file", action="store", default=None,
        help="Run test only for a specific input YAML file"
    )

def pytest_generate_tests(metafunc):
    if "input_file" not in metafunc.fixturenames:
        return

    input_file_arg = metafunc.config.getoption("input_file")

    if input_file_arg:
        metafunc.parametrize("input_file", [Path(input_file_arg)])
    else:
        # 🔧 Always resolve input/ relative to location of conftest.py (PVade/)
        this_dir = Path(__file__).resolve().parent  # PVade/
        input_dir = this_dir / "input"

        all_files = sorted(input_dir.glob("*.yaml"))
        if not all_files:
            raise RuntimeError(f"No input files found in {input_dir}")
        metafunc.parametrize("input_file", all_files)


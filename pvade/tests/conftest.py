from pathlib import Path


def pytest_addoption(parser):
    parser.addoption(
        "--input-file",
        action="store",
        default=None,
        help="Run test only for this specific input YAML file",
    )


def pytest_generate_tests(metafunc):
    input_file_arg = metafunc.config.getoption("input_file")

    if "input_file" in metafunc.fixturenames:
        if input_file_arg:
            metafunc.parametrize("input_file", [Path(input_file_arg)])
        else:
            input_dir = Path("input")
            all_files = list(input_dir.glob("*.yaml"))
            metafunc.parametrize("input_file", all_files)

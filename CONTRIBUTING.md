# Contributing to PVade

Thank you for your interest in contributing to PVade.
Contributions of all kinds are welcome, including:

- Bug reports and reproducible issue cases.
- New features and solver/mesh-generation improvements.
- Test improvements.
- Documentation updates.

## Before You Start

1. Check whether your topic already exists in the issue tracker: https://github.com/NREL/PVade/issues
2. If not, open a new issue with a clear description and expected behavior.
3. Comment on the issue before starting implementation work, especially for larger changes.

## Development Setup

Use the project conda environment from the repository root:

```bash
conda env create -n pvade -f environment.yaml
conda activate pvade
```

This environment includes runtime dependencies (FEniCSx/DOLFINx, Gmsh, MPI, etc.) plus
testing and documentation tooling.

## Repository Layout

- `pvade_main.py`: entry point for running a simulation.
- `pvade/fluid/`, `pvade/structure/`, `pvade/fsi/`: fluid, structural, and fluid-structure
  interaction solver code.
- `pvade/geometry/`: geometry and mesh generation.
- `pvade/IO/`: input parameter parsing, logging, and data streaming.
- `pvade/tests/`: pytest suite (unit and regression tests).
- `test_all_inputs.py`: end-to-end test that runs `pvade_main.py` against every example
  input file.
- `examples/`: example YAML input files.
- `docs/`: Sphinx documentation source.

## Local Validation

Run the unit/regression test suite before opening a pull request:

```bash
PYTHONPATH=. pytest -sv pvade/tests/
```

Run the full end-to-end suite against all example inputs:

```bash
pytest -sv test_all_inputs.py
```

To target a specific input file used by the parametrized tests:

```bash
pytest -sv pvade/tests/ --input-file examples/panels3d.yaml
```

Format code with Black:

```bash
black pvade
```

Build docs locally when changing documentation:

```bash
cd docs
make html
```

## Coding Guidelines

- Follow PEP 8 and keep code changes focused.
- Prefer small, reviewable pull requests over large mixed changes.
- Add or update tests when fixing bugs or adding behavior.
- Keep user-facing defaults and input-file behavior backward compatible where practical.
- Use the `unit` and `regression` pytest markers (defined in `pytest.ini`) appropriately
  when adding new tests.

## Pull Request Checklist

Before submitting a pull request, confirm:

- The change is linked to an issue (or clearly justified).
- `PYTHONPATH=. pytest -sv pvade/tests/` passes locally.
- `pytest -sv test_all_inputs.py` passes locally, if applicable to your change.
- `black pvade` has been applied.
- Documentation is updated when behavior, inputs, or outputs changed.
- The PR description explains what changed, why it changed, and how it was validated.

## CI Notes

Current CI (`.github/workflows/test_pvade.yaml`) runs on pull requests to `main`, `dev`,
`sync`, and `dev_wrap`, and on pushes to `main`:

- `pytest -sv pvade/tests/` and `pytest -sv test_all_inputs.py` on Ubuntu and macOS.
- Black formatting checks.

If your change affects platform behavior, please call that out in the PR description.

## Reporting Bugs

When reporting a bug, include:

- PVade version or commit hash.
- Operating system and Python version.
- Input file (or minimal subset) to reproduce.
- Full traceback and a short reproduction sequence.

Thanks for helping improve PVade.

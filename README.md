# PVOPT
This project will develop an open source fluid-structure interaction model which can be used as part of a modeling chain to provide stressor inputs to mechanical module models to study the physics of failure for degradation mechanisms such as cell cracking, weathering of cracked cells, and glass breakage.

https://github.nrel.gov/pages/warsalan/PVOPT/

# Getting Started
1. To use this software, begin by creating a Conda environment using the provided `environment.yaml` file:
    ```bash
    conda env create -n my_env_name -f environment.yaml
    ```
    where `my_env_name` can be replaced with a short name for your Conda environment. When the environment finishes installing, activate it with:
    ```bash
    conda activate my_env_name
    ```
    from within your activate Conda environment, a simulation can be executed with:
    ```bash
    python main.py --command_line_arg value
    ```

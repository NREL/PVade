# PVade

`PVade` is an open source fluid-structure interaction model which can be used to study wind loading and stability on solar-tracking PV arrays. `PVade` can be used as part of a larger modeling chain to provide stressor inputs to mechanical module models to study the physics of failure for degradation mechanisms such as cell cracking, weathering of cracked cells, and glass breakage. For more information, visit the [PVade Documentation](https://pvade.readthedocs.io/en/latest/index.html).

[![test_pvade](https://github.com/NREL/PVade/actions/workflows/test_pvade.yaml/badge.svg)](https://github.com/NREL/PVade/actions/workflows/test_pvade.yaml)
[![Documentation Status](https://readthedocs.org/projects/pvade/badge/?version=latest)](https://pvade.readthedocs.io/en/latest/?badge=latest)

## Getting Started

New users are encouraged to review the [Getting Started](https://pvade.readthedocs.io/en/latest/how_to_guides/getting_started.html) guide which describes how to create the Conda environment and run the example simulations.

## Developer Quick Start

1. To use this software, begin by creating a Conda environment using the provided `environment.yaml` file:
    ```bash
    conda env create -n my_env_name -f environment.yaml
    ```
    where `my_env_name` can be replaced with a short name for your Conda environment. When the environment finishes installing, activate it with:
    ```bash
    conda activate my_env_name
    ```
2. From within your activate Conda environment, a simulation can be executed with:
    ```bash
    python ns_main.py --command_line_arg value
    ```

## Citation

To cite PVade, please use the "Cite this repository" feature available on the right-hand side of this repository page or copy the BibTeX reference below:

```bash
@software{Young_PVade_PV_Aerodynamic_2023,
    author = {Young, Ethan and Arsalane, Walid and Stanislawski, Brooke and He, Xin and Ivanov, Chris and Dana, Scott and Deceglie, Michael},
    doi = {10.11578/dc.20231208.1},
    month = sep,
    title = {{PVade (PV Aerodynamic Design Engineering) [SWR-23-49]}},
    url = {https://github.com/NREL/PVade},
    year = {2023}
}
```
Panels 3D
=========

This example runs a full 3D fluid-structure interaction simulation for a
single PV panel arrangement and is representative of production-style PVade
usage.

Panels 3D Case Type
-------------------

- Geometry module: ``panels3d``
- Physics: coupled fluid and structure
- Structural solve: enabled

Panels 3D Input File
--------------------

The baseline 3D panels case is defined in:

- ``input/panels3d.yaml``

For the Duramat turbulent-inflow case-study configuration, use:

- ``input/turbinflow_duramat_case_study.yaml``

Key settings include:

- 3D atmospheric domain bounds
- Uniform inflow profile
- Optional turbulence model (Smagorinsky in the default input)
- Panel geometry and orientation controls
- Coupled solver settings and output cadence

Panels 3D Run Command
---------------------

From the repository root:

.. code-block:: bash

	conda run -n PVade mpirun -n 4 python pvade_main.py --input_file input/panels3d.yaml

Duramat case-study run:

.. code-block:: bash

	conda run -n PVade mpirun -n 4 python pvade_main.py --input_file input/turbinflow_duramat_case_study.yaml

For a short debug run:

.. code-block:: bash

	conda run -n PVade mpirun -n 4 python pvade_main.py --input_file input/panels3d.yaml --solver.t_final 0.2

Panels 3D Output
----------------

Results are written under:

- ``output/panels3d/mesh``
- ``output/panels3d/solution``

These outputs contain fluid fields, interface stress projections, and
structural response history.

Panels 3D Tips
--------------

- Change wind direction with ``--fluid.wind_direction <deg>``.
- Tune panel spacing and dimensions in the ``pv_array`` section.
- For lower cost startup runs, increase ``domain.l_char``.

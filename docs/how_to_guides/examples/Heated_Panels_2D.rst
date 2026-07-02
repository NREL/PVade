Heated Panels 2D
================

This example extends the 2D panels FSI case to include thermal transport in the
fluid domain and heat-related boundary conditions.

Heated Panels 2D Case Type
--------------------------

- Geometry module: ``panels2d``
- Physics: coupled fluid and structure with thermal analysis
- Structural solve: enabled
- Thermal solve: enabled

Heated Panels 2D Input File
---------------------------

The case is defined in:

- ``input/heated_panels2d.yaml``

Key settings include:

- Multi-panel row setup (``stream_rows=8``)
- Thermal analysis flag enabled (``general.thermal_analysis: True``)
- Uniform inflow and thermal properties (``fluid.alpha``, ``fluid.beta``, ``fluid.g``)
- Ambient, bottom-wall, and panel temperature controls
  (``fluid.T_ambient``, ``fluid.T_bottom``, ``fluid.T0_panel``)
- Coupled CFD/CSD timestepping and solver settings

Heated Panels 2D Run Command
----------------------------

From the repository root:

.. code-block:: bash

   conda run -n PVade python pvade_main.py --input_file input/heated_panels2d.yaml

For a shorter test run:

.. code-block:: bash

   conda run -n PVade python pvade_main.py --input_file input/heated_panels2d.yaml --solver.t_final 1.0

Heated Panels 2D Output
-----------------------

Results are written under:

- ``output/heatedpanels2d/mesh``
- ``output/heatedpanels2d/solution``

Output files include fluid velocity/pressure, structural response fields, and
fluid temperature when thermal analysis is enabled.

Heated Panels 2D Tips
---------------------

- If runtime is high, increase ``domain.l_char`` for a coarser startup mesh.
- Use ``solver.save_xdmf_interval`` to control output cadence and file size.
- Tune thermal behavior through ``fluid.T_bottom``, ``fluid.T0_panel``, and
  ``fluid.alpha``.

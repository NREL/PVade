Panels 2D
=========

This example runs a 2D fluid-structure interaction simulation for a single
panel section. It is a practical entry point for coupled CFD+CSD workflows.

Panels 2D Case Type
-------------------

- Geometry module: ``panels2d``
- Physics: coupled fluid and structure
- Structural solve: enabled

Panels 2D Input File
--------------------

The default case is defined in:

- ``input/panels2d.yaml``

Key settings include:

- One panel row (``stream_rows=1``, ``span_rows=1``)
- Panel geometry (chord, thickness, tracker angle)
- Fluid velocity reference ``u_ref=0.8``
- Structural elasticity and body-force terms
- Dirichlet structural fixation list via ``structure.bc_list``

Panels 2D Run Command
---------------------

From the repository root:

.. code-block:: bash

	conda run -n PVade python pvade_main.py --input_file input/panels2d.yaml

Panels 2D Output
----------------

Results are written under:

- ``output/panels2d/mesh``
- ``output/panels2d/solution``

The solution output contains flow fields plus structural displacement,
velocity, and acceleration history.

Panels 2D Tips
--------------

- For faster testing, run with ``--solver.t_final 0.2``.
- Adjust structural stiffness through ``structure.elasticity_modulus``.
- Adjust panel orientation through ``pv_array.tracker_angle``.

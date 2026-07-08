Panels 2D
=========

This example runs a 2D fluid-structure simulation of a photovoltaic panel
cross-section using the built-in ``panels2d`` geometry module.

Panels 2D Workflow Overview
---------------------------

- 2D domain and mesh generation for a panel-in-flow setup
- Coupled fluid and structural solves in a single run
- Output of flow and structural fields for post-processing

Panels 2D Input File
--------------------

The reference input file is:

- ``examples/panels2d.yaml``

Key settings in this file include:

- ``general.geometry_module: panels2d``
- ``general.output_dir: output/panels2d``
- ``general.fluid_analysis: True``
- ``general.structural_analysis: True``

Selected parameters:

.. code-block:: yaml

	 domain:
		 x_min: -10
		 x_max: 50
		 y_min: 0
		 y_max: 20
		 l_char: 0.5

	 pv_array:
		 panel_chord: 2.0
		 panel_span: 7.0
		 panel_thickness: 0.1
		 tracker_angle: 30.0

	 fluid:
		 u_ref: 0.8
		 nu: 0.001

	 structure:
		 elasticity_modulus: 1.0e+05
		 poissons_ratio: 0.3
		 bc_list: ["top"]

Panels 2D Run Command
---------------------

From the repository root:

.. code-block:: bash

	 conda run -n PVade python pvade_main.py --input examples/panels2d.yaml

Panels 2D Expected Output
-------------------------

Results are written under:

- ``output/panels2d/mesh``
- ``output/panels2d/solution``
- ``output/panels2d/logfile.log``

Panels 2D Notes
---------------

- Set ``general.mesh_only: true`` to generate meshes without advancing the
	transient solve.
- Use shorter ``solver.t_final`` while testing configuration changes.
- Increase ``domain.l_char`` for faster, lower-resolution debug runs.

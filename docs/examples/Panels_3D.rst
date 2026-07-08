Panels 3D
=========

This example runs a 3D panel case using the ``panels3d`` geometry module.
It is a full fluid-structure setup with configurable inflow, panel mechanics,
and array placement parameters.

Panels 3D Workflow Overview
---------------------------

- 3D panel geometry and domain creation
- Coupled CFD and structural analysis in one workflow
- Transient outputs for aerodynamic loading and structural response

Panels 3D Input File
--------------------

The reference input file is:

- ``examples/panels3d.yaml``

Key settings in this file include:

- ``general.geometry_module: panels3d``
- ``general.output_dir: output/duramat_case_study``
- ``fluid.turbulence_model: smagorinsky``
- ``structure.motor_connection: true``
- ``structure.tube_connection: true``

Selected parameters:

.. code-block:: yaml

	 domain:
		 x_min: -20.0
		 x_max: 100.0
		 y_min: -30.0
		 y_max: 30.0
		 z_min: 0.0
		 z_max: 20.0
		 l_char: 1.25

	 pv_array:
		 panel_chord: 4.1
		 panel_span: 24.25
		 panel_thickness: 0.1
		 elevation: 2.1
		 tracker_angle: 0.0

	 fluid:
		 u_ref: 16.0
		 rho: 1.0
		 nu: 1.8e-05

	 structure:
		 elasticity_modulus: 4.0e+09
		 poissons_ratio: 0.3
		 beta_relaxation: 0.5

Panels 3D Run Command
---------------------

From the repository root:

.. code-block:: bash

	 conda run -n PVade python pvade_main.py --input examples/panels3d.yaml

Panels 3D Expected Output
-------------------------

Results are written under:

- ``output/duramat_case_study/mesh``
- ``output/duramat_case_study/solution``
- ``output/duramat_case_study/logfile.log``

Panels 3D Notes
---------------

- To run a quick verification, reduce ``solver.t_final`` and optionally increase
	``domain.l_char``.
- Keep ``solver.dt`` and ``structure.dt`` consistent for tightly coupled cases.
- Start with ``general.mesh_only: true`` when validating geometry and mesh
	extents.

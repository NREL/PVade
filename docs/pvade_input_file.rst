Input File Parameters
=====================

PVade reads simulation settings from a YAML file passed with:

.. code-block:: bash

   conda run -n PVade python pvade_main.py --input <path-to-yaml>

Top-Level Sections
------------------

PVade input files are organized in these categories:

- general
- domain
- pv_array
- solver
- fluid
- structure

General
-------

The general section controls high-level run behavior.

Common keys:

- test: enable lightweight testing behavior
- geometry_module: selects the geometry implementation (for example panels2d, panels3d, flag2d, cylinder2d, cylinder3d, heliostats3d)
- output_dir: output directory for this run
- input_mesh_dir: reuse prebuilt mesh files
- mesh_only: stop after mesh generation
- structural_analysis: enable structural solve
- fluid_analysis: enable fluid solve
- thermal_analysis: enable thermal/buoyancy model when supported

Domain
------

The domain section defines the computational extents and mesh size control.

Common keys:

- x_min, x_max
- y_min, y_max
- z_min, z_max
- l_char: characteristic element size
- free_slip_along_walls: mesh-motion boundary behavior on outer walls

For 2D cases, only relevant coordinates for the selected geometry are used.

PV Array
--------

The pv_array section defines panel layout and geometry.

Common keys:

- stream_rows, span_rows
- stream_spacing, span_spacing
- panel_chord, panel_span, panel_thickness
- elevation
- tracker_angle
- span_fixation_pts

Solver
------

The solver section controls time integration, solver choices, and output cadence.

Common keys:

- dt: time step size
- t_final: final simulation time
- solver1_ksp ... solver5_ksp
- solver1_pc ... solver5_pc
- save_text_interval
- save_xdmf_interval

Fluid
-----

The fluid section controls flow properties and boundary-condition options.

Common keys:

- u_ref: reference inflow speed
- rho: fluid density
- nu: kinematic viscosity
- dpdx: optional pressure-gradient forcing
- turbulence_model: none, smagorinsky, or wale
- bc_y_min, bc_y_max, bc_z_min, bc_z_max
- wind_direction

Structure
---------

The structure section controls elastic properties and structural constraints.

Common keys:

- dt: structural time step
- rho: structural density
- elasticity_modulus
- poissons_ratio
- body_force_x, body_force_y, body_force_z
- bc_list
- motor_connection
- tube_connection
- beta_relaxation

Minimal Example
---------------

The following is a compact template to start from:

.. code-block:: yaml

   general:
     geometry_module: panels2d
     output_dir: output/panels2d
     mesh_only: false
     structural_analysis: true
     fluid_analysis: true

   domain:
     x_min: -10
     x_max: 50
     y_min: 0
     y_max: 20
     l_char: 0.5

   pv_array:
     stream_rows: 1
     span_rows: 1
     panel_chord: 2.0
     panel_span: 7.0
     panel_thickness: 0.1
     tracker_angle: 30.0

   solver:
     dt: 0.001
     t_final: 0.1
     save_text_interval: 0.01
     save_xdmf_interval: 0.01

   fluid:
     u_ref: 0.8
     nu: 0.001
     turbulence_model: none

   structure:
     dt: 0.001
     elasticity_modulus: 1.0e+05
     poissons_ratio: 0.3

Schema Reference
----------------

For the authoritative list of defaults, ranges, and allowed options, see
:doc:`input_schema`.

.. toctree::
   :maxdepth: 1

   input_schema

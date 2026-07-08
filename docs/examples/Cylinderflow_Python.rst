Cylinderflow Python Script
==========================

This page documents the standalone script ``tutorials/moving cylinder/cylinderflow.py``.

The script demonstrates a full 2D incompressible flow simulation around a
circular obstacle using DOLFINx and Gmsh, including mesh generation,
time-stepping, and field output.

Cylinderflow Script Purpose
---------------------------

- Build a 2D channel-with-cylinder mesh directly in code via Gmsh
- Solve Navier-Stokes using a fractional-step formulation
- Apply moving obstacle velocity and inflow perturbation options
- Write solution output for post-processing

Where It Lives
--------------

- ``tutorials/moving_cylinder/cylinderflow.py``

Main Workflow
-------------

The script performs these steps:

1. Defines geometry and mesh-resolution parameters
2. Builds and tags mesh entities (inlet, outlet, walls, obstacle edge)
3. Creates finite-element spaces for velocity and pressure
4. Applies boundary conditions
5. Advances the solution in time
6. Saves fields to output files

Run Command
-----------

From the repository root:

.. code-block:: bash

  conda run -n PVade python "tutorials/moving_cylinder/cylinderflow.py"

Vortex Shedding Animation
-------------------------

The animation below (``docs/_static/videos/mesh_with_bc.mp4``) shows vorticity shading as
the cylinder oscillates vertically, highlighting wake development and periodic
vortex shedding.

.. raw:: html

  <video controls preload="metadata" width="900" style="max-width:100%;height:auto;border:1px solid #bbb;border-radius:6px;">
    <source src="../_static/videos/mesh_with_bc.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

.. note::

  If the embedded player does not load in your browser, open the file directly:
  ``docs/_build/html/_static/videos/mesh_with_bc.mp4``.

Notes
-----

- This script is a standalone solver example and is separate from the
  YAML-driven ``pvade_main.py --input_file ...`` workflow.
- You can tune mesh and physics parameters directly in the script
  (for example ``Re``, ``dt``, and mesh resolution variables).

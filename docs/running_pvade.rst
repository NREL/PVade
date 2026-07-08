Running PVade
=============

This page shows how to launch PVade simulations from the repository root.

Requirements
------------

- A working PVade environment
- A valid YAML input file
- ParaView (optional, for visualization of XDMF outputs)

.. note::
   If a parameter is omitted in your YAML file, PVade uses the default value
   from the input schema.

Basic Command
-------------

Run PVade with:

.. code-block:: bash

   conda run -n PVade python pvade_main.py --input <path-to-yaml>

Example Inputs In This Repository
---------------------------------

The table below lists example YAML files currently present in this branch.

.. table:: Example cases and input files

   ================================ ================================================
   Case                             Input file
   ================================ ================================================
   Panels 2D                        examples/panels2d.yaml
   Panels 3D                        examples/panels3d.yaml
   Single heliostat 3D              examples/single_heliostat.yaml
   Heated panels 2D                 examples/heated_panels2d.yaml
   ================================ ================================================

Common Run Examples
-------------------

Panels 3D:

.. code-block:: bash

   conda run -n PVade python pvade_main.py --input examples/panels3d.yaml

Panels 2D:

.. code-block:: bash

   conda run -n PVade python pvade_main.py --input examples/panels2d.yaml


Output Location
---------------

Output files are written to the directory set by ``general.output_dir`` in your
input YAML. Typical contents include:

- Mesh files under a mesh subdirectory
- Solution files under a solution subdirectory
- A run log file named logfile.log

CLI Parameter Overrides
-----------------------

You can override YAML values from the command line using dotted keys:

.. code-block:: bash

   conda run -n PVade python pvade_main.py --input examples/panels3d.yaml --solver.dt 0.001 --solver.t_final 0.01

More On Input Parameters
------------------------

For detailed parameter documentation, see :doc:`pvade_input_file` and
:doc:`input_schema`.

.. toctree::
   :maxdepth: 1

   pvade_input_file

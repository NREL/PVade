Cylinder 2D
===========

This example runs a classic 2D laminar cylinder-flow case in a channel.
It is useful for validating fluid-only behavior before moving to fully
coupled FSI panel simulations.

Cylinder 2D Case Type
---------------------

- Geometry module: ``cylinder2d``
- Physics: fluid only
- Structural solve: disabled

Cylinder 2D Input File
----------------------

The default case is defined in:

- ``input/2d_cyld.yaml``

Key settings in that file include:

- Channel bounds ``x=[0,2.5]``, ``y=[0,0.41]``
- Parabolic inflow profile
- No-slip top and bottom walls
- Time step ``dt=0.000625`` and final time ``t_final=3``

Cylinder 2D Run Command
-----------------------

From the repository root:

.. code-block:: bash

	conda run -n PVade python pvade_main.py --input_file input/2d_cyld.yaml

Cylinder 2D Output
------------------

Results are written under:

- ``output/cylinder2d/mesh``
- ``output/cylinder2d/solution``

The primary fluid fields (velocity and pressure) are written to the
XDMF outputs and can be visualized in ParaView.

Cylinder 2D Tips
----------------

- For quick smoke tests, reduce runtime with ``--solver.t_final 0.1``.
- To refine the mesh, decrease ``domain.l_char``.



Cylinder 3D
===========

This example is the 3D extension of the cylinder benchmark and is used to
exercise the fluid solver in a fully three-dimensional setting.

Cylinder 3D Case Type
---------------------

- Geometry module: ``cylinder3d``
- Physics: fluid only
- Structural solve: disabled

Cylinder 3D Input File
----------------------

The default case is defined in:

- ``input/3d_cyld.yaml``

Key settings in that file include:

- Domain bounds ``x=[0,2.5]``, ``y=[0,0.41]``, ``z=[0,0.41]``
- Parabolic inflow profile
- No-slip conditions on top/bottom and side walls
- Time step ``dt=0.001`` and short default run time ``t_final=0.01``

Cylinder 3D Run Command
-----------------------

From the repository root:

.. code-block:: bash

	conda run -n PVade python pvade_main.py --input_file input/3d_cyld.yaml

For parallel execution:

.. code-block:: bash

	conda run -n PVade mpirun -n 4 python pvade_main.py --input_file input/3d_cyld.yaml

Cylinder 3D Output
------------------

Results are written under:

- ``output/cylinder3d/mesh``
- ``output/cylinder3d/solution``

Use ParaView to inspect transient velocity and pressure fields.

Cylinder 3D Tips
----------------

- Increase ``solver.t_final`` for a longer transient.
- Use a smaller ``domain.l_char`` for finer cylinder resolution.

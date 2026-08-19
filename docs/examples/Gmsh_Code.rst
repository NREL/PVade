Gmsh Code Utilities
===================

The ``tutorials/Gmsh_code`` folder contains standalone geometry and meshing prototype
scripts. These files are intended for custom geometry experimentation and
pre-processing, not as the primary PVade input-driven run path.

Location
--------

- ``tutorials/Gmsh_code/demo``
- ``tutorials/Gmsh_code/fake_terrain``

What It Contains
----------------

``demo``
~~~~~~~~

This folder contains a synthetic mountain/terrain meshing prototype with files
such as:

- ``demo_elipsoidal_shape_triangle_surface.py``
- ``demo_elipsoidal_shape_triangle_surface.geo``
- ``demo_elipsoidal_triangle_surface_mesh.vtk``

The Python script builds a 3D geometry in Gmsh and writes a VTK mesh output.

``fake_terrain``
~~~~~~~~~~~~~~~~

This folder contains a terrain-from-data workflow, including:

- ``fake_terrain.csv``: source point cloud / terrain samples
- ``fake_terrain_shape_triangle_surface.py``: interpolation + geometry + meshing
- ``fake_terrain_geo.brep`` and ``fake_terrain.vtk`` outputs

The script reads terrain samples, interpolates a surface, creates volumes,
meshes in 3D, and exports VTK/BREP artifacts.

How This Relates to Example YAML Files
--------------------------------------

The YAML files under ``examples/``  (for example
``examples/panels3d.yaml`` 
are the main supported workflow for
running PVade simulations.

By contrast, files in ``tutorials/Gmsh_code`` are utility/prototype scripts that can help
with custom geometry development before integrating a geometry path into the
main PVade geometry modules.

Running a Gmsh Utility Script
-----------------------------

From the repository root, for example:

.. code-block:: bash

   conda run -n PVade python tutorials/Gmsh_code/demo/demo_elipsoidal_shape_triangle_surface.py

or

.. code-block:: bash

   conda run -n PVade python tutorials/Gmsh_code/fake_terrain/fake_terrain_shape_triangle_surface.py

Outputs are generated in the script working directory as VTK/BREP files.

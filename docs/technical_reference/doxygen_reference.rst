Doxygen Reference
=================

PVade also includes a Doxygen configuration file for generating API-style
reference output directly from source code comments.

Configuration
-------------

The Doxygen configuration is stored in ``docs/Doxyfile``.

It is configured to:

- Scan ``pvade`` recursively
- Include ``pvade_main.py``
- Write generated output into docs/_build/doxygen

Generate Doxygen Docs
---------------------

From the repository root, run:

.. code-block:: bash

   cd docs
   doxygen Doxyfile

After generation, the HTML entry point is:

- docs/_build/doxygen/html/index.html



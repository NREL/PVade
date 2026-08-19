========
Testing
========

This section covers the automated test suite used to validate PVade changes.
The tests live under ``pvade/tests/`` and are executed with ``pytest``.

Test Layout
-----------

- ``pvade/tests/test_input_files.py``: validates that the example and tutorial
  input files can be parsed
- ``pvade/tests/test_Parameters.py``: checks parameter handling and defaults
- ``pvade/tests/test_mesh_movement.py``: exercises mesh motion behavior
- ``pvade/tests/test_fsi_mesh.py``: checks coupled mesh setup
- ``pvade/tests/test_transfer_facet_tags.py``: verifies mesh tag transfer
- ``pvade/tests/test_solve.py``: runs solver-level regression checks

Running Tests
-------------

From the repository root:

.. code-block:: bash

   conda run -n PVade pytest pvade/tests

To target a specific input file used by the parametrized tests:

.. code-block:: bash

   conda run -n PVade pytest pvade/tests --input-file examples/panels3d.yaml

Markers
-------

The test suite defines the following pytest markers in ``pytest.ini``:

- ``unit``: short tests that complete quickly
- ``regression``: longer tests that compare against expected results

Additional Notes
----------------

- Test output artifacts are written under ``pvade/tests/output/``.
- When adding a new input file or tutorial case, consider adding a matching
  test in ``pvade/tests/``.
Synthetic Turbulent Inflow Generation
=====================================

This page documents the notebook used to generate turbulent inflow HDF5 files
for PVade turbulent-inflow simulations.

Notebook Location
-----------------

- ``examples/synthetic_turbulent_inflow/generate_turbulent_inflow_h5_file.ipynb``

What It Does
------------

The notebook uses PyConTurb to generate an unconstrained turbulent wind field
and exports it in the format consumed by PVade.

High-level steps:

1. Build a lateral-vertical sampling grid (:math:`y`, :math:`z`)
2. Define time vector and turbulence parameters
3. Generate turbulent velocity components (:math:`u`, :math:`v`, :math:`w`)
4. Reshape to arrays with shape ``(nt, nz, ny)``
5. Write an HDF5 file into ``input/``

Generated File Format
---------------------

The notebook writes datasets:

- ``time_index``
- ``y_coordinates``
- ``z_coordinates``
- ``u``
- ``v``
- ``w``

A typical output filename is:

- ``input/pct_turb_ny80_nz80_unconstrained_1.0s_dt0.01_uref20.h5``

Dependencies
------------

- ``pyconturb``
- ``numpy``
- ``pandas``
- ``matplotlib``
- ``h5py``

Usage with PVade
----------------

Use the generated HDF5 file in an input YAML such as
``input/turbinflow_duramat_case_study.yaml`` by setting:

- ``fluid.velocity_profile_type: specified_from_file``
- ``fluid.h5_filename: input/<your_generated_file>.h5``

Then run PVade normally with that input file.

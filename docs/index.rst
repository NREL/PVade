.. PVade documentation master file, created by
   sphinx-quickstart on Fri Mar 10 11:27:06 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PVade's documentation!
=================================

PVade is an open source fluid-structure interaction model which can be used to study wind loading and stability on solar-tracking PV arrays. PVade can be used as part of a larger modeling chain to provide stressor inputs to mechanical module models to study the physics of failure for degradation mechanisms such as cell cracking, weathering of cracked cells, and glass breakage.

.. note::
   This is an active research project and there may be areas where the documentation needs additional work to keep up with our latest developments. While we work to close this gap, feel free to raise issues or ask questions on GitHub_.

.. _GitHub: https://github.com/NREL/PVade

Organization
------------

Documentation is currently organized into three main categories:

* :ref:`User Manual`: User guides covering basic topics and use cases for the PVade software
* :ref:`Theory Manual`: Walktrough in the PVade code blocks
* :ref:`Technical Reference`: Programming details on the PVade API and functions
* :ref:`Background`: Information and research sources for fluid and structural solvers and PV topics

New users may find it helpful to review the :ref:`Getting Started` materials first.



.. image:: how_to_guides/benchmark_png/main_animation.gif 
   :alt: StreamPlayer
   :align: center

Contents
--------

.. toctree::
   :maxdepth: 2

   how_to_guides/index
   technical_reference/index
   background/index
..   theory_manual/index

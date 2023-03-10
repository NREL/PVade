Getting Started
===============

Building Conda Environment
--------------------------

To use this software, begin by creating a Conda environment using the provided ``environment.yaml`` file::

  conda env create -n my_env_name -f environment.yaml

where ``my_env_name`` can be replaced with a short name for your Conda environment. When the environment finishes installing, activate it with::

  conda activate my_env_name

from within your activate Conda environment, a simulation can be executed with::

  python main.py --command_line_arg value


Running Examples
----------------

.. Fill in with walkthrough pointing to an example
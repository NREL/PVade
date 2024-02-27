Installing PVade 
=================


On a Unix base machine 
--------------------------

PVade is a software that uses FEniCSx for its Finite Element Computation. 
For more information about FEniCSx please refer to https://github.com/FEniCS/dolfinx.

In addition to FEniCSx, PVade uses multiple python packages as part of the pre- and post-processing steps. 
PVade dependencies are included in environment.yaml. 

In order to start using PVade, we can use Conda/Mamba for the creation of an environement containg all the necessary dependencies. 

In order to obtain Mamba we can use the following ressource https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html.

In order to obtain Conda we can use the following ressource https://conda.io/projects/conda/en/latest/user-guide/install/index.html.

.. Note:: 
   We recommand using Mamba for its fast installation of the environement. 



Building Mamba Environment

To use this software, begin by creating a Conda environment using the provided ``environment.yaml`` file::

  mamba env create -n my_env_name -f environment.yaml

where ``my_env_name`` can be replaced with a short name for your Conda environment. When the environment finishes installing, activate it with::

  mamba activate my_env_name

from within your activate Conda environment, a simulation can be executed with::

  python main.py --command_line_arg value


We can test the successful installation of PVade and it's MPI implementation by running the following example ::
  
  mpirun -np $num_cores python -u $PVade/example/poissoneq.py 64  cg none 1

The example solve a Poisson's equation in 3 dimensions using 64 elements and 1 order Lagrange shape functions with cg as the ksp solver and no preconditioners. 
For more details about the poisson's problem we refer the use to the folowing link https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html 

On NREL HPC machine Kestrel 
----------------------------

In order to use PVade on Kestrel, we can use one of the two options.

* A conda/mamba installation 
* Loading FEniCSx as a module  



1. conda/mamba installation

In order to install PVade, it is recommend to use compute node. 
You can allocate one use it interactively through: 

.. code:: bash

   ~$ salloc --nodes=1 --time=4:00:00 --partition=$partition_name --account $account_name --mem=0 --exclusive

Make sure you specify the partition name ``$partition_name`` and the account name ``$account_name``. 
Next, we clone the repository from https://github.com/NREL/PVade.git.

.. code:: bash

   ~$ git clone https://github.com/NREL/PVade.git

.. note:: 
   the same can be achieved by downloading the latest release from https://github.com/NREL/PVade/releases


We will refer to ``$PVade`` as the location of the cloned repo. 
We change the directory to ``$PVade`` and load mamba. 


.. code:: bash

   ~$ cd $PVade/
   ~$ module unload PrgEnv-cray/8.3.3
   ~$ module load mamba 


.. note::

   The same can be achived by using Conda.
   Mamba was shown to be faster.

We then create an environment ``my_env_name`` and activate it.

.. code::

   ~$ mamba env create -n PVade_public -f environment.yaml
   ~$ mamba activate my_env_name

To test the installation we can run an example using the command 

.. code::

   mpirun -np $num_cores python -u $PVade/example/poissoneq.py 64  cg none 1

The example solve a Poisson's equation in 3 dimensions using 64 elements and 1 order Lagrange shape functions with cg as the ksp solver and no preconditioners. 

.. note::

   The conda installed PVade requires the use of mpirun instead of srun since it was not configured against Slurm.

   
   
2. Module access 

On Kestrel, PVade is installed and available as a module. 
PVade can be accessed by loading:

.. code::

   module load fenicsx

This instance of PVade leverages a FEniCSx installation that leverages GNU Programming environment and cray-mpich for its mpi communication.


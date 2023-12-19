HPC Jobs
========


PVade on Kestrel 
----------------

In order to use PVade on Kestrel, we can use one of the two options.

* A conda/mamba installation 
* Loading FEniCSx as a module  



1. conda/mamba installation

In order to install PVade, it is recommend to use compute node. 
You can allocate one use it interactively through: 

.. code:: bash

   ~$ salloc --nodes=1 --time=4:00:00 --partition=$partion_name --account $account_name --mem=0 --exclusive

Make sure you specify the partition name *$partion_name* and the account name *$account_name*. 
Next, we clone the repository from https://github.com/NREL/PVade.git.

.. code:: bash

   ~$ git clone https://github.com/NREL/PVade.git

.. note:: 
   the same can be achieved by downloading the latest release from https://github.com/NREL/PVade/releases


We will refer to *$PVade* as the location of the cloned repo. 
We change the directory to *$PVade* and load mamba. 


.. code:: bash

   ~$ cd $PVade/
   ~$ module unload PrgEnv-cray/8.3.3
   ~$ module load mamba 


.. note::

   The same can be achived by using Conda.
   Mamba was shown to be faster.

We then create an environment *my_env_name* and activate it.

.. code::

   ~$ mamba env create -n PVade_public -f environment.yaml
   ~$ mamba activate my_env_name

To test the installation we can run an example using the command 

.. code::

   mpirun -np #ncores python -u $PVade/example/poissoneq.py 64  cg none 1

The example solve a Poisson's equation in 3 dimensions using 64 elements and 1 order Lagrange shape functions with cg as the ksp solver and no preconditioners. 

.. note::

   The conda installed PVade requires the use of mpirun instead of srun since it was not configured against Slurm.

An Example for a job script is presented below 


.. code::

   Job script: 
   
   #!/bin/bash
   #SBATCH --ntasks-per-node=104
   #SBATCH --partition=shared
   #SBATCH --time=1:00:00
   #SBATCH --account=hpcapps 
   #SBATCH --job-name=example-pvade
   #SBATCH --nodes=1
   #SBATCH --error=pvade_example.err
   #SBATCH --output=pvade_example.out
   #SBATCH --mem=0
   #SBATCH --exclusive
   
   module purge
   module load mamba
   mamba activate my_env_name
   export OMP_NUM_THREADS=1
   
   mpirun -np $ncores python -u /kfs2/projects/hpcapps/warsalan/fenicsx_bench/src/poissoneq.py 64  cg none 1
   
   
2. Module access 

On Kestrel, PVade is installed and available as a module. 
PVade can be accessed by loading:

.. code::

   module load fenicsx

This instance of PVade leverages a FEniCSx installation that leverages GNU Programming environment and cray-mpich for its mpi communication.

a Job script example is shown below: 


.. code::
   
    #!/bin/bash
    
    #SBATCH --ntasks-per-node=104
    #SBATCH --partition=shared
    #SBATCH --time=1:00:00
    #SBATCH --account=hpcapps
    #SBATCH --job-name=example-pvade
    #SBATCH --nodes=1
    #SBATCH --error=pvade_example.err
    #SBATCH --output=pvade_example.out
    #SBATCH --mem=0
    #SBATCH --exclusive
    
    
    module purge
    ml PrgEnv-gnu
    ml fenicsx/0.6.0-gcc
    export OMP_NUM_THREADS=1
    
    srun -n 104 python -u /kfs2/projects/hpcapps/warsalan/fenicsx_bench/src/poissoneq.py 64  cg none 1


.. note::

   Things to keep in mind when using the FEnicsX module are 
     * PrgEnv-gnu needs to be loaded to acces gcc and cray-mpich 
     * *srun* is the luncher to be used 

 






.. Fill in with walkthrough pointing to an example

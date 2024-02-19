Running on HPC
==============


Job submission on Kestrel 
------------------------

As Mentioned in ,In order to use PVade on Kestrel, we can use one of the two options.

* A conda/mamba installation 
* Loading FEniCSx as a module  



1. conda/mamba installation

.. note::

   The conda installed PVade requires the use of mpirun instead of srun since it was not configured against Slurm.

An Example for a job script is presented below 


.. code::

   Job script: 
   
   #!/bin/bash
   #SBATCH --ntasks-per-node=104
   #SBATCH --partition=shared
   #SBATCH --time=1:00:00
   #SBATCH --account=$account_name 
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
   
   mpirun -np $ncores python -u $PVade/example/poissoneq.py 64  cg none 1
   
   
2. Module access 


a Job script example is shown below: 


.. code::
   
    #!/bin/bash
    
    #SBATCH --ntasks-per-node=104
    #SBATCH --partition=shared
    #SBATCH --time=1:00:00
    #SBATCH --account=$account_name
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
    
    srun -n 104 python -u $PVade/example/poissoneq.py 64  cg none 1


.. note::

   Things to keep in mind when using the FEnicsX module are 
     * PrgEnv-gnu needs to be loaded to acces gcc and cray-mpich 
     * *srun* is the luncher to be used 

 
.. PVade Performance on Kestrel 
.. ----------------------------





.. Fill in with walkthrough pointing to an example

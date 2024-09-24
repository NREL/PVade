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



Accessing the DOLFINx programming environment on Windows: a review
------------------------------------------------------------------------------

Current as of: May 2024.

[Instructions for DOLFINx](https://github.com/FEniCS/dolfinx/blob/main/README.md)

1. Using WSL2

A Windows 11 computer should have WSL natively. In a command prompt, ``wsl --status`` will tell you if you have WSL version 2, which is generally preferred. 

To install a Ubuntu distro in WSL 2, in a command prompt, run::

   wsl --install

You'll be instructed to create a username and password. After install, your prompt will be logged in to Ubuntu on WSL. You can also access this prompt in the future by searching your apps for Ubuntu.

In Ubuntu, follow the instructions to install the DOLFINx environment.::


   add-apt-repository ppa:fenics-packages/fenics
   apt update
   apt install fenicsx


This may take up to an hour to install. 

Following installation, you can access your environment by opening Ubuntu. You can use ``apt`` or ``pip`` to install any additional packages you need. 
If you want to access this environment in VSCode, you can use the [WSL extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl), which works very similar to remotely accessing HPC systems through SSH with the [SSH extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh).

I found this method to be the easiest to set up, access, and understand.

2. Using Docker

[Install](https://docs.docker.com/desktop/install/windows-install/) Docker. Docker Desktop needs a license, as NREL is a large organization. I don't know if these are available. However, a Docker Destop license is not necessary for installing a DOLFINx environment in Docker.

Docker uses images, which are recipes for programming environments, and containers, which are instances based on an image. Containers can be created and destroyed, entered and exited. 

To get up and running with docker, run ::

   docker run --name fenicsx -ti dolfinx/dolfinx:nightly


The image, ``dolfinx/dolfinx:nightly``, can be replaced with other options listed in the instructions. 
In most cases I would recommend ``dolfinx/dolfinx:stable``, but I used the nightly version to avoid a bug with DOLFINx that hopefully would be resolved by the time this is relevant.

This instruction creates a container from the image, and the first time it is run, it also pulls the image data. This may take up to 20 minutes. If you run this command again, it will create a new container from the same image. Note that you would want to leave out or change the `--name fenicsx` tag, and that it would take far less time the second time, as the image is already pulled.

A container runs one command. By default, that command is to open a terminal. When you exit that terminal, the container also exits. 

You may re-enter a Docker container by running::

   docker start -i fenics


Exiting and re-entering containers will keep data like files downloaded and packages installed, 
on top of the original image. Removing and re-creating a new container will not keep this data, only the original image. 

To install additional packages, you may use ``pip`` while you are inside your docker container.

There are ways to keep files from one container to another or from your container to your local computer. 
There are also ways to modify and update the docker container with additional packages. I do not know them.

Docker works on Windows. If you understand Docker, I'd recommend using Docker for DOLFINx. 
If you don't, I'd recommend WSL2. 

3. Using Conda
Windows supports Miniconda and Anaconda. Current DOLFINx instructions read::


   conda create -n fenicsx-env
   conda activate fenicsx-env
   conda install -c conda-forge fenics-dolfinx mpich pyvista


On conda-forge, pyvista supports Windows, but mpich and fenics-dolfinx do not. If that changed, you could use conda environments on Windows by:
   1. Installing Miniconda or Anaconda
   2. Opening Anaconda prompt
   3. Following the above instructions to create and set up the environment

It's unclear whether mpich and fenics-dolfinx cannot be ported to Windows, or just haven't yet been. At present, though, they aren't, so a conda DOLFINx environment through conda-forge is not possible.

4. Using Spack

Windows only technically supports spack. In theory, one would use spack on Windows according to [these instructions](https://spack.readthedocs.io/en/latest/getting_started.html#spack-on-windows). Your procedure would look like:
   1. Installing prerequisites VSCode with C++ compiler options, Python, Git

   2. Cloning spack 
   ::
      git clone https://github.com/spack/spack.git
   
   3. Opening a spack prompt, by running ``bin\spack_cmd.bat``. 

   4. Setting up spack
   ::
      spack compiler find
      spack external find cmake
      spack external find ninja
   
   5. Setting up your spack environment
   ::
      spack env create fenicsx-env
      spack env activate fenicsx-env
      spack add fenics-dolfinx+adios2 py-fenics-dolfinx cflags="-O3" fflags="-O3"
      spack install
   

I got no further than step 1; I couldn't find options to install VSCode with the needed C++ compiler. Many spackages are also not supported by Windows, so you likely would also not be able to run the `spack add` line in step 5.

Development to get spack to work on Windows is underway. Development to port the relevant spackages to Windows is not. Verdict: DOLFINx with Spack on Windows is impossible at current stages, and a pain even if possible.



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


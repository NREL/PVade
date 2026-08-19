Installing PVade 
=================


On a Unix base machine 
--------------------------

PVade is a software that uses FEniCSx for its Finite Element Computation. 
For more information about FEniCSx please refer to https://github.com/FEniCS/dolfinx.

In addition to FEniCSx, PVade uses multiple Python packages as part of the pre- and post-processing steps.
PVade dependencies are included in environment.yaml.

To start using PVade, we can use Conda/Mamba to create an environment containing all the necessary dependencies.

To obtain Mamba, we can use the following resource: https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html.

To obtain Conda, we can use the following resource: https://conda.io/projects/conda/en/latest/user-guide/install/index.html.

.. Note:: 
   We recommend using Mamba for its fast installation of the environment. 



Building Mamba Environment
----------------------------

To use this software, begin by creating a Conda environment using the provided ``environment.yaml`` file::

  mamba env create -n my_env_name -f environment.yaml

where ``my_env_name`` can be replaced with a short name for your Conda environment. When the environment finishes installing, activate it with::

  mamba activate my_env_name

From within your activated Conda environment, a simulation can be executed with::

  python pvade_main.py --command_line_arg value


We can test the successful installation of PVade and its MPI implementation by running the following example::
  
  mpirun -np $num_cores python -u $PVade/tutorials/poissoneq.py 64  cg none 1

The example solves a Poisson's equation in 3 dimensions using 64 elements and 1st order Lagrange shape functions with cg as the ksp solver and no preconditioners. 
For more details about the Poisson's problem, we refer the user to the following link: https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html 



On a Windows machine
------------------------------------------------------------------------------

See the [Instructions for DOLFINx](https://github.com/FEniCS/dolfinx/blob/main/README.md)
for background. Four installation options are described below, in order of
recommendation.

Option 1: Using WSL2 (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the easiest method to set up, access, and understand.

1. Check whether WSL2 is already installed. A Windows 11 computer should have
   WSL natively. In a command prompt, run ``wsl --status`` to check whether you
   have WSL version 2, which is generally preferred.
2. Install a Ubuntu distro in WSL2. In a command prompt, run::

     wsl --install

   You will be prompted to create a username and password. After installation,
   your prompt will be logged in to Ubuntu on WSL. You can also access this
   prompt later by searching your apps for Ubuntu.
3. In Ubuntu, install the DOLFINx environment::

     add-apt-repository ppa:fenics-packages/fenics
     apt update
     apt install fenicsx

   This may take up to an hour to install.
4. After installation, access your environment by opening Ubuntu. Use ``apt``
   or ``pip`` to install any additional packages you need. To access this
   environment in VSCode, use the
   [WSL extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl),
   which works similarly to remotely accessing HPC systems through SSH with the
   [SSH extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh).

Option 2: Using Docker
~~~~~~~~~~~~~~~~~~~~~~~

If you are already comfortable with Docker, we recommend using Docker for
DOLFINx; otherwise, we recommend WSL2 instead.

1. [Install](https://docs.docker.com/desktop/install/windows-install/) Docker.
   Docker Desktop requires a license for large organizations such as NLR;
   check license availability before proceeding. A Docker Desktop license is
   not necessary for installing a DOLFINx environment in Docker.
2. Docker uses images, which are recipes for programming environments, and
   containers, which are instances based on an image. Containers can be
   created and destroyed, entered and exited.
3. Create and start a container by running::

     docker run --name fenicsx -ti dolfinx/dolfinx:nightly

   The image ``dolfinx/dolfinx:nightly`` can be replaced with other options
   listed in the instructions; in most cases we recommend
   ``dolfinx/dolfinx:stable``.
4. The first time this command is run, it also pulls the image data, which may
   take up to 20 minutes. Running the command again creates a new container
   from the same image; in that case, leave out or change the
   ``--name fenicsx`` tag, and expect it to take far less time since the image
   is already pulled.
5. A container runs one command. By default, that command opens a terminal.
   When you exit that terminal, the container also exits.
6. Re-enter a Docker container by running::

     docker start -i fenics

   Exiting and re-entering containers keeps data such as downloaded files and
   installed packages on top of the original image. Removing and re-creating a
   container does not keep this data; only the original image is preserved.
7. To install additional packages, use ``pip`` while inside your Docker
   container.

Option 3: Using Conda
~~~~~~~~~~~~~~~~~~~~~~

.. note::

   As of this writing, ``mpich`` and ``fenics-dolfinx`` are not available for
   Windows on conda-forge, so this option is not currently possible. It is
   included here in case that changes.

Windows supports Miniconda and Anaconda. Current DOLFINx instructions read::

   conda create -n fenicsx-env
   conda activate fenicsx-env
   conda install -c conda-forge fenics-dolfinx mpich pyvista

On conda-forge, ``pyvista`` supports Windows, but ``mpich`` and
``fenics-dolfinx`` do not. If that changes, you could use conda environments on
Windows by:

1. Installing Miniconda or Anaconda
2. Opening Anaconda prompt
3. Following the above instructions to create and set up the environment

Option 4: Using Spack
~~~~~~~~~~~~~~~~~~~~~~

Windows only technically supports Spack, per
[these instructions](https://spack.readthedocs.io/en/latest/getting_started.html#spack-on-windows).
Follow this procedure:

1. Install prerequisites: VSCode with C++ compiler options, Python, and Git.
2. Clone Spack::

     git clone https://github.com/spack/spack.git

3. Open a spack prompt by running ``bin\spack_cmd.bat``.
4. Set up Spack::

     spack compiler find
     spack external find cmake
     spack external find ninja

5. Set up your Spack environment::

     spack env create fenicsx-env
     spack env activate fenicsx-env
     spack add fenics-dolfinx+adios2 py-fenics-dolfinx cflags="-O3" fflags="-O3"
     spack install


On NLR HPC machine Kestrel 
----------------------------

In order to use PVade on Kestrel, we can use one of the two options.

* A conda/mamba installation 
* Loading FEniCSx as a module  



1. conda/mamba installation

In order to install PVade, it is recommended to use a compute node. 
You can allocate one and use it interactively through: 

.. code:: bash

   ~$ salloc --nodes=1 --time=4:00:00 --partition=$partition_name --account $account_name --mem=0 --exclusive

Make sure you specify the partition name ``$partition_name`` and the account name ``$account_name``. 
Next, we clone the repository from https://github.com/NatLabRockies/PVade.

.. code:: bash

   ~$ git clone https://github.com/NatLabRockies/PVade.git

.. note:: 
   the same can be achieved by downloading the latest release from https://github.com/NatLabRockies/PVade/releases


We will refer to ``$PVade`` as the location of the cloned repo. 
We change the directory to ``$PVade`` and load mamba. 


.. code:: bash

   ~$ cd $PVade/
   ~$ module unload PrgEnv-cray/8.3.3
   ~$ module load mamba 


.. note::

   The same can be achieved by using Conda.
   Mamba was shown to be faster.

We then create an environment ``my_env_name`` and activate it.

.. code::

   ~$ mamba env create -n PVade_public -f environment.yaml
   ~$ mamba activate my_env_name

To test the installation we can run an example using the command 

.. code::

   mpirun -np $num_cores python -u $PVade/tutorials/poissoneq.py 64  cg none 1

The example solves a Poisson's equation in 3 dimensions using 64 elements and 1st order Lagrange shape functions with cg as the ksp solver and no preconditioners. 

.. note::

   The conda installed PVade requires the use of mpirun instead of srun since it was not configured against Slurm.

   
   
2. Module access 

On Kestrel, PVade is installed and available as a module. 
PVade can be accessed by loading:

.. code::

   module load fenicsx

This instance of PVade leverages a FEniCSx installation that leverages GNU Programming environment and cray-mpich for its mpi communication.


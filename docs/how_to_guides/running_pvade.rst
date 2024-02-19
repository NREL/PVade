Running PVade 
===============

Running PVade requires the following:

- A successful installation of the PVade environement 
- An input file with the simulation parameters see (refer to inpt page)
- Paraview to visualise the results


.. Note:: 
   If an input parameter is not defined in the input file, PVade will use a preset value for the parameter


Once we obtain a successful installation of PVade, we can execute the main script `ns_main.py` to run a simulation.
'ns_main.py' takes as an argument an input file that describes the simulation to run.

as of now there are 5 geometries included in PVade: 

- a 2D flag 
- a 2D cylinder 
- a 3D cylinder 
- 2D panels
- 3D panels 

we can run any of the examples by pointing to the correct input file located under $PVade/input 

.. container::
   :name: tab:my_label

   .. table:: input file for each Example

      =========== ==================== 
      Examples    Input File 
      =========== ==================== 
      2D flag     flag2d.yaml
      2D cylinder 2d_cyld.yaml 
      3D cylinder 3d_cyld.yaml
      2D panels   sim_params_2D.yaml
      3D panels   sim_params.yaml
      =========== ==================== 


In order to run a Flag 2D example we can execute the following:

.. code::
    python $PVade/ns_main.py --input input/flag2d.yaml



For more detail on the input parameters, please refer to :ref:`Input File parameters`    

.. toctree::
   :maxdepth: 1

   pvade_input_file

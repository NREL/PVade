Input File parameters 
=====================

The input parameters for PVade are orgnized in categories:

- General 
- Domain
- Pv_array
- Solver
- Fluid
- Structure 



General 
   The General Category controls the simulation fundametal through the following sub-parameters:

   - test: is a boolean flag that controls if the run is a test or a production case (True or False)

   - geometry_module: Conrols the example to run. The user hase the choise between 5 examples currently implemented (panels3d, panels2d, flag2d, panels3d, panels2d) 

   - output_dir: Sets the location of the output files (Takes a string of the path location)

   - mesh_only: A boolean flag to stop the simulation after the mesh creation step. (True to stop after the mesh gets created, False to conduct the whole simulation)

   - input_mesh_dir: Sets the location of Mesh already available. (Takes a string of the mesh location, e.g. output/panels3d/mesh)

   - structural_analysis: Boolean flag to conduct the Structural Analyis  

   - fluid_analysis: Boolean flag to conduct the Fluid Analyis


Domain 
   The Domain category defines the geometric parameters necessary for the creation of the  computational domain.
   The Comutational is a 3D box for 3D problems and a rectangle for 2D problems. 
   The Box delimitations are controlled by: 

   - x_min, x_max in the x-axis 
   - y_min, y_max in the y_axis 
   - z_min, z_max in the z-axis 


Note: for 2D problems the rectangle is in the xz-plane.

   The last parameter is l_char, which controls the mesh density and number of elements.


pv_array
   PV_array sets the panel's parameters.
   
   - stream_rows: The number of rows in the streamwise direction.
   - elevation: Height of the panels of the ground. 
   - stream_spacing: The spacing between panels in the streamwise direction.
   - panel_chord: Panel's width. 
   - panel_span: Panel's length. 
   - panel_thickness: Panel's Thickness. 
   - tracker_angle: The angle of the panel with respect to the x-axis. 
   - span_spacing: Spacing in the spanwise direction.
   - span_rows: The number of rows in the spanwise direction
   - span_fixation_pts: controls the point where the boundary condition of the torque tube is set 

.. note::
   - The streamwise direction is along the x-axis 

   - The spanwise direction is along the y-axis 

   - The elvation is computed from the center of mass of the panel.

   - The spacing is computed from the center of mass of the panels.

solver:
  The Solver category controls the types of solvers used at each step of the simulation

  - dt: Time step of the fluid simulation 
  - t_final: The total simulation time for the CFD 
  - solver1_ksp: takes a ksp_type
  - solver2_ksp: takes a ksp_type
  - solver3_ksp: takes a ksp_type
  - solver4_ksp: takes a ksp_type
  - solver1_pc: takes preonly
  - solver2_pc: takes preonly
  - solver3_pc: takes preonly
  - solver4_pc: takes preonly
  - save_text_interval: The interval at which PVade will generate output text files 
  - save_xdmf_interval: The interval at which PVade will generate Solution files (.xdmf format)


.. note::
   - solver1_ksp, solver2_ksp and solver3_ksp set the ksp solver at each step of the ipcs scheme used to solve Navier stokes equations.

   - solver1_pc, solver2_pc and solver3_pc set the perceonditioner  solver at each step of the ipcs scheme used to solve Navier stokes equations.

   - solver4_ksp and solver4_cg set the ksp solver and precondioner respectively for solve 4, which computes the stresses on the fluid side. 

   - For a list of ksp solvers and preconditioners you can visit {insert link for dolfinx solvers}
  

fluid:
  The fluid Category sets the CFD parameters for the fluid simulation
   
  - velocity_profile_type: General shape of inflow velocity profile
  - u_ref: Reference velocity at the center of the panel  
  - nu: Dynamic viscosity 
  - turbulence_model:  can be set to smagorinsky or null for no turbulence  
  - periodic: Boolean flag for periodic boundary conditions 
  - bc_y_max: can be set to slip noslip free
  - bc_y_min: can be set to slip noslip free
  - bc_z_max: can be set to slip noslip free
  - bc_z_min: can be set to slip noslip free
  - wind_direction: set the wind direction angle with respect to the panels.  Define numbers here with illustration 


structure:
  The structure Category sets the CSD parameters for the structural simulation

  - dt : set the time step for the CSD simulation 
  - elasticity_modulus: set the Elasticity modulus for the structure 
  - poissons_ratio: set poisson's ratio for the structure
  - body_force_x: set the x component of the body force 
  - body_force_y: set the y component of the body force
  - body_force_z: set the z component of the body force
  - bc_list: takes a string list with the sides to constrain e.g. [left]. add image displaying sides and explain type of constraint  
  - tube_connection: Boolean flag to constrain rotation around a a torque tube. explain torque tube True 

Input file Structure
--------------------

.. toctree::
   :maxdepth: 2

   input_schema

.. Fill in with walkthrough pointing to an example

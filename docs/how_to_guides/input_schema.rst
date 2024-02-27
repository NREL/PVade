Input Schema
============


::

  title: PV Wind Schema
  description: Definitions for a PV wind simulation/optimization
  type: object
  additionalProperties: False
  properties:
    problem:
      type: string
    general:
      type: object
      # required:
      properties:
        output_dir:
          default: output/
          type: string
          description: The folder that will contain the output files for this run, formatted as a single string.
    domain:
      type: object
      # required:
      # - x_min
      # - x_max
      properties:
        x_min:
          default: -100
          minimum: -1000
          maximum: 1000
          type: number
          description: The minimum x-value of the fluid domain.
          units: meter
        x_max:
          default: 100
          minimum: -1000
          maximum: 1000
          type: number
          description: The maximum x-value of the fluid domain.
          units: meter
        y_min:
          default: -100
          minimum: -1000
          maximum: 1000
          type: number
          description: The minimum y-value of the fluid domain.
          units: meter
        y_max:
          default: 100
          minimum: -1000
          maximum: 1000
          type: number
          description: The maximum y-value of the fluid domain.
          units: meter
        z_min:
          default: -100
          minimum: -1000
          maximum: 1000
          type: number
          description: The minimum z-value of the fluid domain.
          units: meter
        z_max:
          default: 100
          minimum: -1000
          maximum: 1000
          type: number
          description: The maximum z-value of the fluid domain.
          units: meter
        l_char:
          default: 10.0
          minimum: 0.01
          maximum: 100.0
          type: number
          description: The characteristic length to use when building the mesh.
          units: meter
    pv_array:
      type: object
      properties:
        num_rows:
          default: 3
          minimum: 1
          maximum: 10
          type: integer
          description: The number of panel rows in the array.
          units: None
        elevation:
          default: 1.5
          minimum: 0.0
          maximum: 10.0
          type: number
          description: The vertical distance between the center of the panel and the ground.
          units: meter
        spacing:
          default: [7.0]
          minimum: 1.0
          maximum: 10.0
          type: array
          description: The separation between panel rows in the streamwise direction.
          units: meter
        panel_length:
          default: 2.0
          minimum: 0.0
          maximum: 10.0
          type: number
          description: The length of the panels in the streamwise direction.
          units: meter
        panel_width:
          default: 7.0
          minimum: 1.0
          # maximum: 10.0
          type: number
          description: The length of the panel in the spanwise direction.
          units: meter
        panel_thickness:
          default: 0.1
          minimum: 0.01
          maximum: 1.0
          type: number
          description: The thickness of the panel.
          units: meter
        tracker_angle:
          default: 30.0
          minimum: -90.0
          maximum: 90.0
          type: number
          description: The orientation of the panels, positive indicates the upstream edge is closer to the ground and negative indicates the downstream edge is closer to the ground.
          units: degree
    solver:
      type: object
      properties:
        dt:
          default: 0.002
          minimum: 0.0
          type: number
          description: The timestep size to use in both the fluid and structural solver.
          units: second
        t_final:
          default: 0.1
          minimum: 0.0
          type: number
          description: The final simulation time.
          units: second
        save_text_interval:
          default: 0.02
          minimum: 0.0
          type: number
          description: The interval to use between saving text output files.
          units: second
        save_xdmf_interval:
          default: 0.1
          minimum: 0.0
          type: number
          description: The interval to use between saving visualization/XDMF output files.
          units: second
        solver1_ksp:
          default: cg
        solver2_ksp:
          default: cg
        solver3_ksp:
          default: cg
        solver1_pc:
          default: hypre
        solver2_pc:
          default: hypre
        solver3_pc:
          default: hypre
    fluid:
      type: object
      required:
        - turbulence_model
      if:
        properties:
          turbulence_model:
            const: smagorinsky
      then:
        required:
          - c_s
      else:
        if:
          properties:
            turbulence_model:
              const: wale
        then:
          required:
            - c_w
      properties:
        u_ref:
          default: 8.0
          minimum: 0.0
          maximum: 10.0
          type: number
          description: The velocity of the wind as measured at the panel elevation.
          units: meter/second
        nu:
          default: 1.8e-05
          minimum: 1.0e-06
          type: number
          description: The kinematic viscosity of the fluid.
          units: meter^s/second
        dpdx:
          default: 0.0
          minimum: 0.0
          type: number
          description: The constant pressure gradient to use to accelerate the flow.
          units: Pa/m
        turbulence_model:
          default: smagorinsky
          type: string
          description: The turbulence model to use in the fluid solver.
          enum:
            - none
            - smagorinsky
            - wale
        c_s:
          default: 0.6
          minimum: 0.0
          type: number
          description: The Smagorinsky coefficient.
        c_w:
          default: 0.5
          minimum: 0.0
          type: number
          description: The WALE coefficient.

    structure:
      type: object
      # required:
      properties:
        youngs:
          default:  190.0e+9
          minimum: 1.0e+9
          maximum: 500.0e+9
          type: number
          description: The effective Young's modulus of the panel structure.
          units: Pascal

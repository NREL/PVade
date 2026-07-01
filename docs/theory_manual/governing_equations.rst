Governing Equations
===================

PVade solves a coupled fluid-structure interaction (FSI) problem for
photovoltaic (PV) array configurations. The coupled model consists of:

- Incompressible flow in a moving fluid domain (ALE form)
- Nonlinear structural elastodynamics in a Lagrangian frame
- Traction/displacement exchange at the fluid-structure interface

Problem Decomposition
---------------------

Let :math:`\Omega_f(t)` denote the fluid domain and :math:`\Omega_s` the
structural reference domain. Their interface is :math:`\Gamma_{fs}(t)`.

The coupled solution at each time step advances:

- Fluid velocity and pressure in :math:`\Omega_f(t)`
- Structural displacement, velocity, and acceleration in :math:`\Omega_s`
- Mesh displacement that maps the fluid domain through ALE kinematics

Interface Coupling Conditions
-----------------------------

At the interface, PVade enforces:

1. Kinematic compatibility (matching motion)

.. math::

	u_f = u_{mesh} = \dot{y}_s \quad \text{on } \Gamma_{fs}(t)

2. Dynamic equilibrium (matching tractions)

.. math::

	\sigma_f n_f + \sigma_s n_s = 0 \quad \text{on } \Gamma_{fs}(t)

where :math:`\sigma_f` is the fluid Cauchy stress and :math:`\sigma_s` is the
structural traction measure mapped consistently to the interface.

Numerical Strategy
------------------

PVade uses a partitioned staggered strategy:

1. Solve the fluid system on the current deformed mesh
2. Project interface stress/traction to the structure side
3. Solve the nonlinear structural system
4. Move the fluid mesh with the new structural displacement increment
5. Continue to the next time step

This workflow is implemented in the core managers:

- :mod:`pvade.fluid.FlowManager`
- :mod:`pvade.structure.StructureMain`
- :mod:`pvade.fsi.FSI`
- :mod:`pvade.geometry.MeshManager`

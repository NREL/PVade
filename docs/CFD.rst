Computational Fluid Dynamics (CFD)
==================================

PVade solves incompressible flow using a fractional-step method in an
Arbitrary Lagrangian-Eulerian (ALE) frame to account for moving meshes.

Strong Form (ALE)
-----------------

For fluid velocity :math:`u`, pressure :math:`p`, density :math:`\rho`, and
kinematic viscosity :math:`\nu`, the incompressible Navier-Stokes equations in
ALE form are

.. math::

	\rho \left(\frac{\partial u}{\partial t} + (u-u_m)\cdot\nabla u\right)
	= -\nabla p + \nabla\cdot\left(2\rho\nu\,\varepsilon(u)\right) + f,

.. math::

	\nabla\cdot u = 0,

with mesh velocity :math:`u_m` and strain-rate tensor

.. math::

	\varepsilon(u)=\frac{1}{2}(\nabla u + \nabla u^T).

Stress Tensor and Traction
--------------------------

The fluid Cauchy stress used by PVade is

.. math::

	\sigma_f = 2\rho\nu\,\varepsilon(u) - pI,

and the interface traction on panel surfaces is

.. math::

	t_f = \sigma_f(-n_f).

Fractional-Step Time Integration
--------------------------------

PVade uses a 3-step Incremental Pressure Correction Scheme (IPCS):

1. Tentative velocity solve
2. Pressure Poisson correction
3. Velocity correction

The method combines:

- Crank-Nicolson treatment for diffusion
- Adams-Bashforth treatment for convection
- ALE convective velocity :math:`u-u_m`

Thermal Extension
-----------------

When thermal analysis is enabled, PVade also solves an advection-diffusion
equation for temperature :math:`\theta`:

.. math::

	\frac{\partial \theta}{\partial t} + u\cdot\nabla\theta
	- \alpha\nabla^2\theta = 0.

For convection-dominated cases, SUPG stabilization is activated based on an
approximate Peclet number threshold.

Boundary Conditions
-------------------

Implemented inflow options include:

- Uniform profile
- Parabolic profile
- Log-law atmospheric profile
- Time-resolved profile from HDF5 inflow data

Pressure is typically constrained at the outlet, and no-slip/slip options are
applied on external boundaries and panel surfaces depending on case settings.

Implementation Mapping for CFD
------------------------------

Main implementation is in:

- :mod:`pvade.fluid.FlowManager`
- :mod:`pvade.fluid.boundary_conditions`

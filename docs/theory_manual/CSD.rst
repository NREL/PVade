Computational Structural Dynamics (CSD)
=======================================

PVade solves nonlinear elastodynamics of the PV structure in a Lagrangian
reference frame using finite elements.

Kinematics
----------

Let :math:`\Omega_0` be the reference configuration and :math:`\Omega_t` the
deformed configuration. With reference coordinates :math:`X` and displacement
:math:`y(X,t)`:

.. math::

   x(X,t)=X+y(X,t).

Velocity and acceleration are

.. math::

   v = \dot{y}, \qquad a = \ddot{y}.

The deformation gradient and Green-Lagrange strain are

.. math::

   F = I + \nabla_X y,

.. math::

   C = F^T F, \qquad E = \frac{1}{2}(C-I).

Constitutive Model
------------------

PVade uses the Saint Venant-Kirchhoff model:

.. math::

   S = \lambda\,\text{tr}(E)I + 2\mu E,

where :math:`S` is the second Piola-Kirchhoff stress and

.. math::

   \lambda = \frac{E_Y\nu}{(1+\nu)(1-2\nu)},
   \qquad
   \mu = \frac{E_Y}{2(1+\nu)}.

The first Piola-Kirchhoff stress is

.. math::

   P = FS.

Weak Form in the Reference Configuration
----------------------------------------

Find :math:`y` such that for all virtual displacements :math:`w`:

.. math::

   \int_{\Omega_0} \rho_0 w\cdot a\,d\Omega
   + \int_{\Omega_0} \nabla_X w : P\,d\Omega
   - \int_{\Omega_0} \rho_0 w\cdot f\,d\Omega
   - \int_{\Gamma_{0,h}} w\cdot\hat{h}\,d\Gamma = 0.

Time Integration
----------------

The structural solver uses a generalized-alpha/Newmark-family update with
state variables displacement, velocity, and acceleration.

The kinematic updates are expressed in the standard form

.. math::

   a_{n+1} = \frac{u_{n+1}-u_n-\Delta t\,v_n}{\beta\Delta t^2}
   - \frac{1-2\beta}{2\beta}a_n,

.. math::

   v_{n+1} = v_n + \Delta t\left[(1-\gamma)a_n + \gamma a_{n+1}\right].

Rayleigh damping terms are included through user-configurable coefficients.

Boundary and Loading Conditions
-------------------------------

PVade supports:

- Dirichlet constraints on selected panel faces
- Optional pinning along torque-tube and motor-connection lines
- Body force loading
- Interface traction loading transferred from the fluid solution

Implementation Mapping for CSD
------------------------------

Main implementation is in:

- :mod:`pvade.structure.ElasticityAnalysis`
- :mod:`pvade.structure.StructureMain`
- :mod:`pvade.structure.boundary_conditions`

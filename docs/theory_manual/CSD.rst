Elasto-dynamics
================

Kinematics

:math:`\Omega_0` is the refrence frame at intial configuration of our
strcuture. :math:`\Omega_t` is the refrence frame of our structure in
the deformed state. :math:`X` is the the coordinates of the initial or
reference configuration. :math:`y` is the dispalacement with respect to
the initial configuration.

The mapping between :math:`\Omega_0` anf :math:`\Omega_t` can be
expressed as

.. math:: x(X,t) = X + y(X,t)

The velocity :math:`u` and acceleration :math:`a` of the structure are
obtained by differentiating the displacement :math:`y` with respect to
time holding the material coordinate :math:`X` fixed

.. math::

   \begin{aligned}
   u &=& \frac{dy}{dt} \\
   a &=& \frac{d^2y}{dt^2} \\
   \end{aligned}

The deformation Gradient :math:`F` is expressed as

.. math::

   \begin{aligned}
   F &=& \frac{\partial x }{\partial X} = \nabla u + I \\
   \end{aligned}

the Cauchy–Green deformation tensor :math:`C` is expressed as

.. math::

   \begin{aligned}
   C &=& F^T F 
   \end{aligned}

the Green–Lagrange strain tensor :math:`E` is expressed as

.. math::

   \begin{aligned}
   E &=& \frac{1}{2} (C - I ) = 0.5 \times (F^T F - I ) \\
   \end{aligned}

The determinant of the deformation gradient :math:`J` is given by

.. math::

   \begin{aligned}
   J &=& \text{det} F \\
   \end{aligned}

Principal of virtual work and variational form of structural mechanics 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The derivation of the weak form starts with the principal of virtual
work,

.. math::

   \begin{aligned}
       \delta W &=& \delta W_{int} +  \delta W_{ext} = 0 
   \end{aligned}

where :math:`W`, :math:`W_{ext}` and :math:`W_{int}` are the total work,
external and internal work, respectively, and :math:`\delta` denotes
their variation with respect to the virtual displacement :math:`w`.

.. math:: \delta W =  \frac{d}{d \varepsilon} W(y+\varepsilon W) |_{\varepsilon=0}

The external virtual work includes work done by the inertial and body
forces and surface tractions.

.. math:: \delta W_{ext} = \int_{\Omega_t} w.\rho (f-a) d\Omega + \int_{(\Gamma_t)_h} w.h d\Gamma,

The internal virtual work is due to the internal stresses and can be
expressed as

.. math:: \delta W_{int} = - \int_{\Omega_0} \delta E : S d\Omega

:math:`S` is is the second Piola–Kirchhoff stress tensor, which is
symmetric and work-conjugate to :math:`E`.

the variational formulation of the structural mechanics problem:find the
structural displacement :math:`y` such that for all :math:`w`

.. math:: \int_{\Omega_t} w . \rho a d \Omega  + \int_{\Omega_0} \delta E : S d\Omega  - \int_{\Omega_t} w.\rhof d \Omega - \int_{(\Gamma_t)_h} w.h d\Gamma = 0

Structural Mechanics Formulation in the Reference Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

we first start by formulating the external virtual work in the reference
configuration

.. math:: \int_{\Omega_t} w. \rho (a-f) d \Omega =  \int_{\Omega_0} w. \rho_0 (a-f) d \Omega

where :math:`\rho_0` is the mass density in the reference configuration.
the following variational formulation of the structural mechanics
problem posed in :math:`\Omega_0`

.. math:: \int_{\Omega_0} w. \rho_0 a d \Omega  + \int_{\Omega_0} \nabla_X w : P d\Omega  - \int_{\Omega_0} w.\rho_0 f d \Omega - \int_{(\Gamma_0)_h} w. \hat{h} d\Gamma = 0

The variation of strain :math:`\delta E` is expressed as

.. math:: \delta E = \frac{1}{2} (F^T \nabla_X w + \nabla_x w^T F)

:math:`\nabla_X` denotes the gradient taken with respect to the spatial
coordinates of the reference configuration. Due to the symmetry of
:math:`S`, the scalar product :math:`\delta E :S` simplifies to

.. math:: \delta E : S = \nabla_x w : P

where

.. math:: P = FS

and using the Saint Venant–Kirchhoff model

.. math:: S = \lambda \times tr(E) \times I   + 2 \times \mu  E

with

.. math::

   \begin{aligned}
       \lambda &=& \frac{E \nu}{((1.0 + \nu )  (1.0 - 2.0 * \nu))} \\
       \mu &=& \frac{E}{ (2.0  (1.0 + \nu))}
   \end{aligned}

recover boundary conditions are 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \begin{aligned}
       \rho_0 (a - f ) - \nabla_X \dot P &=& 0 \text{on} \Omega_0 \\
       y_i &=&  g_i \text{on} (\Gamma_0)_{g_i} \\ 
       P\hat{n} = \hat{h} \text{on} (\Gamma_0)_{h} \\ 
   \end{aligned}

Follower pressure load

.. math:: \int_{(\Gamma_t)_h} w \dot h d\Gamma_t = - \int w \dot pn d\Gamma_t = - \int_{(\Gamma_0)_h} w \dot p J F^{-T} \hat{n} d \Gamma_0

Fluid Solve
-----------

Mesh deformation
----------------

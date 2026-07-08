Mesh Movement (ALE)
===================

PVade updates the fluid mesh as the structure deforms by using an
Arbitrary Lagrangian-Eulerian (ALE) description.

ALE Kinematics
--------------

Let :math:`d_m` be fluid mesh displacement and :math:`u_m` mesh velocity:

.. math::

	u_m = \frac{\partial d_m}{\partial t}.

The convective term in the flow equations is evaluated with relative velocity
:math:`u-u_m`, ensuring consistency between moving-grid transport and physical
advection.

Interface-Driven Motion
-----------------------

At each FSI coupling step:

1. Structural displacement increment is computed
2. Interface nodes on the fluid side receive consistent displacement
3. Interior fluid nodes are updated to maintain mesh quality
4. CFD equations are advanced on the updated mesh

This preserves kinematic continuity at the interface while allowing large-scale
panel motion in the surrounding flow domain.

Boundary Treatment
------------------

PVade distinguishes:

- Interior deforming surfaces (typically panel boundaries)
- External far-field boundaries with constrained or selective motion

Boundary groups are identified via facet tags and mapped into velocity boundary
conditions used by the fluid solver.

Numerical Notes
---------------

- Mesh displacement fields are represented in dedicated vector spaces
- Previous and current mesh states are retained for time integration
- Mesh velocity is scaled by :math:`\Delta t` during fluid-step assembly

Implementation Mapping for Mesh Movement
----------------------------------------

Main implementation is in:

- :mod:`pvade.geometry.MeshManager`
- :mod:`pvade.fluid.FlowManager`

Mesh Generation
===============

PVade uses Gmsh to build geometry-conforming meshes for coupled fluid and
structure simulations.

Pipeline Overview
-----------------

Mesh generation follows these stages:

1. Create CAD entities for the selected case geometry
2. Assign physical groups (facets and cells) for boundary and subdomain tags
3. Generate second-order mesh entities
4. Convert the Gmsh model to DOLFINx mesh objects
5. Extract fluid and structure submeshes from tagged parent cells
6. Transfer facet tags from parent mesh to each submesh

Domain Markers
--------------

A central marker dictionary maps human-readable names to integer tags. Typical
markers include:

- External boundaries: ``x_min``, ``x_max``, ``y_min``, ``y_max``, ``z_min``, ``z_max``
- Interface/surfaces: ``internal_surface`` and panel-specific facets
- Cell regions: ``fluid`` and ``structure``

These tags drive all later boundary-condition construction in both CFD and CSD.

Submesh Extraction
------------------

From the parent mesh, PVade creates:

- ``domain.fluid.msh`` from cells tagged as fluid
- ``domain.structure.msh`` from cells tagged as structure

The associated facet and cell meshtags are rebuilt on each submesh so the
solver modules can apply conditions directly using local tags.

Parallel Considerations
-----------------------

The parent mesh is distributed across MPI ranks using DOLFINx partitioners.
Marker dictionaries and fixation-point arrays are broadcast so all ranks share
a consistent interpretation of tagged entities.

Implementation Mapping for Mesh Generation
------------------------------------------

Main implementation is in:

- :mod:`pvade.geometry.MeshManager`
- case-specific geometry builders under :mod:`pvade.geometry.*.DomainCreation`

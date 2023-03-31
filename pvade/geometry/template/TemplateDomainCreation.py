import gmsh
import numpy as np


class TemplateDomainCreation:
    """This class creates the geometry used for a given example.
    Gmsh is used to create the computational domain

    """

    def __init__(self, params):
        """The class is initialised here

        Args:
            params (_type_): _description_
        """
        # Store a full copy of params on this object
        self.params = params

        # Get MPI communicators
        self.comm = self.params.comm
        self.rank = self.params.rank
        self.num_procs = self.params.num_procs

        self.x_min_marker = 1
        self.x_max_marker = 2
        self.y_min_marker = 3
        self.y_max_marker = 4
        self.z_min_marker = 5
        self.z_max_marker = 6
        self.internal_surface_marker = 7
        self.fluid_marker = 8
        self.structure_marker = 8

        # Initialize Gmsh options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        # All ranks create a Gmsh model object
        self.gmsh_model = gmsh.model()
        self.gmsh_model.add("domain")
        self.gmsh_model.setCurrent("domain")

    def build(self):
        """
            panels: This function creates the computational domain for a 3d simulation involving N panels.
            The panels are set at a distance apart, rotated at an angle theta and are elevated with a distance H from the ground.
            panels2d: This function creates the computational domain for a 2d simulation involving N panels.
            The panels are set at a distance apart, rotated at an angle theta and are elevated with a distance H from the ground.
            cylinder3d: This function creates the computational domain for a flow around a 3D cylinder.
            cylinder2d: This function creates the computational domain for a flow around a 2D cylinder.
        Returns:
            The function returns gmsh.model which contains the geometric description of the computational domain
        """
        pass

    def set_length_scales(self):
        """This function call defines the characteristic length for the mesh in locations of interst
        LcMin,LcMax,DistMin and DistMax are used to create a refined mesh in specific locations
        which results in a high fidelity mesh without using a uniform element size in the whole mesh.
        """
        if self.rank == 0:
            all_pts = self.gmsh_model.occ.getEntities(0)
            self.gmsh_model.mesh.setSize(all_pts, self.params.domain.l_char)

    def mark_surfaces(self):
        """This function call iterates over all boundaries and assigns tags for each boundary.
        The Tags are being used when appying boundaty condition.
        """
        # Loop through all surfaces to find periodic tags

        self.ndim = self.gmsh_model.get_dimension()

        # Surfaces are the entities with dimension 1 less than the mesh dimension
        # i.e., surfaces have dim=2 (facets) on a 3d mesh
        # and dim=1 (lines) on a 2d mesh
        surf_ids = self.gmsh_model.occ.getEntities(self.ndim - 1)

        self.dom_tags = {}

        for surf in surf_ids:
            tag = surf[1]

            com = self.gmsh_model.occ.getCenterOfMass(self.ndim - 1, tag)

            if np.isclose(com[0], self.params.domain.x_min):
                self.dom_tags["x_min"] = [tag]

            elif np.allclose(com[0], self.params.domain.x_max):
                self.dom_tags["x_max"] = [tag]

            elif np.allclose(com[1], self.params.domain.y_min):
                self.dom_tags["y_min"] = [tag]

            elif np.allclose(com[1], self.params.domain.y_max):
                self.dom_tags["y_max"] = [tag]

            elif self.ndim == 3 and np.allclose(com[2], self.params.domain.z_min):
                self.dom_tags["z_min"] = [tag]

            elif self.ndim == 3 and np.allclose(com[2], self.params.domain.z_max):
                self.dom_tags["z_max"] = [tag]

            else:
                if "internal_surface" in self.dom_tags:
                    self.dom_tags["internal_surface"].append(tag)
                else:
                    self.dom_tags["internal_surface"] = [tag]

        self.gmsh_model.addPhysicalGroup(self.ndim, [1], self.fluid_marker)
        self.gmsh_model.setPhysicalName(self.ndim, self.fluid_marker, "fluid")

        self.gmsh_model.addPhysicalGroup(
            self.ndim - 1, self.dom_tags["x_min"], self.x_min_marker
        )
        self.gmsh_model.setPhysicalName(self.ndim - 1, self.x_min_marker, "x_min")
        self.gmsh_model.addPhysicalGroup(
            self.ndim - 1, self.dom_tags["x_max"], self.x_max_marker
        )
        self.gmsh_model.setPhysicalName(self.ndim - 1, self.x_max_marker, "x_max")
        self.gmsh_model.addPhysicalGroup(
            self.ndim - 1, self.dom_tags["y_min"], self.y_min_marker
        )
        self.gmsh_model.setPhysicalName(self.ndim - 1, self.y_min_marker, "y_min")
        self.gmsh_model.addPhysicalGroup(
            self.ndim - 1, self.dom_tags["y_max"], self.y_max_marker
        )
        self.gmsh_model.setPhysicalName(self.ndim - 1, self.y_max_marker, "y_max")

        if self.ndim == 3:
            self.gmsh_model.addPhysicalGroup(
                self.ndim - 1, self.dom_tags["z_min"], self.z_min_marker
            )
            self.gmsh_model.setPhysicalName(self.ndim - 1, self.z_min_marker, "z_min")
            self.gmsh_model.addPhysicalGroup(
                self.ndim - 1, self.dom_tags["z_max"], self.z_max_marker
            )
            self.gmsh_model.setPhysicalName(self.ndim - 1, self.z_max_marker, "z_max")

        self.gmsh_model.addPhysicalGroup(
            self.ndim - 1,
            self.dom_tags["internal_surface"],
            self.internal_surface_marker,
        )
        self.gmsh_model.setPhysicalName(
            self.ndim - 1, self.internal_surface_marker, "internal_surface"
        )

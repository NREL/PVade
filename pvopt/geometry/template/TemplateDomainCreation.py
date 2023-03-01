from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import VectorFunctionSpace, FunctionSpace
from dolfinx.cpp import mesh as cppmesh
from mpi4py import MPI
import gmsh
import numpy as np
import os
import time


class TemplateDomainCreation:
    def __init__(self, params):

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
        pass

    def set_length_scales(self):

        if self.rank == 0:
            all_pts = self.gmsh_model.occ.getEntities(0)
            self.gmsh_model.mesh.setSize(all_pts, self.params.domain.l_char)

    def mark_surfaces(self):
        """Creates boundary tags using gmsh"""
        # Loop through all surfaces to find periodic tags
        surf_ids = self.gmsh_model.occ.getEntities(2)

        self.dom_tags = {}

        for surf in surf_ids:
            tag = surf[1]

            com = self.gmsh_model.occ.getCenterOfMass(2, tag)

            if np.isclose(com[0], self.params.domain.x_min):
                self.dom_tags["x_min"] = [tag]

            elif np.allclose(com[0], self.params.domain.x_max):
                self.dom_tags["x_max"] = [tag]

            elif np.allclose(com[1], self.params.domain.y_min):
                self.dom_tags["y_min"] = [tag]

            elif np.allclose(com[1], self.params.domain.y_max):
                self.dom_tags["y_max"] = [tag]

            elif np.allclose(com[2], self.params.domain.z_min):
                self.dom_tags["z_min"] = [tag]

            elif np.allclose(com[2], self.params.domain.z_max):
                self.dom_tags["z_max"] = [tag]

            else:
                if "internal_surface" in self.dom_tags:
                    self.dom_tags["internal_surface"].append(tag)
                else:
                    self.dom_tags["internal_surface"] = [tag]
        print(self.dom_tags)

        self.gmsh_model.addPhysicalGroup(3, [1], self.fluid_marker)
        self.gmsh_model.setPhysicalName(3, self.fluid_marker, "fluid")

        self.gmsh_model.addPhysicalGroup(2, self.dom_tags["x_min"], self.x_min_marker)
        self.gmsh_model.setPhysicalName(2, self.x_min_marker, "x_min")
        self.gmsh_model.addPhysicalGroup(2, self.dom_tags["x_max"], self.x_max_marker)
        self.gmsh_model.setPhysicalName(2, self.x_max_marker, "x_max")
        self.gmsh_model.addPhysicalGroup(2, self.dom_tags["y_min"], self.y_min_marker)
        self.gmsh_model.setPhysicalName(2, self.y_min_marker, "y_min")
        self.gmsh_model.addPhysicalGroup(2, self.dom_tags["y_max"], self.y_max_marker)
        self.gmsh_model.setPhysicalName(2, self.y_max_marker, "y_max")
        self.gmsh_model.addPhysicalGroup(2, self.dom_tags["z_min"], self.z_min_marker)
        self.gmsh_model.setPhysicalName(2, self.z_min_marker, "z_min")
        self.gmsh_model.addPhysicalGroup(2, self.dom_tags["z_max"], self.z_max_marker)
        self.gmsh_model.setPhysicalName(2, self.z_max_marker, "z_max")
        self.gmsh_model.addPhysicalGroup(
            2, self.dom_tags["internal_surface"], self.internal_surface_marker
        )
        self.gmsh_model.setPhysicalName(
            2, self.internal_surface_marker, "internal_surface"
        )

from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import VectorFunctionSpace, FunctionSpace
from dolfinx.cpp import mesh as cppmesh
from mpi4py import MPI
from pvopt.geometry.template.TemplateDomainCreation import TemplateDomainCreation

import gmsh
import numpy as np
import os
import time


class DomainCreation(TemplateDomainCreation):
    def __init__(self, params):
        """Initialize the DomainCreation object
         This initializes an object that creates the computational domain.

        Args:
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object
        """
        super().__init__(params)

    def build(self):
        """This function creates the computational domain for a flow around a 3D cylinder.

        Returns:
            The function returns gmsh.model which contains the geometric description of the computational domain
        """
        # Compute and store some useful geometric quantities
        self.x_span = self.params.domain.x_max - self.params.domain.x_min
        self.y_span = self.params.domain.y_max - self.params.domain.y_min
        self.z_span = self.params.domain.z_max - self.params.domain.z_min

        # Create the fluid domain extent
        domain = self.gmsh_model.occ.addBox(
            self.params.domain.x_min,
            self.params.domain.y_min,
            self.params.domain.z_min,
            self.x_span,
            self.y_span,
            self.z_span,
        )

        self.cyld_radius = 0.05
        height = 100.0

        cylinder = self.gmsh_model.occ.addCylinder(
            0.5, -height / 2.0, 0.2, 0.0, height, 0.0, self.cyld_radius
        )

        # Remove each panel from the overall domain
        self.gmsh_model.occ.cut([(3, domain)], [(3, cylinder)])

        self.gmsh_model.occ.synchronize()

    def set_length_scales(self):

        # Define a distance field from the cylinder
        self.gmsh_model.mesh.field.add("Distance", 1)
        self.gmsh_model.mesh.field.setNumbers(
            1, "FacesList", self.dom_tags["internal_surface"]
        )

        self.gmsh_model.mesh.field.add("Threshold", 2)
        self.gmsh_model.mesh.field.setNumber(2, "IField", 1)
        self.gmsh_model.mesh.field.setNumber(2, "LcMin", self.params.domain.l_char)
        self.gmsh_model.mesh.field.setNumber(
            2, "LcMax", 6.0 * self.params.domain.l_char
        )
        self.gmsh_model.mesh.field.setNumber(2, "DistMin", 2.0 * self.cyld_radius)
        self.gmsh_model.mesh.field.setNumber(2, "DistMax", 4.0 * self.cyld_radius)

        self.gmsh_model.mesh.field.setAsBackgroundMesh(2)

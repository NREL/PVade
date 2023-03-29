from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import VectorFunctionSpace, FunctionSpace
from dolfinx.cpp import mesh as cppmesh
from mpi4py import MPI
from pvade.geometry.template.TemplateDomainCreation import TemplateDomainCreation

import gmsh
import numpy as np
import os
import time


class DomainCreation(TemplateDomainCreation):
    def __init__(self, params):
        super().__init__(params)

    def build(self):

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
        return self.gmsh_model
    
    def _set_length_scales(self,pv_model,dom_tags):
        res_min = self.params.domain.l_char
        if self.rank == 0:
            # Define a distance field from the immersed panels
            distance = pv_model.mesh.field.add("Distance", 1)
            pv_model.mesh.field.setNumbers(distance, "FacesList", dom_tags["panel_surface"])
            
            threshold = pv_model.mesh.field.add("Threshold")
            pv_model.mesh.field.setNumber(threshold, "IField", distance)


            factor = self.params.domain.l_char
            
            self.cyld_radius = self.params.domain.cyld_radius
            resolution = factor * self.cyld_radius / 10
            pv_model.mesh.field.setNumber(threshold, "LcMin", resolution)
            pv_model.mesh.field.setNumber(threshold, "LcMax", 20 * resolution)
            pv_model.mesh.field.setNumber(threshold, "DistMin", .5 * self.cyld_radius)
            pv_model.mesh.field.setNumber(threshold, "DistMax", self.cyld_radius)

            
            # Define a distance field from the immersed panels
            zmin_dist = pv_model.mesh.field.add("Distance")
            pv_model.mesh.field.setNumbers(zmin_dist, "FacesList", dom_tags["bottom"])

            zmin_thre = pv_model.mesh.field.add("Threshold")
            pv_model.mesh.field.setNumber(zmin_thre, "IField", zmin_dist)
            pv_model.mesh.field.setNumber(zmin_thre, "LcMin", 2*resolution)
            pv_model.mesh.field.setNumber(zmin_thre, "LcMax", 5*resolution)
            pv_model.mesh.field.setNumber(zmin_thre, "DistMin", 0.1)
            pv_model.mesh.field.setNumber(zmin_thre, "DistMax", 0.5)
            
            xy_dist = pv_model.mesh.field.add("Distance")
            pv_model.mesh.field.setNumbers(xy_dist, "FacesList", dom_tags["left"])
            pv_model.mesh.field.setNumbers(xy_dist, "FacesList", dom_tags["right"])
            
            xy_thre = pv_model.mesh.field.add("Threshold")
            pv_model.mesh.field.setNumber(xy_thre, "IField", xy_dist)
            pv_model.mesh.field.setNumber(xy_thre, "LcMin", 2 * resolution)
            pv_model.mesh.field.setNumber(xy_thre, "LcMax", 5* resolution)
            pv_model.mesh.field.setNumber(xy_thre, "DistMin", 0.1)
            pv_model.mesh.field.setNumber(xy_thre, "DistMax", 0.5)


            minimum = pv_model.mesh.field.add("Min")
            pv_model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, xy_thre, zmin_thre ])
            pv_model.mesh.field.setAsBackgroundMesh(minimum)
            return pv_model
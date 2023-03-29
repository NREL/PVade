from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import VectorFunctionSpace, FunctionSpace
from dolfinx.cpp import mesh as cppmesh
from mpi4py import MPI
from pvade.geometry.template.TemplateDomainCreation import TemplateDomainCreation

import gmsh
import numpy as np
import os
import time
import ufl
import dolfinx

# import meshio


class DomainCreation(TemplateDomainCreation):
    def __init__(self, params):
        super().__init__(params)

    def build(self):

        # Compute and store some useful geometric quantities
        self.x_span = self.params.domain.x_max - self.params.domain.x_min
        self.y_span = self.params.domain.y_max - self.params.domain.y_min
        self.z_span = self.params.domain.z_max - self.params.domain.z_min
        tracker_angle_rad = np.radians(self.params.pv_array.tracker_angle)

        # Create the fluid domain extent
        domain = self.gmsh_model.occ.addBox(
            self.params.domain.x_min,
            self.params.domain.y_min,
            self.params.domain.z_min,
            self.x_span,
            self.y_span,
            self.z_span,
        )

        for panel_id in range(self.params.pv_array.num_rows):
            panel_box = self.gmsh_model.occ.addBox(
                -0.5 * self.params.pv_array.panel_length,
                0.0,
                -0.5 * self.params.pv_array.panel_thickness,
                self.params.pv_array.panel_length,
                self.params.pv_array.panel_width,
                self.params.pv_array.panel_thickness,
            )

            # Rotate the panel currently centered at (0, 0, 0)
            self.gmsh_model.occ.rotate(
                [(3, panel_box)], 0, 0, 0, 0, -1, 0, tracker_angle_rad
            )

            # Translate the panel [panel_loc, 0, elev]
            self.gmsh_model.occ.translate(
                [(3, panel_box)],
                panel_id * self.params.pv_array.spacing[0],
                0,
                self.params.pv_array.elevation,
            )

            # Remove each panel from the overall domain
            self.gmsh_model.occ.cut([(3, domain)], [(3, panel_box)])

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
            
            resolution = factor * 10*self.params.pv_array.panel_thickness/2
            half_panel = self.params.pv_array.panel_length * np.cos(self.params.pv_array.tracker_angle)
            pv_model.mesh.field.setNumber(threshold, "LcMin", resolution*0.5)
            pv_model.mesh.field.setNumber(threshold, "LcMax", 5*resolution)
            pv_model.mesh.field.setNumber(threshold, "DistMin", self.params.pv_array.spacing[0])
            pv_model.mesh.field.setNumber(threshold, "DistMax", self.params.pv_array.spacing+half_panel)


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
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import VectorFunctionSpace, FunctionSpace
from dolfinx.cpp import mesh as cppmesh
from mpi4py import MPI
from pvopt.geometry.template.TemplateDomainCreation import TemplateDomainCreation

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


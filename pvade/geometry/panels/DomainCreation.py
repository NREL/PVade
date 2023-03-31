import gmsh
import numpy as np

from pvade.geometry.template.TemplateDomainCreation import TemplateDomainCreation

class DomainCreation(TemplateDomainCreation):
    """_summary_ test

    Args:
        TemplateDomainCreation (_type_): _description_
    """

    def __init__(self, params):
        """Initialize the DomainCreation object
         This initializes an object that creates the computational domain.

        Args:
            params (:obj:`pvade.Parameters.SimParams`): A SimParams object
        """
        super().__init__(params)

    def build(self):
        """This function creates the computational domain for a 3d simulation involving N panels.
            The panels are set at a distance apart, rotated at an angle theta and are elevated with a distance H from the ground.

        Returns:
            The function returns gmsh.model which contains the geometric description of the computational domain
        """
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

    def set_length_scales(self):
        res_min = self.params.domain.l_char
        
        # Define a distance field from the immersed panels
        distance = self.gmsh_model.mesh.field.add("Distance", 1)
        self.gmsh_model.mesh.field.setNumbers(distance, "FacesList", self.dom_tags["internal_surface"])
        
        threshold = self.gmsh_model.mesh.field.add("Threshold")
        self.gmsh_model.mesh.field.setNumber(threshold, "IField", distance)


        factor = self.params.domain.l_char
        
        resolution = factor * 10*self.params.pv_array.panel_thickness/2
        half_panel = self.params.pv_array.panel_length * np.cos(self.params.pv_array.tracker_angle)
        self.gmsh_model.mesh.field.setNumber(threshold, "LcMin", resolution*0.5)
        self.gmsh_model.mesh.field.setNumber(threshold, "LcMax", 5*resolution)
        self.gmsh_model.mesh.field.setNumber(threshold, "DistMin", self.params.pv_array.spacing[0])
        self.gmsh_model.mesh.field.setNumber(threshold, "DistMax", self.params.pv_array.spacing+half_panel)


        # Define a distance field from the immersed panels
        zmin_dist = self.gmsh_model.mesh.field.add("Distance")
        self.gmsh_model.mesh.field.setNumbers(zmin_dist, "FacesList", self.dom_tags["z_min"])

        zmin_thre = self.gmsh_model.mesh.field.add("Threshold")
        self.gmsh_model.mesh.field.setNumber(zmin_thre, "IField", zmin_dist)
        self.gmsh_model.mesh.field.setNumber(zmin_thre, "LcMin", 2*resolution)
        self.gmsh_model.mesh.field.setNumber(zmin_thre, "LcMax", 5*resolution)
        self.gmsh_model.mesh.field.setNumber(zmin_thre, "DistMin", 0.1)
        self.gmsh_model.mesh.field.setNumber(zmin_thre, "DistMax", 0.5)
        
        xy_dist = self.gmsh_model.mesh.field.add("Distance")
        self.gmsh_model.mesh.field.setNumbers(xy_dist, "FacesList", self.dom_tags["x_min"])
        self.gmsh_model.mesh.field.setNumbers(xy_dist, "FacesList", self.dom_tags["x_max"])
        
        xy_thre = self.gmsh_model.mesh.field.add("Threshold")
        self.gmsh_model.mesh.field.setNumber(xy_thre, "IField", xy_dist)
        self.gmsh_model.mesh.field.setNumber(xy_thre, "LcMin", 2 * resolution)
        self.gmsh_model.mesh.field.setNumber(xy_thre, "LcMax", 5* resolution)
        self.gmsh_model.mesh.field.setNumber(xy_thre, "DistMin", 0.1)
        self.gmsh_model.mesh.field.setNumber(xy_thre, "DistMax", 0.5)


        minimum = self.gmsh_model.mesh.field.add("Min")
        self.gmsh_model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, xy_thre, zmin_thre ])
        self.gmsh_model.mesh.field.setAsBackgroundMesh(minimum)


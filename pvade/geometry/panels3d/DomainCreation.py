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

    def build(self, params):
        """This function creates the computational domain for a 3d simulation involving N panels.
            The panels are set at a distance apart, rotated at an angle theta and are elevated with a distance H from the ground.

        Returns:
            The function returns gmsh.model which contains the geometric description of the computational domain
        """
        # Compute and store some useful geometric quantities
        self.x_span = params.domain.x_max - params.domain.x_min
        self.y_span = params.domain.y_max - params.domain.y_min
        self.z_span = params.domain.z_max - params.domain.z_min
        tracker_angle_rad = np.radians(params.pv_array.tracker_angle)

        ndim = 3

        # Create the fluid domain extent
        domain_id = self.gmsh_model.occ.addBox(
            params.domain.x_min,
            params.domain.y_min,
            params.domain.z_min,
            self.x_span,
            self.y_span,
            self.z_span,
            0,
        )

        domain_tag = (ndim, domain_id)

        domain_tag_list = []
        domain_tag_list.append(domain_tag)

        panel_tag_list = []

        for k in range(params.pv_array.num_rows):
            panel_id = self.gmsh_model.occ.addBox(
                -0.5 * params.pv_array.panel_length,
                0.0,
                -0.5 * params.pv_array.panel_thickness,
                params.pv_array.panel_length,
                params.pv_array.panel_width,
                params.pv_array.panel_thickness,
            )

            panel_tag = (ndim, panel_id)
            panel_tag_list.append(panel_tag)

            # Rotate the panel currently centered at (0, 0, 0)
            self.gmsh_model.occ.rotate([panel_tag], 0, 0, 0, 0, 1, 0, tracker_angle_rad)

            # Translate the panel [panel_loc, 0, elev]
            self.gmsh_model.occ.translate(
                [panel_tag],
                k * params.pv_array.spacing[0],
                0,
                params.pv_array.elevation,
            )

        # Fragment all panels from the overall domain
        self.gmsh_model.occ.fragment(domain_tag_list, panel_tag_list)

        self.gmsh_model.occ.synchronize()

    def set_length_scales(self, params, domain_markers):
        res_min = params.domain.l_char

        # Define a distance field from the immersed panels
        distance = self.gmsh_model.mesh.field.add("Distance", 1)
        self.gmsh_model.mesh.field.setNumbers(
            distance, "FacesList", domain_markers["internal_surface"]["gmsh_tags"]
        )

        threshold = self.gmsh_model.mesh.field.add("Threshold")
        self.gmsh_model.mesh.field.setNumber(threshold, "IField", distance)

        factor = params.domain.l_char

        resolution = factor * 10 * params.pv_array.panel_thickness / 2
        half_panel = params.pv_array.panel_length * np.cos(
            params.pv_array.tracker_angle
        )
        self.gmsh_model.mesh.field.setNumber(threshold, "LcMin", resolution * 0.5)
        self.gmsh_model.mesh.field.setNumber(threshold, "LcMax", 5 * resolution)
        self.gmsh_model.mesh.field.setNumber(
            threshold, "DistMin", params.pv_array.spacing[0]
        )
        self.gmsh_model.mesh.field.setNumber(
            threshold, "DistMax", params.pv_array.spacing + half_panel
        )

        # Define a distance field from the immersed panels
        zmin_dist = self.gmsh_model.mesh.field.add("Distance")
        self.gmsh_model.mesh.field.setNumbers(
            zmin_dist, "FacesList", domain_markers["z_min"]["gmsh_tags"]
        )

        zmin_thre = self.gmsh_model.mesh.field.add("Threshold")
        self.gmsh_model.mesh.field.setNumber(zmin_thre, "IField", zmin_dist)
        self.gmsh_model.mesh.field.setNumber(zmin_thre, "LcMin", 2 * resolution)
        self.gmsh_model.mesh.field.setNumber(zmin_thre, "LcMax", 5 * resolution)
        self.gmsh_model.mesh.field.setNumber(zmin_thre, "DistMin", 0.1)
        self.gmsh_model.mesh.field.setNumber(zmin_thre, "DistMax", 0.5)

        xy_dist = self.gmsh_model.mesh.field.add("Distance")
        self.gmsh_model.mesh.field.setNumbers(
            xy_dist, "FacesList", domain_markers["x_min"]["gmsh_tags"]
        )
        self.gmsh_model.mesh.field.setNumbers(
            xy_dist, "FacesList", domain_markers["x_max"]["gmsh_tags"]
        )

        xy_thre = self.gmsh_model.mesh.field.add("Threshold")
        self.gmsh_model.mesh.field.setNumber(xy_thre, "IField", xy_dist)
        self.gmsh_model.mesh.field.setNumber(xy_thre, "LcMin", 2 * resolution)
        self.gmsh_model.mesh.field.setNumber(xy_thre, "LcMax", 5 * resolution)
        self.gmsh_model.mesh.field.setNumber(xy_thre, "DistMin", 0.1)
        self.gmsh_model.mesh.field.setNumber(xy_thre, "DistMax", 0.5)

        minimum = self.gmsh_model.mesh.field.add("Min")
        self.gmsh_model.mesh.field.setNumbers(
            minimum, "FieldsList", [threshold, xy_thre, zmin_thre]
        )
        self.gmsh_model.mesh.field.setAsBackgroundMesh(minimum)

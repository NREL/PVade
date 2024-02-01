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
        """This function creates the computational domain for a 2d simulation involving N panels.
            The panels are set at a distance apart, rotated at an angle theta and are elevated with a distance H from the ground.

        Returns:
            The function returns gmsh.model which contains the geometric description of the computational domain
        """

        # Compute and store some useful geometric quantities
        self.x_span = params.domain.x_max - params.domain.x_min
        self.y_span = params.domain.y_max - params.domain.y_min
        # self.z_span = params.domain.z_max - params.domain.z_min
        tracker_angle_rad = np.radians(params.pv_array.tracker_angle)

        # Create the fluid domain extent
        domain = self.gmsh_model.occ.addRectangle(
            params.domain.x_min,
            params.domain.y_min,
            0,  # params.domain.z_min,
            self.x_span,
            self.y_span
            # self.z_span,
        )

        for panel_id in range(params.pv_array.stream_rows):
            panel_box = self.gmsh_model.occ.addRectangle(
                -0.5 * params.pv_array.panel_chord,
                -0.5 * params.pv_array.panel_thickness,
                0,
                params.pv_array.panel_chord,
                params.pv_array.panel_thickness,
            )

            # Rotate the panel currently centered at (0, 0, 0)
            self.gmsh_model.occ.rotate(
                [(2, panel_box)], 0, 0, 0, 0, 0, 1, tracker_angle_rad
            )

            # Translate the panel [panel_loc, 0, elev]
            self.gmsh_model.occ.translate(
                [(2, panel_box)],
                panel_id * params.pv_array.stream_spacing[0],
                params.pv_array.elevation,
                0,
            )

            # Remove each panel from the overall domain
            self.gmsh_model.occ.cut([(2, domain)], [(2, panel_box)])

        self.gmsh_model.occ.synchronize()

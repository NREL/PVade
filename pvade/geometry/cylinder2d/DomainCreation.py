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
        """This function creates the computational domain for a flow around a 2D cylinder.

        Returns:
            The function returns gmsh.model which contains the geometric description of the computational domain
        """

        # All ranks create a Gmsh model object
        c_x = c_y = 0.2
        r = self.params.domain.cyld_radius

        rectangle = self.gmsh_model.occ.addRectangle(
            self.params.domain.x_min,
            self.params.domain.y_min,
            0,
            self.params.domain.x_max-self.params.domain.x_min,
            self.params.domain.y_max-self.params.domain.y_min)

        obstacle = self.gmsh_model.occ.addDisk(c_x, c_y, 0, r, r)

        self.gmsh_model.occ.cut([(2, rectangle)], [(2, obstacle)])

        self.gmsh_model.occ.synchronize()

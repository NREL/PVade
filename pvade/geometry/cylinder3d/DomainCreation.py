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
        res_min = self.params.domain.l_char
        
        # Define a distance field from the immersed panels
        distance = self.gmsh_model.mesh.field.add("Distance", 1)
        self.gmsh_model.mesh.field.setNumbers(distance, "FacesList", self.dom_tags["internal_surface"])
        
        threshold = self.gmsh_model.mesh.field.add("Threshold")
        self.gmsh_model.mesh.field.setNumber(threshold, "IField", distance)

        factor = self.params.domain.l_char
        
        self.cyld_radius = self.params.domain.cyld_radius
        resolution = factor * self.cyld_radius / 10
        self.gmsh_model.mesh.field.setNumber(threshold, "LcMin", resolution)
        self.gmsh_model.mesh.field.setNumber(threshold, "LcMax", 20 * resolution)
        self.gmsh_model.mesh.field.setNumber(threshold, "DistMin", .5 * self.cyld_radius)
        self.gmsh_model.mesh.field.setNumber(threshold, "DistMax", self.cyld_radius)

        
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


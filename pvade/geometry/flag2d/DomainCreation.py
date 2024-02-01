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

    def _add_to_domain_markers(self, marker_name, gmsh_tags, entity_type):
        # Create a dictionary to hold the gmsh tags associated with
        # x_min, x_max, y_min, y_max, z_min, z_max panel surfaces and domain walls

        if not hasattr(self, "domain_markers"):
            self.domain_markers = {}
            # Must start indexing at 1, if starting at 0, things marked "0"
            # are indistinguishable from things which receive no marking (and have default value of 0)
            self.domain_markers["_current_idx"] = 1

        assert isinstance(gmsh_tags, list)
        assert entity_type in ["cell", "facet"]

        marker_dict = {
            "idx": self.domain_markers["_current_idx"],
            "gmsh_tags": gmsh_tags,
            "entity": entity_type,
        }

        self.domain_markers[marker_name] = marker_dict
        self.domain_markers["_current_idx"] += 1

    def build_FSI(self, params):
        """This function creates the computational domain for a flow around a 2D cylinder.

        Returns:
            The function returns gmsh.model which contains the geometric description of the computational domain
        """

        self.ndim = 2
        self.numpy_pt_total_array = np.zeros((1, 6)) + np.nan

        # Add the rectangular domain spanning [x_min, x_max] and [y_min, y_max], z=0
        domain = self.gmsh_model.occ.addRectangle(
            params.domain.x_min,
            params.domain.y_min,
            0,
            params.domain.x_max - params.domain.x_min,
            params.domain.y_max - params.domain.y_min,
        )

        # Add the flag pole centered at (x, y) = (0.2, 0.2)
        c_x = 0.2
        c_y = 0.2
        radius = (
            params.pv_array.panel_span
        )  # Not a good variable name, but hijacking a definition from the standard input.yaml file
        flag_pole = self.gmsh_model.occ.addDisk(c_x, c_y, 0, radius, radius)

        # Add the flag body extending from the flag pole
        length = (
            params.pv_array.panel_chord
        )  # Not a good variable name, but hijacking a definition from the standard input.yaml file
        thickness = (
            params.pv_array.panel_thickness
        )  # Not a good variable name, but hijacking a definition from the standard input.yaml file

        # Flag starts at the center of the flagpole (will be unioned with flagpole) so total length is radius+length
        if length > 0:
            flag_body = self.gmsh_model.occ.addRectangle(
                c_x, c_y - 0.5 * thickness, 0, radius + length, thickness
            )
        else:
            flag_body = self.gmsh_model.occ.addRectangle(
                c_x, c_y - 0.5 * thickness, 0, 1e-3 + length, thickness
            )

        # self.gmsh_model.occ.cut([(2, domain)], [(2, obstacle)])
        fused_flag = self.gmsh_model.occ.fuse([(2, flag_body)], [(2, flag_pole)])
        self.gmsh_model.occ.fragment([(2, 1)], [(2, fused_flag[0][0][1])])

        self.gmsh_model.occ.synchronize()

        # Surfaces are the entities with dimension equal to the mesh dimension -1
        surf_tag_list = self.gmsh_model.occ.getEntities(self.ndim - 1)

        internal_surface = []

        for surf_tag in surf_tag_list:
            surf_id = surf_tag[1]
            com = self.gmsh_model.occ.getCenterOfMass(self.ndim - 1, surf_id)

            # sturctures tagging
            if np.isclose(com[0], params.domain.x_min):
                self._add_to_domain_markers("x_min", [surf_id], "facet")

            elif np.allclose(com[0], params.domain.x_max):
                self._add_to_domain_markers("x_max", [surf_id], "facet")

            elif np.allclose(com[1], params.domain.y_min):
                self._add_to_domain_markers("y_min", [surf_id], "facet")

            elif np.allclose(com[1], params.domain.y_max):
                self._add_to_domain_markers("y_max", [surf_id], "facet")

        self._add_to_domain_markers("left_0", [5], "facet")
        self._add_to_domain_markers("bottom_0", [6], "facet")
        self._add_to_domain_markers("right_0", [7], "facet")
        self._add_to_domain_markers("top_0", [8], "facet")

        # Tag objects as either structure or fluid
        vol_tag_list = self.gmsh_model.occ.getEntities(self.ndim)
        print(vol_tag_list)
        structure_vol_list = []
        fluid_vol_list = []

        for k, vol_tag in enumerate(vol_tag_list):
            vol_id = vol_tag[1]

            if k == 0:
                # Solid Cell
                structure_vol_list.append(vol_id)
                print("structure", vol_id)
            else:
                # Fluid Cell
                fluid_vol_list.append(vol_id)
                print("fluid", vol_id)

        self._add_to_domain_markers("structure", structure_vol_list, "cell")
        self._add_to_domain_markers("fluid", fluid_vol_list, "cell")

        # Record all the data collected to domain_markers as physics groups with physical names
        # because we are creating domain_markers _within_ the build method, we don't need to
        # call the separate, default mark_surfaces method from TemplateDomainCreation.
        # TODO: IN THE FUTURE, ALL BUILD METHODS SHOULD CREATE DOMAIN_MARKERS AND PERFORM THIS OPERATION
        # VERSUS TRYING TO EXTRACT THE DATA AFTER THE FACT.
        for key, data in self.domain_markers.items():
            if isinstance(data, dict) and "gmsh_tags" in data:
                # print(key)
                # Cells (i.e., entities of dim = msh.topology.dim)
                if data["entity"] == "cell":
                    self.gmsh_model.addPhysicalGroup(
                        self.ndim, data["gmsh_tags"], data["idx"]
                    )
                    print(f"Making {key} = {data['idx']}")
                    self.gmsh_model.setPhysicalName(self.ndim, data["idx"], key)

                # Facets (i.e., entities of dim = msh.topology.dim - 1)
                if data["entity"] == "facet":
                    self.gmsh_model.addPhysicalGroup(
                        self.ndim - 1, data["gmsh_tags"], data["idx"]
                    )
                    self.gmsh_model.setPhysicalName(self.ndim - 1, data["idx"], key)

    def set_length_scales(self, params, domain_markers):
        """This function call defines the characteristic length for the mesh in locations of interst
        LcMin,LcMax,DistMin and DistMax are used to create a refined mesh in specific locations
        which results in a high fidelity mesh without using a uniform element size in the whole mesh.
        """
        if self.rank == 0:
            all_pts = self.gmsh_model.occ.getEntities(0)
            self.gmsh_model.mesh.setSize(all_pts, params.domain.l_char)

            # Set the left-hand side (inflow) of the computational domain to use elements 2x the size of l_char
            eps = 1.0e-6
            left_edge = gmsh.model.getEntitiesInBoundingBox(
                params.domain.x_min - eps,
                params.domain.y_min - eps,
                0.0 - eps,
                params.domain.x_min + eps,
                params.domain.y_max + eps,
                0.0 + eps,
            )
            self.gmsh_model.mesh.setSize(left_edge, 2.0 * params.domain.l_char)

            # Set the left-hand side (outflow) of the computational domain to use elements 4x the size of l_char
            right_edge = gmsh.model.getEntitiesInBoundingBox(
                params.domain.x_max - eps,
                params.domain.y_min - eps,
                0.0 - eps,
                params.domain.x_max + eps,
                params.domain.y_max + eps,
                0.0 + eps,
            )
            self.gmsh_model.mesh.setSize(right_edge, 4.0 * params.domain.l_char)

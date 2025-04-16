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

    # def build(self, params):
    #     """This function creates the computational domain for a 2d simulation involving N panels.
    #         The panels are set at a distance apart, rotated at an angle theta and are elevated with a distance H from the ground.

    #     Returns:
    #         The function returns gmsh.model which contains the geometric description of the computational domain
    #     """

    #     # Compute and store some useful geometric quantities
    #     self.x_span = params.domain.x_max - params.domain.x_min
    #     self.y_span = params.domain.y_max - params.domain.y_min
    #     # self.z_span = params.domain.z_max - params.domain.z_min
    #     tracker_angle_rad = np.radians(params.pv_array.tracker_angle)

    #     # Create the fluid domain extent
    #     domain = self.gmsh_model.occ.addRectangle(
    #         params.domain.x_min,
    #         params.domain.y_min,
    #         0,  # params.domain.z_min,
    #         self.x_span,
    #         self.y_span,
    #         # self.z_span,
    #     )

    #     for panel_id in range(params.pv_array.stream_rows):
    #         panel_box = self.gmsh_model.occ.addRectangle(
    #             -0.5 * params.pv_array.panel_chord,
    #             -0.5 * params.pv_array.panel_thickness,
    #             0,
    #             params.pv_array.panel_chord,
    #             params.pv_array.panel_thickness,
    #         )

    #         # Rotate the panel currently centered at (0, 0, 0)
    #         self.gmsh_model.occ.rotate(
    #             [(2, panel_box)], 0, 0, 0, 0, 0, 1, tracker_angle_rad
    #         )

    #         # Translate the panel [panel_loc, 0, elev]
    #         self.gmsh_model.occ.translate(
    #             [(2, panel_box)],
    #             panel_id * params.pv_array.stream_spacing,
    #             params.pv_array.elevation,
    #             0,
    #         )

    #         # Remove each panel from the overall domain
    #         self.gmsh_model.occ.cut([(2, domain)], [(2, panel_box)])

    #     self.gmsh_model.occ.synchronize()
    #     self.numpy_pt_total_array = []

    def build_FSI(self, params):
        """This function creates the computational domain for a 2d simulation involving N panels.
            The panels are set at a distance apart, rotated at an angle theta and are elevated with a distance H from the ground.

        Returns:
            The function returns gmsh.model which contains the geometric description of the computational domain
        """

        # Compute and store some useful geometric quantities
        self.x_span = params.domain.x_max - params.domain.x_min
        self.y_span = params.domain.y_max - params.domain.y_min
        # self.z_span = params.domain.z_max - params.domain.z_min

        self.ndim = 2

        # Create the fluid domain extent
        domain = self.gmsh_model.occ.addRectangle(
            params.domain.x_min,
            params.domain.y_min,
            0,  # params.domain.z_min,
            self.x_span,
            self.y_span,
            # self.z_span,
        )
        domain_tag = (self.ndim, domain)
        domain_tag_list = []
        domain_tag_list.append(domain_tag)

        panel_tag_list = []
        panel_ct = 0
        prev_surf_tag = []
        for panel_id in range(params.pv_array.stream_rows):
            if isinstance(params.pv_array.tracker_angle, list):
                tracker_angle_rad = np.radians(params.pv_array.tracker_angle[panel_id])
            else:
                tracker_angle_rad = np.radians(params.pv_array.tracker_angle)

            panel_box = self.gmsh_model.occ.addRectangle(
                -0.5 * params.pv_array.panel_chord,
                -0.5 * params.pv_array.panel_thickness,
                0,
                params.pv_array.panel_chord,
                params.pv_array.panel_thickness,
            )
            panel_tag = (self.ndim, panel_box)
            panel_tag_list.append(panel_tag)

            # Calculate where the center of the shifted panel will be
            x_center = panel_id * params.pv_array.stream_spacing
            y_center = params.pv_array.elevation

            # Translate the panel to the location [x_center, y_center, 0]
            self.gmsh_model.occ.translate(
                [panel_tag],
                x_center,
                y_center,
                0,
            )

            # Get the bounding box for this panel
            panel_bounding_box = self.gmsh_model.occ.getBoundingBox(
                panel_tag[0], panel_tag[1]
            )

            # Store the x_min, y_min, x_max, y_max points
            # corresponding to the un-rotated (tracking_angle = 0) configuration
            x_min_panel = panel_bounding_box[0]
            y_min_panel = panel_bounding_box[1]
            # z_min_panel = panel_bounding_box[2] # not needed in 2d
            x_max_panel = panel_bounding_box[3]
            y_max_panel = panel_bounding_box[4]
            # z_max_panel = panel_bounding_box[5] # not needed in 2d

            self.gmsh_model.occ.synchronize()

            # Get the list of 1D surfaces (edges) that make up this panel
            surf_tags_for_this_panel = self.gmsh_model.getBoundary([panel_tag])

            for surf_tag in surf_tags_for_this_panel:
                surf_dim = surf_tag[0]
                surf_id = surf_tag[1]
                com = self.gmsh_model.occ.getCenterOfMass(surf_dim, surf_id)

                if np.isclose(com[0], x_min_panel):
                    self._add_to_domain_markers(
                        f"left_{panel_id:.0f}", [surf_id], "facet"
                    )

                elif np.isclose(com[0], x_max_panel):
                    self._add_to_domain_markers(
                        f"right_{panel_id:.0f}", [surf_id], "facet"
                    )

                elif np.isclose(com[1], y_min_panel):
                    self._add_to_domain_markers(
                        f"bottom_{panel_id:.0f}", [100], "facet"
                    )

                elif np.isclose(com[1], y_max_panel):
                    self._add_to_domain_markers(
                        f"top_{panel_id:.0f}", [surf_id], "facet"
                    )

            # Rotate the panel currently centered at
            # (x_center, y_center, 0)
            # about the z axis
            self.gmsh_model.occ.rotate(
                [(2, panel_box)], x_center, y_center, 0, 0, 0, 1, tracker_angle_rad
            )

        # Remove each panel from the overall domain, this will change the
        # numbering used in domain_tag, but NOT in the panels
        # which is why tagging them in the above loop is acceptable.
        self.gmsh_model.occ.fragment(domain_tag_list, panel_tag_list)
        self.gmsh_model.occ.synchronize()

        self.numpy_pt_total_array = np.zeros((params.pv_array.stream_rows, 6)) + np.nan

        # Surfaces are the entities with dimension equal to the mesh dimension -1
        # it is not necessary to loop over all surfaces, since the panel
        # surfaces have been tagged already, but this ensures any ordering change
        # doesn't cause problems in the future
        all_surf_tag_list = self.gmsh_model.occ.getEntities(self.ndim - 1)

        for surf_tag in all_surf_tag_list:
            surf_dim = surf_tag[0]
            surf_id = surf_tag[1]
            com = self.gmsh_model.occ.getCenterOfMass(surf_dim, surf_id)

            if np.isclose(com[0], params.domain.x_min):
                self._add_to_domain_markers("x_min", [surf_id], "facet")

            elif np.isclose(com[0], params.domain.x_max):
                self._add_to_domain_markers("x_max", [surf_id], "facet")

            elif np.isclose(com[1], params.domain.y_min):
                self._add_to_domain_markers("y_min", [surf_id], "facet")

            elif np.isclose(com[1], params.domain.y_max):
                self._add_to_domain_markers("y_max", [surf_id], "facet")

        # Tag objects as either structure or fluid
        vol_tag_list = self.gmsh_model.occ.getEntities(self.ndim)
        print("vol_tag_list = ", vol_tag_list)
        structure_vol_list = []
        fluid_vol_list = []

        for k, vol_tag in enumerate(vol_tag_list):
            vol_id = vol_tag[1]

            if k < params.pv_array.stream_rows:
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

        # # Surfaces are the entities with dimension equal to the mesh dimension
        # all_surf_tag_list = self.gmsh_model.occ.getEntities(self.ndim)
        # structure_surf_list = []
        # fluid_surf_list = []

        # for surf_tag in all_surf_tag_list:
        #     surf_id = surf_tag[1]

        #     if surf_id <= params.pv_array.stream_rows :
        #         # Solid Cell
        #         structure_surf_list.append(surf_id)
        #     else:
        #         # Fluid Cell
        #         fluid_surf_list.append(surf_id)

        # self._add_to_domain_markers("structure", structure_surf_list, "facet")
        # self._add_to_domain_markers("fluid", fluid_surf_list, "facet")

    def set_length_scales(self, params, domain_markers):
        """This function call defines the characteristic length for the mesh in locations of interst
        LcMin,LcMax,DistMin and DistMax are used to create a refined mesh in specific locations
        which results in a high fidelity mesh without using a uniform element size in the whole mesh.
        """

        all_pts = self.gmsh_model.occ.getEntities(0)
        # self.gmsh_model.mesh.setSize(all_pts, params.domain.l_char)

        outflow = self.gmsh_model.mesh.field.add("Distance")
        self.gmsh_model.mesh.field.setNumbers(
            outflow, "FacesList", domain_markers["x_max"]["gmsh_tags"]
        )

        outflow_thresh = self.gmsh_model.mesh.field.add("Threshold")
        self.gmsh_model.mesh.field.setNumber(outflow_thresh, "IField", outflow)
        self.gmsh_model.mesh.field.setNumber(
            outflow_thresh, "LcMin", 2.0 * params.domain.l_char
        )
        self.gmsh_model.mesh.field.setNumber(
            outflow_thresh, "LcMax", params.domain.l_char
        )
        self.gmsh_model.mesh.field.setNumber(outflow_thresh, "DistMin", 10.0)
        self.gmsh_model.mesh.field.setNumber(outflow_thresh, "DistMax", 20.0)

        near_panel_tags = []

        for key, val in domain_markers.items():
            if "bottom" in key or "top" in key or "left" in key or "right" in key:
                near_panel_tags.append(val["gmsh_tags"][0])

        near_panel = self.gmsh_model.mesh.field.add("Distance")
        self.gmsh_model.mesh.field.setNumbers(near_panel, "EdgesList", near_panel_tags)

        near_panel_thresh = self.gmsh_model.mesh.field.add("Threshold")
        self.gmsh_model.mesh.field.setNumber(near_panel_thresh, "IField", near_panel)
        self.gmsh_model.mesh.field.setNumber(
            near_panel_thresh, "LcMin", params.domain.l_char
        )
        self.gmsh_model.mesh.field.setNumber(
            near_panel_thresh, "LcMax", 25.0 * params.domain.l_char
        )
        self.gmsh_model.mesh.field.setNumber(near_panel_thresh, "DistMin", 10.0)
        self.gmsh_model.mesh.field.setNumber(near_panel_thresh, "DistMax", 40.0)

        minimum = self.gmsh_model.mesh.field.add("Min")
        self.gmsh_model.mesh.field.setNumbers(
            minimum, "FieldsList", [near_panel_thresh, outflow_thresh]
        )

        self.gmsh_model.mesh.field.setAsBackgroundMesh(near_panel_thresh)

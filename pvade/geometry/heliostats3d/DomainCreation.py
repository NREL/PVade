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
        """This function creates the computational domain for a 3d simulation involving N panels.
            The panels are set at a distance apart, rotated at an angle theta and are elevated with a distance H from the ground.

        Returns:
            The function returns gmsh.model which contains the geometric description of the computational domain
        """

        def Rx(theta):
            rot_matrix = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(theta), -np.sin(theta)],
                    [0.0, np.sin(theta), np.cos(theta)],
                ]
            )

            return rot_matrix

        def Ry(theta):
            rot_matrix = np.array(
                [
                    [np.cos(theta), 0.0, np.sin(theta)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(theta), 0.0, np.cos(theta)],
                ]
            )

            return rot_matrix

        def Rz(theta):
            rot_matrix = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta), np.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

            return rot_matrix

        # Compute and store some useful geometric quantities
        self.x_span = params.domain.x_max - params.domain.x_min
        self.y_span = params.domain.y_max - params.domain.y_min
        self.z_span = params.domain.z_max - params.domain.z_min

        half_chord = 0.5 * params.pv_array.panel_chord
        half_span = 0.5 * params.pv_array.panel_span
        half_thickness = 0.5 * params.pv_array.panel_thickness

        tracker_angle_rad = np.radians(params.pv_array.tracker_angle)

        array_rotation = (params.fluid.wind_direction + 90.0) % 360.0
        array_rotation_rad = np.radians(array_rotation)

        # The centroid of each panel in the x-direction (these should start at x=0)
        x_centers = np.linspace(
            0.0,
            params.pv_array.stream_spacing * (params.pv_array.stream_rows - 1),
            params.pv_array.stream_rows,
        )

        x_center_of_mass = np.mean(x_centers)

        # The centroid of each panel in the y-direction (these should be centered about y=0)
        y_centers = np.linspace(
            0.0,
            params.pv_array.span_spacing * (params.pv_array.span_rows - 1),
            params.pv_array.span_rows,
        )

        y_centers -= np.mean(y_centers)
        y_center_of_mass = np.mean(y_centers)

        self.ndim = 3

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

        domain_tag = (self.ndim, domain_id)

        domain_tag_list = []
        domain_tag_list.append(domain_tag)

        panel_tag_list = []
        panel_ct = 0

        for k, yy in enumerate(y_centers):
            for j, xx in enumerate(x_centers):
                # Create an 0-tracking-degree panel centered at (x, y, z) = (0, 0, 0)
                panel_id = self.gmsh_model.occ.addBox(
                    -half_chord,
                    -half_span,
                    -half_thickness,
                    params.pv_array.panel_chord,
                    params.pv_array.panel_span,
                    params.pv_array.panel_thickness,
                )

                panel_tag = (self.ndim, panel_id)
                panel_tag_list.append(panel_tag)

                numpy_pt_list = []
                embedded_lines_tag_list = []

                # Add a bisecting line to the bottom of the panel in the spanwise direction
                # apply 0.5 to half_span to shorten torque tube length
                pt_1 = self.gmsh_model.occ.addPoint(
                    0, -0.5 * half_span, -half_thickness
                )
                pt_2 = self.gmsh_model.occ.addPoint(0, 0.5 * half_span, -half_thickness)

                numpy_pt_list.append(
                    [
                        0,
                        -0.5 * half_span,
                        -half_thickness,
                        0,
                        0.5 * half_span,
                        -half_thickness,
                    ]
                )

                torque_tube_id = self.gmsh_model.occ.addLine(pt_1, pt_2)
                torque_tube_tag = (1, torque_tube_id)
                embedded_lines_tag_list.append(torque_tube_tag)

                # Add lines in the streamwise direction to mimic sections of panel held rigid by motor
                if params.pv_array.span_fixation_pts is not None:
                    if not isinstance(params.pv_array.span_fixation_pts, list):
                        num_fixation_pts = int(
                            np.floor(
                                params.pv_array.panel_span
                                / params.pv_array.span_fixation_pts
                            )
                        )

                        fixation_pts_list = []

                        for k in range(1, num_fixation_pts + 1):
                            next_pt = k * params.pv_array.span_fixation_pts

                            eps = 1e-9

                            if (
                                next_pt > eps
                                and next_pt < params.pv_array.panel_span - eps
                            ):
                                fixation_pts_list.append(next_pt)

                    else:
                        fixation_pts_list = params.pv_array.span_fixation_pts

                    for fp in fixation_pts_list:
                        pt_1 = self.gmsh_model.occ.addPoint(
                            -0.5 * half_chord, -half_span + fp, -half_thickness
                        )
                        pt_2 = self.gmsh_model.occ.addPoint(
                            0.5 * half_chord, -half_span + fp, -half_thickness
                        )

                        # FIXME: don't add the fixation points into the numpy tagging for now
                        numpy_pt_list.append(
                            [
                                -0.5 * half_chord,
                                -half_span + fp,
                                -half_thickness,
                                0.5 * half_chord,
                                -half_span + fp,
                                -half_thickness,
                            ]
                        )

                        fixed_pt_id = self.gmsh_model.occ.addLine(pt_1, pt_2)
                        fixed_pt_tag = (1, fixed_pt_id)

                        embedded_lines_tag_list.append(fixed_pt_tag)

                # Store the result of fragmentation, it holds all the small surfaces we need to tag
                panel_frags = self.gmsh_model.occ.fragment(
                    [panel_tag], embedded_lines_tag_list
                )

                # extract just the first entry, and remove the 3d entry in position 0
                panel_surfs = panel_frags[0]
                panel_surfs.pop(0)
                panel_surfs = [k[1] for k in panel_surfs]

                # TODO: USE THESE UNAMBIGUOUS NAMES IN A FUTURE REFACTOR
                # self._add_to_domain_markers(f"x_min_{panel_ct:.0f}", [panel_surfs[0]], "facet")
                # self._add_to_domain_markers(f"x_max_{panel_ct:.0f}", [panel_surfs[1]], "facet")
                # self._add_to_domain_markers(f"y_min_{panel_ct:.0f}", [panel_surfs[2]], "facet")
                # self._add_to_domain_markers(f"y_max_{panel_ct:.0f}", [panel_surfs[3]], "facet")
                # self._add_to_domain_markers(f"z_min_{panel_ct:.0f}", panel_surfs[4:-1], "facet")
                # self._add_to_domain_markers(f"z_max_{panel_ct:.0f}", [panel_surfs[-1]], "facet")

                self._add_to_domain_markers(
                    f"front_{panel_ct:.0f}", [panel_surfs[0]], "facet"
                )
                self._add_to_domain_markers(
                    f"back_{panel_ct:.0f}", [panel_surfs[1]], "facet"
                )
                self._add_to_domain_markers(
                    f"left_{panel_ct:.0f}", [panel_surfs[2]], "facet"
                )
                self._add_to_domain_markers(
                    f"right_{panel_ct:.0f}", [panel_surfs[3]], "facet"
                )
                self._add_to_domain_markers(
                    f"bottom_{panel_ct:.0f}", panel_surfs[4:-1], "facet"
                )
                self._add_to_domain_markers(
                    f"top_{panel_ct:.0f}", [panel_surfs[-1]], "facet"
                )

                # self._add_to_domain_markers(f"right_{panel_ct:.0f}", [panel_surfs[1]], "facet")#correct
                # self._add_to_domain_markers(f"left_{panel_ct:.0f}", [panel_surfs[2]], "facet") # should be left

                # self._add_to_domain_markers(f"top_{panel_ct:.0f}", [panel_surfs[4]], "facet")
                # self._add_to_domain_markers(f"bottom_{panel_ct:.0f}", [panel_surfs[3]], "facet")# should be bottom

                # self._add_to_domain_markers(f"back_{panel_ct:.0f}", panel_surfs[5:7], "facet")
                # self._add_to_domain_markers(f"front_{panel_ct:.0f}", [panel_surfs[-1]], "facet")# should be front

                panel_ct += 1

                # Rotate the panel by its tracking angle along the y-axis (currently centered at (0, 0, 0))
                self.gmsh_model.occ.rotate(
                    [panel_tag], 0, 0, 0, 0, 1, 0, tracker_angle_rad
                )

                numpy_pt_panel_array = np.array(numpy_pt_list)
                numpy_pt_panel_array = np.reshape(numpy_pt_panel_array, (-1, self.ndim))

                numpy_pt_panel_array = np.dot(
                    numpy_pt_panel_array, Ry(tracker_angle_rad).T
                )

                # if not hasattr(self, "numpy_pt_array"):
                #     numpy_pt_array = np.array(numpy_pt_list)
                # else:
                #     numpy_pt_array = np.vcat(numpy_pt_array, np.array(numpy_pt_list))

                # Translate the panel by (x_center, y_center, elev)
                self.gmsh_model.occ.translate(
                    [panel_tag],
                    xx,
                    yy,
                    params.pv_array.elevation,
                )

                numpy_pt_panel_array[:, 0] += xx
                numpy_pt_panel_array[:, 1] += yy
                numpy_pt_panel_array[:, 2] += params.pv_array.elevation

                # Rotate the panel about the center of the full array as a proxy for changing wind direction (x_center, y_center, 0)
                self.gmsh_model.occ.rotate(
                    [panel_tag],
                    x_center_of_mass,
                    y_center_of_mass,
                    0,
                    0,
                    0,
                    1,
                    array_rotation_rad,
                )

                numpy_pt_panel_array[:, 0] -= x_center_of_mass
                numpy_pt_panel_array[:, 1] -= y_center_of_mass

                numpy_pt_panel_array = np.dot(
                    numpy_pt_panel_array, Rz(array_rotation_rad).T
                )

                numpy_pt_panel_array[:, 0] += x_center_of_mass
                numpy_pt_panel_array[:, 1] += y_center_of_mass

                if hasattr(self, "numpy_pt_total_array"):
                    self.numpy_pt_total_array = np.vstack(
                        (self.numpy_pt_total_array, numpy_pt_panel_array)
                    )
                else:
                    self.numpy_pt_total_array = np.copy(numpy_pt_panel_array)

                # Check that this panel still exists in the confines of the domain
                bbox = self.gmsh_model.occ.get_bounding_box(panel_tag[0], panel_tag[1])

                if bbox[0] < params.domain.x_min:
                    raise ValueError(
                        f"Panel with location (x, y) = ({xx}, {yy}) extends past x_min wall."
                    )
                if bbox[1] < params.domain.y_min:
                    raise ValueError(
                        f"Panel with location (x, y) = ({xx}, {yy}) extends past y_min wall."
                    )
                if bbox[3] > params.domain.x_max:
                    raise ValueError(
                        f"Panel with location (x, y) = ({xx}, {yy}) extends past x_max wall."
                    )
                if bbox[4] > params.domain.y_max:
                    raise ValueError(
                        f"Panel with location (x, y) = ({xx}, {yy}) extends past y_max wall."
                    )

        # Fragment all panels from the overall domain
        self.gmsh_model.occ.fragment(domain_tag_list, panel_tag_list)

        self.gmsh_model.occ.synchronize()

        self.numpy_pt_total_array = np.reshape(
            self.numpy_pt_total_array, (-1, int(2 * self.ndim))
        )

        # import matplotlib.pyplot as plt
        # for k in self.numpy_pt_total_array:
        #     plt.plot([k[0], k[3]], [k[1], k[4]])
        # plt.show()

        # Surfaces are the entities with dimension equal to the mesh dimension -1
        surf_tag_list = self.gmsh_model.occ.getEntities(self.ndim - 1)

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

            elif np.allclose(com[2], params.domain.z_min):
                self._add_to_domain_markers("z_min", [surf_id], "facet")

            elif np.allclose(com[2], params.domain.z_max):
                self._add_to_domain_markers("z_max", [surf_id], "facet")

        # Volumes are the entities with dimension equal to the mesh dimension
        vol_tag_list = self.gmsh_model.occ.getEntities(self.ndim)
        structure_vol_list = []
        fluid_vol_list = []

        for vol_tag in vol_tag_list:
            vol_id = vol_tag[1]

            if vol_id <= params.pv_array.stream_rows * params.pv_array.span_rows:
                # Solid Cell
                structure_vol_list.append(vol_id)
            else:
                # Fluid Cell
                fluid_vol_list.append(vol_id)

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
                    self.gmsh_model.setPhysicalName(self.ndim, data["idx"], key)

                # Facets (i.e., entities of dim = msh.topology.dim - 1)
                if data["entity"] == "facet":
                    self.gmsh_model.addPhysicalGroup(
                        self.ndim - 1, data["gmsh_tags"], data["idx"]
                    )
                    self.gmsh_model.setPhysicalName(self.ndim - 1, data["idx"], key)

    def build_structure_old(self, params):
        """This function creates the computational domain for a 3d simulation involving N panels.
            The panels are set at a distance apart, rotated at an angle theta and are elevated with a distance H from the ground.

        Returns:
            The function returns gmsh.model which contains the geometric description of the computational domain
        """
        # # Compute and store some useful geometric quantities
        # self.x_span = params.domain.x_max - params.domain.x_min
        # self.y_span = params.domain.y_max - params.domain.y_min
        # self.z_span = params.domain.z_max - params.domain.z_min
        tracker_angle_rad = np.radians(params.pv_array.tracker_angle)

        self.ndim = 3

        # # Create the fluid domain extent
        # domain_id = self.gmsh_model.occ.addBox(
        #     params.domain.x_min,
        #     params.domain.y_min,
        #     params.domain.z_min,
        #     self.x_span,
        #     self.y_span,
        #     self.z_span,
        #     0,
        # )

        # domain_tag = (self.ndim, domain_id)

        # domain_tag_list = []
        # domain_tag_list.append(domain_tag)

        panel_tag_list = []
        for j in range(params.pv_array.span_rows):
            for k in range(params.pv_array.stream_rows):
                panel_id = self.gmsh_model.occ.addBox(
                    -0.5 * params.pv_array.panel_chord,
                    0.0,
                    -0.5 * params.pv_array.panel_thickness,
                    params.pv_array.panel_chord,
                    params.pv_array.panel_span,
                    params.pv_array.panel_thickness,
                )

                panel_tag = (self.ndim, panel_id)
                panel_tag_list.append(panel_tag)

                # Rotate the panel currently centered at (0, 0, 0)
                self.gmsh_model.occ.rotate(
                    [panel_tag], 0, 0, 0, 0, 1, 0, tracker_angle_rad
                )

                # Translate the panel [panel_loc, 0, elev]
                self.gmsh_model.occ.translate(
                    [panel_tag],
                    k * params.pv_array.stream_spacing,
                    j * params.pv_array.span_spacing,
                    params.pv_array.elevation,
                )

        # Fragment all panels from the overall domain
        # self.gmsh_model.occ.fragment(domain_tag_list, panel_tag_list)

        self.gmsh_model.occ.synchronize()

    def build_structure(self, params):
        def Rx(theta):
            rot_matrix = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(theta), -np.sin(theta)],
                    [0.0, np.sin(theta), np.cos(theta)],
                ]
            )

            return rot_matrix

        def Ry(theta):
            rot_matrix = np.array(
                [
                    [np.cos(theta), 0.0, np.sin(theta)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(theta), 0.0, np.cos(theta)],
                ]
            )

            return rot_matrix

        def Rz(theta):
            rot_matrix = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta), np.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

            return rot_matrix

        # Compute and store some useful geometric quantities
        self.x_span = params.domain.x_max - params.domain.x_min
        self.y_span = params.domain.y_max - params.domain.y_min
        self.z_span = params.domain.z_max - params.domain.z_min

        half_chord = 0.5 * params.pv_array.panel_chord
        half_span = 0.5 * params.pv_array.panel_span
        half_thickness = 0.5 * params.pv_array.panel_thickness

        tracker_angle_rad = np.radians(params.pv_array.tracker_angle)

        array_rotation = (params.fluid.wind_direction + 90.0) % 360.0
        array_rotation_rad = np.radians(array_rotation)

        # The centroid of each panel in the x-direction (these should start at x=0)
        x_centers = np.linspace(
            0.0,
            params.pv_array.stream_spacing * (params.pv_array.stream_rows - 1),
            params.pv_array.stream_rows,
        )

        x_center_of_mass = np.mean(x_centers)

        # The centroid of each panel in the y-direction (these should be centered about y=0)
        y_centers = np.linspace(
            0.0,
            params.pv_array.span_spacing * (params.pv_array.span_rows - 1),
            params.pv_array.span_rows,
        )

        y_centers -= np.mean(y_centers)
        y_center_of_mass = np.mean(y_centers)

        self.ndim = 3

        # # Create the fluid domain extent
        # domain_id = self.gmsh_model.occ.addBox(
        #     params.domain.x_min,
        #     params.domain.y_min,
        #     params.domain.z_min,
        #     self.x_span,
        #     self.y_span,
        #     self.z_span,
        #     0,
        # )

        # domain_tag = (self.ndim, domain_id)

        domain_tag_list = []
        # domain_tag_list.append(domain_tag)

        panel_tag_list = []
        panel_ct = 0

        for k, yy in enumerate(y_centers):
            for j, xx in enumerate(x_centers):
                # Create an 0-tracking-degree panel centered at (x, y, z) = (0, 0, 0)
                panel_id = self.gmsh_model.occ.addBox(
                    -half_chord,
                    -half_span,
                    -half_thickness,
                    params.pv_array.panel_chord,
                    params.pv_array.panel_span,
                    params.pv_array.panel_thickness,
                )

                panel_tag = (self.ndim, panel_id)
                panel_tag_list.append(panel_tag)

                numpy_pt_list = []
                embedded_lines_tag_list = []

                # Add a bisecting line to the bottom of the panel in the spanwise direction
                pt_1 = self.gmsh_model.occ.addPoint(0, -half_span, -half_thickness)
                pt_2 = self.gmsh_model.occ.addPoint(0, half_span, -half_thickness)

                numpy_pt_list.append(
                    [0, -half_span, -half_thickness, 0, half_span, -half_thickness]
                )

                torque_tube_id = self.gmsh_model.occ.addLine(pt_1, pt_2)
                torque_tube_tag = (1, torque_tube_id)
                embedded_lines_tag_list.append(torque_tube_tag)

                # Add lines in the streamwise direction to mimic sections of panel held rigid by motor
                if params.pv_array.span_fixation_pts is not None:
                    if not isinstance(params.pv_array.span_fixation_pts, list):
                        num_fixation_pts = int(
                            np.floor(
                                params.pv_array.panel_span
                                / params.pv_array.span_fixation_pts
                            )
                        )

                        fixation_pts_list = []

                        for k in range(1, num_fixation_pts + 1):
                            next_pt = k * params.pv_array.span_fixation_pts

                            eps = 1e-9

                            if (
                                next_pt > eps
                                and next_pt < params.pv_array.panel_span - eps
                            ):
                                fixation_pts_list.append(next_pt)

                    else:
                        fixation_pts_list = params.pv_array.span_fixation_pts

                    for fp in fixation_pts_list:
                        pt_1 = self.gmsh_model.occ.addPoint(
                            -half_chord, -half_span + fp, -half_thickness
                        )
                        pt_2 = self.gmsh_model.occ.addPoint(
                            half_chord, -half_span + fp, -half_thickness
                        )

                        # FIXME: don't add the fixation points into the numpy tagging for now
                        numpy_pt_list.append(
                            [
                                -half_chord,
                                -half_span + fp,
                                -half_thickness,
                                half_chord,
                                -half_span + fp,
                                -half_thickness,
                            ]
                        )

                        fixed_pt_id = self.gmsh_model.occ.addLine(pt_1, pt_2)
                        fixed_pt_tag = (1, fixed_pt_id)

                        embedded_lines_tag_list.append(fixed_pt_tag)

                # Store the result of fragmentation, it holds all the small surfaces we need to tag
                panel_frags = self.gmsh_model.occ.fragment(
                    [panel_tag], embedded_lines_tag_list
                )

                # extract just the first entry, and remove the 3d entry in position 0
                panel_surfs = panel_frags[0]
                panel_surfs.pop(0)
                panel_surfs = [k[1] for k in panel_surfs]

                # TODO: USE THESE UNAMBIGUOUS NAMES IN A FUTURE REFACTOR
                # self._add_to_domain_markers(f"x_min_{panel_ct:.0f}", [panel_surfs[0]], "facet")
                # self._add_to_domain_markers(f"x_max_{panel_ct:.0f}", [panel_surfs[1]], "facet")
                # self._add_to_domain_markers(f"y_min_{panel_ct:.0f}", [panel_surfs[2]], "facet")
                # self._add_to_domain_markers(f"y_max_{panel_ct:.0f}", [panel_surfs[3]], "facet")
                # self._add_to_domain_markers(f"z_min_{panel_ct:.0f}", panel_surfs[4:-1], "facet")
                # self._add_to_domain_markers(f"z_max_{panel_ct:.0f}", [panel_surfs[-1]], "facet")

                # self._add_to_domain_markers(f"front_{panel_ct:.0f}", [panel_surfs[0]], "facet")
                # self._add_to_domain_markers(f"back_{panel_ct:.0f}", [panel_surfs[1]], "facet")
                # self._add_to_domain_markers(f"left_{panel_ct:.0f}", [panel_surfs[2]], "facet")
                # self._add_to_domain_markers(f"right_{panel_ct:.0f}", [panel_surfs[3]], "facet")
                # self._add_to_domain_markers(f"bottom_{panel_ct:.0f}", panel_surfs[4:-1], "facet")
                # self._add_to_domain_markers(f"top_{panel_ct:.0f}", [panel_surfs[-1]], "facet")

                self._add_to_domain_markers(
                    f"front_{panel_ct:.0f}", [panel_surfs[1]], "facet"
                )  # should be bottom
                self._add_to_domain_markers(
                    f"back_{panel_ct:.0f}", [panel_surfs[2]], "facet"
                )

                self._add_to_domain_markers(
                    f"left_{panel_ct:.0f}", [panel_surfs[3]], "facet"
                )  # should be left
                self._add_to_domain_markers(
                    f"right_{panel_ct:.0f}", [panel_surfs[4]], "facet"
                )  # correct 4

                self._add_to_domain_markers(
                    f"bottom_{panel_ct:.0f}", panel_surfs[5:-1], "facet"
                )
                self._add_to_domain_markers(
                    f"top_{panel_ct:.0f}", [panel_surfs[-1]], "facet"
                )  # should be front

                panel_ct += 1

                # Rotate the panel by its tracking angle along the y-axis (currently centered at (0, 0, 0))
                self.gmsh_model.occ.rotate(
                    [panel_tag], 0, 0, 0, 0, 1, 0, tracker_angle_rad
                )

                numpy_pt_panel_array = np.array(numpy_pt_list)
                numpy_pt_panel_array = np.reshape(numpy_pt_panel_array, (-1, self.ndim))

                numpy_pt_panel_array = np.dot(
                    numpy_pt_panel_array, Ry(tracker_angle_rad).T
                )

                # if not hasattr(self, "numpy_pt_array"):
                #     numpy_pt_array = np.array(numpy_pt_list)
                # else:
                #     numpy_pt_array = np.vcat(numpy_pt_array, np.array(numpy_pt_list))

                # Translate the panel by (x_center, y_center, elev)
                self.gmsh_model.occ.translate(
                    [panel_tag],
                    xx,
                    yy,
                    params.pv_array.elevation,
                )

                numpy_pt_panel_array[:, 0] += xx
                numpy_pt_panel_array[:, 1] += yy
                numpy_pt_panel_array[:, 2] += params.pv_array.elevation

                # Rotate the panel about the center of the full array as a proxy for changing wind direction (x_center, y_center, 0)
                self.gmsh_model.occ.rotate(
                    [panel_tag],
                    x_center_of_mass,
                    y_center_of_mass,
                    0,
                    0,
                    0,
                    1,
                    array_rotation_rad,
                )

                numpy_pt_panel_array[:, 0] -= x_center_of_mass
                numpy_pt_panel_array[:, 1] -= y_center_of_mass

                numpy_pt_panel_array = np.dot(
                    numpy_pt_panel_array, Rz(array_rotation_rad).T
                )

                numpy_pt_panel_array[:, 0] += x_center_of_mass
                numpy_pt_panel_array[:, 1] += y_center_of_mass

                if hasattr(self, "numpy_pt_total_array"):
                    self.numpy_pt_total_array = np.vstack(
                        (self.numpy_pt_total_array, numpy_pt_panel_array)
                    )
                else:
                    self.numpy_pt_total_array = np.copy(numpy_pt_panel_array)

                # Check that this panel still exists in the confines of the domain
                bbox = self.gmsh_model.occ.get_bounding_box(panel_tag[0], panel_tag[1])

                if bbox[0] < params.domain.x_min:
                    raise ValueError(
                        f"Panel with location (x, y) = ({xx}, {yy}) extends past x_min wall."
                    )
                if bbox[1] < params.domain.y_min:
                    raise ValueError(
                        f"Panel with location (x, y) = ({xx}, {yy}) extends past y_min wall."
                    )
                if bbox[3] > params.domain.x_max:
                    raise ValueError(
                        f"Panel with location (x, y) = ({xx}, {yy}) extends past x_max wall."
                    )
                if bbox[4] > params.domain.y_max:
                    raise ValueError(
                        f"Panel with location (x, y) = ({xx}, {yy}) extends past y_max wall."
                    )

        # Fragment all panels from the overall domain
        self.gmsh_model.occ.fragment(domain_tag_list, panel_tag_list)

        self.gmsh_model.occ.synchronize()

        self.numpy_pt_total_array = np.reshape(
            self.numpy_pt_total_array, (-1, int(2 * self.ndim))
        )

        # import matplotlib.pyplot as plt
        # for k in self.numpy_pt_total_array:
        #     plt.plot([k[0], k[3]], [k[1], k[4]])
        # plt.show()

        # # Surfaces are the entities with dimension equal to the mesh dimension -1
        # surf_tag_list = self.gmsh_model.occ.getEntities(self.ndim-1)

        # for surf_tag in surf_tag_list:
        #     surf_id = surf_tag[1]
        #     com = self.gmsh_model.occ.getCenterOfMass(self.ndim-1, surf_id)

        #     #sturctures tagging
        #     if np.isclose(com[0], params.domain.x_min):
        #         self._add_to_domain_markers("x_min", [surf_id], "facet")

        #     elif np.allclose(com[0], params.domain.x_max):
        #         self._add_to_domain_markers("x_max", [surf_id], "facet")

        #     elif np.allclose(com[1], params.domain.y_min):
        #         self._add_to_domain_markers("y_min", [surf_id], "facet")

        #     elif np.allclose(com[1], params.domain.y_max):
        #         self._add_to_domain_markers("y_max", [surf_id], "facet")

        #     elif np.allclose(com[2], params.domain.z_min):
        #         self._add_to_domain_markers("z_min", [surf_id], "facet")

        #     elif np.allclose(com[2], params.domain.z_max):
        #         self._add_to_domain_markers("z_max", [surf_id], "facet")

        # Volumes are the entities with dimension equal to the mesh dimension
        vol_tag_list = self.gmsh_model.occ.getEntities(self.ndim)
        structure_vol_list = []
        # fluid_vol_list = []

        for vol_tag in vol_tag_list:
            vol_id = vol_tag[1]
            structure_vol_list.append(vol_id)
        #     vol_id = vol_tag[1]

        #     if vol_id <= params.pv_array.stream_rows * params.pv_array.span_rows:
        #         # Solid Cell

        # else:
        #     # Fluid Cell
        #     fluid_vol_list.append(vol_id)

        self._add_to_domain_markers("structure", structure_vol_list, "cell")
        # self._add_to_domain_markers("fluid", fluid_vol_list, "cell")

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
                    self.gmsh_model.setPhysicalName(self.ndim, data["idx"], key)

                # Facets (i.e., entities of dim = msh.topology.dim - 1)
                if data["entity"] == "facet":
                    self.gmsh_model.addPhysicalGroup(
                        self.ndim - 1, data["gmsh_tags"], data["idx"]
                    )
                    self.gmsh_model.setPhysicalName(self.ndim - 1, data["idx"], key)

    def set_length_scales_DEV(self, params, domain_markers):
        res_min = params.domain.l_char

        # self.gmsh_model.mesh.field.setNumbers(
        # distance, "FacesList", domain_markers["internal_surface"]["gmsh_tags"]
        # )

        # thresholds = []
        # distances= []
        internal_surface_tags = []

        for panel_id in range(params.pv_array.stream_rows * params.pv_array.span_rows):
            internal_surface_tags.append(
                domain_markers[f"bottom_{panel_id}"]["gmsh_tags"][0]
            )
            internal_surface_tags.append(
                domain_markers[f"top_{panel_id}"]["gmsh_tags"][0]
            )
            internal_surface_tags.append(
                domain_markers[f"left_{panel_id}"]["gmsh_tags"][0]
            )
            internal_surface_tags.append(
                domain_markers[f"right_{panel_id}"]["gmsh_tags"][0]
            )
            internal_surface_tags.append(
                domain_markers[f"front_{panel_id}"]["gmsh_tags"][0]
            )
            internal_surface_tags.append(
                domain_markers[f"back_{panel_id}"]["gmsh_tags"][0]
            )

        min_dis_list = []

        # Define a distance field from the immersed panels
        distance = self.gmsh_model.mesh.field.add("Distance")
        self.gmsh_model.mesh.field.setNumbers(
            distance, "FacesList", internal_surface_tags
        )

        threshold = self.gmsh_model.mesh.field.add("Threshold")
        self.gmsh_model.mesh.field.setNumber(threshold, "IField", distance)

        factor = params.domain.l_char

        resolution = factor * 10 * params.pv_array.panel_thickness / 2
        # half_panel = params.pv_array.panel_chord * np.cos(params.pv_array.tracker_angle)

        # self.gmsh_model.mesh.field.setNumber(threshold, "LcMin", resolution * 0.5)

        factor = params.domain.l_char

        # resolution = factor * 10 * params.pv_array.panel_thickness / 2
        half_panel = params.pv_array.panel_chord * np.cos(params.pv_array.tracker_angle)

        self.gmsh_model.mesh.field.setNumber(threshold, "LcMin", factor * 0.5)

        self.gmsh_model.mesh.field.setNumber(threshold, "LcMax", 2 * factor)
        self.gmsh_model.mesh.field.setNumber(
            threshold, "DistMin", params.pv_array.stream_spacing
        )
        self.gmsh_model.mesh.field.setNumber(
            threshold, "DistMax", params.pv_array.stream_spacing + half_panel
        )

        min_dis_list.append(threshold)

        if params.general.fluid_analysis == True:
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

            min_dis_list.append(xy_thre)
            min_dis_list.append(zmin_thre)

        minimum = self.gmsh_model.mesh.field.add("Min")
        self.gmsh_model.mesh.field.setNumbers(minimum, "FieldsList", min_dis_list)
        self.gmsh_model.mesh.field.setAsBackgroundMesh(minimum)

    def set_length_scales(self, params, domain_markers):
        res_min = params.domain.l_char

        # self.gmsh_model.mesh.field.setNumbers(
        # distance, "FacesList", domain_markers["internal_surface"]["gmsh_tags"]
        # )

        # thresholds = []
        # distances= []
        internal_surface_tags = []

        for panel_id in range(params.pv_array.stream_rows * params.pv_array.span_rows):
            internal_surface_tags.append(
                domain_markers[f"bottom_{panel_id}"]["gmsh_tags"][0]
            )
            internal_surface_tags.append(
                domain_markers[f"top_{panel_id}"]["gmsh_tags"][0]
            )
            internal_surface_tags.append(
                domain_markers[f"left_{panel_id}"]["gmsh_tags"][0]
            )
            internal_surface_tags.append(
                domain_markers[f"right_{panel_id}"]["gmsh_tags"][0]
            )
            internal_surface_tags.append(
                domain_markers[f"front_{panel_id}"]["gmsh_tags"][0]
            )
            internal_surface_tags.append(
                domain_markers[f"back_{panel_id}"]["gmsh_tags"][0]
            )

        min_dist = []

        # Define a distance field from the immersed panels
        distance = self.gmsh_model.mesh.field.add("Distance")
        self.gmsh_model.mesh.field.setNumbers(
            distance, "FacesList", internal_surface_tags
        )

        threshold = self.gmsh_model.mesh.field.add("Threshold")
        self.gmsh_model.mesh.field.setNumber(threshold, "IField", distance)

        factor = params.domain.l_char

        resolution = factor * 10 * params.pv_array.panel_thickness / 2
        tracker_angle_rad = np.radians(params.pv_array.tracker_angle)
        half_panel = params.pv_array.panel_chord * np.cos(tracker_angle_rad)
        self.gmsh_model.mesh.field.setNumber(threshold, "LcMin", resolution * 0.5)
        self.gmsh_model.mesh.field.setNumber(threshold, "LcMax", 3 * resolution)
        self.gmsh_model.mesh.field.setNumber(
            threshold, "DistMin", 0.3 * params.pv_array.stream_spacing
        )
        self.gmsh_model.mesh.field.setNumber(
            threshold, "DistMax", params.pv_array.stream_spacing + half_panel
        )
        min_dist.append(threshold)

        if params.general.fluid_analysis == True:
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
            min_dist.append(xy_thre)
            min_dist.append(zmin_thre)

        minimum = self.gmsh_model.mesh.field.add("Min")
        self.gmsh_model.mesh.field.setNumbers(minimum, "FieldsList", min_dist)
        self.gmsh_model.mesh.field.setAsBackgroundMesh(minimum)

import gmsh
import numpy as np


class TemplateDomainCreation:
    """This class creates the geometry used for a given example.
    Gmsh is used to create the computational domain

    """

    def __init__(self, params):
        """The class is initialised here

        Args:
            params (_type_): _description_
        """

        # Get MPI communicators
        self.comm = params.comm
        self.rank = params.rank
        self.num_procs = params.num_procs

        # Initialize Gmsh options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        # All ranks create a Gmsh model object
        self.gmsh_model = gmsh.model()
        self.gmsh_model.add("domain")
        self.gmsh_model.setCurrent("domain")

    def build(self, params):
        """
            panels: This function creates the computational domain for a 3d simulation involving N panels.
            The panels are set at a distance apart, rotated at an angle theta and are elevated with a distance H from the ground.
            panels2d: This function creates the computational domain for a 2d simulation involving N panels.
            The panels are set at a distance apart, rotated at an angle theta and are elevated with a distance H from the ground.
            cylinder3d: This function creates the computational domain for a flow around a 3D cylinder.
            cylinder2d: This function creates the computational domain for a flow around a 2D cylinder.
        Returns:
            The function returns gmsh.model which contains the geometric description of the computational domain
        """
        pass

    def set_length_scales(self, params, domain_markers):
        """This function call defines the characteristic length for the mesh in locations of interst
        LcMin,LcMax,DistMin and DistMax are used to create a refined mesh in specific locations
        which results in a high fidelity mesh without using a uniform element size in the whole mesh.
        """
        if self.rank == 0:
            all_pts = self.gmsh_model.occ.getEntities(0)
            self.gmsh_model.mesh.setSize(all_pts, params.domain.l_char)

    def mark_surfaces(self, params, domain_markers):
        """This function call iterates over all boundaries and assigns tags for each boundary.
        The Tags are being used when appying boundary condition.
        """

        self.ndim = self.gmsh_model.get_dimension()

        # Surfaces are the entities with dimension 1 less than the mesh dimension
        # i.e., surfaces have dim=2 (facets) on a 3d mesh
        # and dim=1 (lines) on a 2d mesh
        surf_tag_list = self.gmsh_model.occ.getEntities(self.ndim - 1)

        panel_id = 0
        count = 0
        for surf_tag in surf_tag_list:
            surf_id = surf_tag[1]
            com = self.gmsh_model.occ.getCenterOfMass(self.ndim - 1, surf_id)
            # print (com)

            # sturctures tagging
            num_of_panel_facets = self.ndim * 2
            domain_facets = self.ndim * 2

            if np.isclose(com[0], params.domain.x_min):
                domain_markers["x_min"]["gmsh_tags"].append(surf_id)
                # print("x_min found")

            elif np.allclose(com[0], params.domain.x_max):
                domain_markers["x_max"]["gmsh_tags"].append(surf_id)
                # print("x_max found")

            elif np.allclose(com[1], params.domain.y_min):
                domain_markers["y_min"]["gmsh_tags"].append(surf_id)
                # print("y_min found")

            elif np.allclose(com[1], params.domain.y_max):
                domain_markers["y_max"]["gmsh_tags"].append(surf_id)
                # print("y_max found")

            elif self.ndim == 3 and np.allclose(com[2], params.domain.z_min):
                domain_markers["z_min"]["gmsh_tags"].append(surf_id)
                # print("z_min found")

            elif self.ndim == 3 and np.allclose(com[2], params.domain.z_max):
                domain_markers["z_max"]["gmsh_tags"].append(surf_id)
                # print("z_max found")

            else:
                if params.general.geometry_module == "cylinder3d":
                    domain_markers[f"cylinder_side"]["gmsh_tags"].append(surf_id)
                elif (
                    params.general.geometry_module == "panels3d"
                    or params.general.geometry_module == "heliostats3d"
                ):
                    # tgging in 3d starts with all panels then moves to domain boundaries
                    # false panels start after boundary
                    if surf_id < surf_tag_list[0][1] + num_of_panel_facets * (
                        params.pv_array.stream_rows * params.pv_array.span_rows
                    ):
                        tags = np.arange(
                            start=surf_tag_list[0][1],
                            stop=surf_tag_list[5][1] + 1,
                            step=1,
                        ) + 6 * (panel_id)
                        # tags = np.array([1,2,3,4,5,6])
                        if surf_id == tags[0]:
                            domain_markers[f"bottom_{panel_id}"]["gmsh_tags"].append(
                                surf_id
                            )
                            # dom_tags[f"x_right_pannel_{panel_id}"] = [tag]
                        elif surf_id == tags[1]:
                            domain_markers[f"top_{panel_id}"]["gmsh_tags"].append(
                                surf_id
                            )
                            # dom_tags[f"x_left_pannel_{panel_id}"] = [tag]
                        elif surf_id == tags[2]:
                            domain_markers[f"left_{panel_id}"]["gmsh_tags"].append(
                                surf_id
                            )
                            # dom_tags[f"y_right_pannel_{panel_id}"] = [tag]
                        elif surf_id == tags[3]:
                            domain_markers[f"right_{panel_id}"]["gmsh_tags"].append(
                                surf_id
                            )
                            # dom_tags[f"y_left_pannel_{panel_id}"] = [tag]
                        elif surf_id == tags[4]:
                            domain_markers[f"back_{panel_id}"]["gmsh_tags"].append(
                                surf_id
                            )
                            # dom_tags[f"z_right_pannel_{panel_id}"] = [tag]
                        elif surf_id == tags[5]:
                            domain_markers[f"front_{panel_id}"]["gmsh_tags"].append(
                                surf_id
                            )
                            # dom_tags[f"z_left_pannel_{panel_id}"] = [tag]
                        count = count + 1
                        if count == num_of_panel_facets:
                            panel_id = panel_id + 1
                            count = 0
                elif params.general.geometry_module == "panels2d":
                    # tgging in 2d starts with domain boundary then moves to panels
                    if surf_id >= surf_tag_list[0][1] + num_of_panel_facets * (
                        params.pv_array.stream_rows
                    ):
                        tags = np.arange(
                            start=surf_tag_list[0][1] + domain_facets,
                            stop=surf_tag_list[num_of_panel_facets][1] + domain_facets,
                            step=1,
                        ) + 6 * (panel_id)
                        # tags = np.array([1,2,3,4,5,6])
                        if surf_id == tags[0]:
                            domain_markers[f"bottom_{panel_id}"]["gmsh_tags"].append(
                                surf_id
                            )
                            # dom_tags[f"x_right_pannel_{panel_id}"] = [tag]
                        elif surf_id == tags[1]:
                            domain_markers[f"top_{panel_id}"]["gmsh_tags"].append(
                                surf_id
                            )
                            # dom_tags[f"x_left_pannel_{panel_id}"] = [tag]
                        elif surf_id == tags[2]:
                            domain_markers[f"left_{panel_id}"]["gmsh_tags"].append(
                                surf_id
                            )
                            # dom_tags[f"y_right_pannel_{panel_id}"] = [tag]
                        elif surf_id == tags[3]:
                            domain_markers[f"right_{panel_id}"]["gmsh_tags"].append(
                                surf_id
                            )
                            # dom_tags[f"y_left_pannel_{panel_id}"] = [tag]
                        # elif surf_id == tags[4]:
                        #         domain_markers[f"back_{panel_id}"]["gmsh_tags"].append(surf_id)
                        #         # dom_tags[f"z_right_pannel_{panel_id}"] = [tag]
                        # elif surf_id == tags[5]:
                        #         domain_markers[f"front_{panel_id}"]["gmsh_tags"].append(surf_id)
                        #         # dom_tags[f"z_left_pannel_{panel_id}"] = [tag]
                        count = count + 1
                        if count == num_of_panel_facets:
                            panel_id = panel_id + 1
                            count = 0

        # Volumes are the entities with dimension equal to the mesh dimension
        vol_tag_list = self.gmsh_model.occ.getEntities(self.ndim)

        if len(vol_tag_list) > 1:
            for vol_tag in vol_tag_list:
                vol_id = vol_tag[1]

                if vol_id <= (params.pv_array.stream_rows * params.pv_array.span_rows):
                    # This is a panel volume, vol_id = [1, 2, ..., num_panels]
                    domain_markers["structure"]["gmsh_tags"].append(vol_id)
                    # domain_markers[f"panel_{vol_id-1}"]["gmsh_tags"].append(vol_id)

                else:
                    # This is the fluid around the panels, vol_id = num_panels+1
                    domain_markers["fluid"]["gmsh_tags"].append(vol_id)

        else:
            vol_tag = vol_tag_list[0]
            vol_id = vol_tag[1]
            domain_markers["fluid"]["gmsh_tags"].append(vol_id)

        for key, data in domain_markers.items():
            if len(data["gmsh_tags"]) > 0:
                # print(key)
                # Cells (i.e., entities of dim = msh.topology.dim)
                if data["entity"] == "cell":
                    self.gmsh_model.addPhysicalGroup(
                        self.ndim, data["gmsh_tags"], data["idx"]
                    )
                    self.gmsh_model.setPhysicalName(self.ndim, data["idx"], key)
                    # print(key)

                # Facets (i.e., entities of dim = msh.topology.dim - 1)
                if data["entity"] == "facet":
                    self.gmsh_model.addPhysicalGroup(
                        self.ndim - 1, data["gmsh_tags"], data["idx"]
                    )
                    self.gmsh_model.setPhysicalName(self.ndim - 1, data["idx"], key)
                    # print(key)

        return domain_markers

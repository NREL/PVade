from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import VectorFunctionSpace, FunctionSpace
from dolfinx.cpp import mesh as cppmesh

from mpi4py import MPI
import gmsh
import numpy as np
import os
import time
import ufl
import dolfinx
import meshio

# from pvopt.geometry.panels.domain_creation   import *
class FSIDomain:
    def __init__(self, params):

        # Get MPI communicators
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()

        # Store a full copy of params on this object
        self.params = params

        self.x_min_marker = 1
        self.y_min_marker = 2
        self.z_min_marker = 3
        self.x_max_marker = 4
        self.y_max_marker = 5
        self.z_max_marker = 6
        self.internal_surface_marker = 7
        self.fluid_marker = 8

    def build(self):
        test = 1
        if test == 0:
            gmsh.initialize()

            L = 2.2
            H = 0.41
            c_x = c_y = 0.2
            r = 0.05
            gdim = 2
            mesh_comm = MPI.COMM_WORLD
            model_rank = 0
            if mesh_comm.rank == model_rank:
                rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
                obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

            if mesh_comm.rank == model_rank:
                fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
                gmsh.model.occ.synchronize()
            fluid_marker = 1
            if mesh_comm.rank == model_rank:
                volumes = gmsh.model.getEntities(dim=gdim)
                assert len(volumes) == 1
                gmsh.model.addPhysicalGroup(
                    volumes[0][0], [volumes[0][1]], fluid_marker
                )
                gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

            (
                self.inlet_marker,
                self.outlet_marker,
                self.wall_marker,
                self.obstacle_marker,
            ) = (2, 3, 4, 5)
            inflow, outflow, walls, obstacle = [], [], [], []
            if mesh_comm.rank == model_rank:
                boundaries = gmsh.model.getBoundary(volumes, oriented=False)
                for boundary in boundaries:
                    center_of_mass = gmsh.model.occ.getCenterOfMass(
                        boundary[0], boundary[1]
                    )
                    if np.allclose(center_of_mass, [0, H / 2, 0]):
                        inflow.append(boundary[1])
                    elif np.allclose(center_of_mass, [L, H / 2, 0]):
                        outflow.append(boundary[1])
                    elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(
                        center_of_mass, [L / 2, 0, 0]
                    ):
                        walls.append(boundary[1])
                    else:
                        obstacle.append(boundary[1])
                gmsh.model.addPhysicalGroup(1, walls, self.wall_marker)
                gmsh.model.setPhysicalName(1, self.wall_marker, "Walls")
                gmsh.model.addPhysicalGroup(1, inflow, self.inlet_marker)
                gmsh.model.setPhysicalName(1, self.inlet_marker, "Inlet")
                gmsh.model.addPhysicalGroup(1, outflow, self.outlet_marker)
                gmsh.model.setPhysicalName(1, self.outlet_marker, "Outlet")
                gmsh.model.addPhysicalGroup(1, obstacle, self.obstacle_marker)
                gmsh.model.setPhysicalName(1, self.obstacle_marker, "Obstacle")

                # Create distance field from obstacle.
                # Add threshold of mesh sizes based on the distance field
                # LcMax -                  /--------
                #                      /
                # LcMin -o---------/
                #        |         |       |
                #       Point    DistMin DistMax
                res_min = r / 3
                if mesh_comm.rank == model_rank:
                    distance_field = gmsh.model.mesh.field.add("Distance")
                    gmsh.model.mesh.field.setNumbers(
                        distance_field, "EdgesList", obstacle
                    )
                    threshold_field = gmsh.model.mesh.field.add("Threshold")
                    gmsh.model.mesh.field.setNumber(
                        threshold_field, "IField", distance_field
                    )
                    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
                    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
                    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
                    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
                    min_field = gmsh.model.mesh.field.add("Min")
                    gmsh.model.mesh.field.setNumbers(
                        min_field, "FieldsList", [threshold_field]
                    )
                    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

                if mesh_comm.rank == model_rank:
                    gmsh.option.setNumber("Mesh.Algorithm", 8)
                    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
                    gmsh.option.setNumber("Mesh.RecombineAll", 1)
                    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
                    gmsh.model.mesh.generate(gdim)
                    gmsh.model.mesh.setOrder(2)
                    gmsh.model.mesh.optimize("Netgen")

                self.msh, self._, self.ft = gmshio.model_to_mesh(
                    gmsh.model, mesh_comm, model_rank, gdim=gdim
                )
                # Specify names for the mesh elements
                self.msh.name = "pv_domain"
                self._.name = f"{self.msh.name}_cells"
                self.ft.name = f"{self.msh.name}_facets"
                with XDMFFile(self.msh.comm, "output_mesh_name.pvd", "w") as fp:
                    fp.write_mesh(self.msh)

        else:
            self.mesh_comm = MPI.COMM_WORLD
            self.model_rank = 0
            self.gdim = 3

            # Initialize Gmsh options
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

            # All ranks create a Gmsh model object
            self.pv_model = gmsh.model()

            # Only rank 0 builds the geometry and meshes the domain
            # if self.rank == 0:
            self._construct_geometry()
            self._mark_surfaces()
            self._set_length_scales_mod()

            if self.params.fluid.periodic:
                self._enforce_periodicity()

            self._generate_mesh()

            # All ranks receive their portion of the mesh from rank 0 (like an MPI scatter)
            self.msh, self.mt, self.ft = gmshio.model_to_mesh(
                self.pv_model, self.comm, 0
            )

            # Specify names for the mesh elements
            self.msh.name = "pv_domain"
            self.mt.name = f"{self.msh.name}_cells"
            self.ft.name = f"{self.msh.name}_facets"

    def read(self):
        if self.rank == 0:
            print("Reading the mesh from file ...")
        with dolfinx.io.XDMFFile(
            MPI.COMM_WORLD, self.params.general.output_dir_mesh + "/mesh.xdmf", "r"
        ) as xdmf:
            self.msh = xdmf.read_mesh(name="Grid")

        self.msh.topology.create_connectivity(
            self.msh.topology.dim - 1, self.msh.topology.dim
        )
        with XDMFFile(
            MPI.COMM_WORLD, self.params.general.output_dir_mesh + "/mesh_mf.xdmf", "r"
        ) as infile:
            self.ft = infile.read_meshtags(self.msh, "Grid")
        if self.rank == 0:
            print("Done.")

    def _construct_geometry(self):

        self.pv_model.add("pv_domain")
        self.pv_model.setCurrent("pv_domain")

        # Compute and store some useful geometric quantities
        self.x_span = self.params.domain.x_max - self.params.domain.x_min
        self.y_span = self.params.domain.y_max - self.params.domain.y_min
        self.z_span = self.params.domain.z_max - self.params.domain.z_min
        tracker_angle_rad = np.radians(self.params.pv_array.tracker_angle)

        # Create the fluid domain extent
        domain = self.pv_model.occ.addBox(
            self.params.domain.x_min,
            self.params.domain.y_min,
            self.params.domain.z_min,
            self.x_span,
            self.y_span,
            self.z_span,
        )

        for panel_id in range(self.params.pv_array.num_rows):
            panel_box = self.pv_model.occ.addBox(
                -0.5 * self.params.pv_array.panel_length,
                0.0,
                -0.5 * self.params.pv_array.panel_thickness,
                self.params.pv_array.panel_length,
                self.params.pv_array.panel_width,
                self.params.pv_array.panel_thickness,
            )

            # Rotate the panel currently centered at (0, 0, 0)
            self.pv_model.occ.rotate(
                [(3, panel_box)], 0, 0, 0, 0, -1, 0, tracker_angle_rad
            )

            # Translate the panel [panel_loc, 0, elev]
            self.pv_model.occ.translate(
                [(3, panel_box)],
                panel_id * self.params.pv_array.spacing[0],
                0,
                self.params.pv_array.elevation,
            )

            # Remove each panel from the overall domain
            self.pv_model.occ.cut([(3, domain)], [(3, panel_box)])

        self.pv_model.occ.synchronize()

    def _construct_geometry_mod(self):

        if self.mesh_comm.rank == self.model_rank:
            # Initialize Gmsh options
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.model.add("pv_domain")
            gmsh.model.setCurrent("pv_domain")

            # Compute and store some useful geometric quantities
            self.x_span = self.params.domain.x_max - self.params.domain.x_min
            self.y_span = self.params.domain.y_max - self.params.domain.y_min
            self.z_span = self.params.domain.z_max - self.params.domain.z_min
            tracker_angle_rad = np.radians(self.params.pv_array.tracker_angle)

            # Create the fluid domain extent
            domain = gmsh.model.occ.addBox(
                self.params.domain.x_min,
                self.params.domain.y_min,
                self.params.domain.z_min,
                self.x_span,
                self.y_span,
                self.z_span,
            )

            for panel_id in range(self.params.pv_array.num_rows):
                panel_box = gmsh.model.occ.addBox(
                    -0.5 * self.params.pv_array.panel_length,
                    0.0,
                    -0.5 * self.params.pv_array.panel_thickness,
                    self.params.pv_array.panel_length,
                    self.params.pv_array.panel_width,
                    self.params.pv_array.panel_thickness,
                )

                # Rotate the panel currently centered at (0, 0, 0)
                gmsh.model.occ.rotate(
                    [(3, panel_box)], 0, 0, 0, 0, -1, 0, tracker_angle_rad
                )

                # Translate the panel [panel_loc, 0, elev]
                gmsh.model.occ.translate(
                    [(3, panel_box)],
                    panel_id * self.params.pv_array.spacing[0],
                    0,
                    self.params.pv_array.elevation,
                )

                # Remove each panel from the overall domain
                gmsh.model.occ.cut([(3, domain)], [(3, panel_box)])

            gmsh.model.occ.synchronize()

    def _mark_surfaces(self):

        # Loop through all surfaces to find periodic tags
        surf_ids = self.pv_model.occ.getEntities(2)

        self.dom_tags = {}

        for surf in surf_ids:
            tag = surf[1]

            com = self.pv_model.occ.getCenterOfMass(2, tag)

            if np.isclose(com[0], self.params.domain.x_min):
                self.dom_tags["left"] = [tag]

            elif np.allclose(com[0], self.params.domain.x_max):
                self.dom_tags["right"] = [tag]

            elif np.allclose(com[1], self.params.domain.y_min):
                self.dom_tags["front"] = [tag]

            elif np.allclose(com[1], self.params.domain.y_max):
                self.dom_tags["back"] = [tag]

            elif np.allclose(com[2], self.params.domain.z_min):
                self.dom_tags["bottom"] = [tag]

            elif np.allclose(com[2], self.params.domain.z_max):
                self.dom_tags["top"] = [tag]

            else:
                if "panel_surface" in self.dom_tags:
                    self.dom_tags["panel_surface"].append(tag)
                else:
                    self.dom_tags["panel_surface"] = [tag]

        self.pv_model.addPhysicalGroup(3, [1], 11)
        self.pv_model.setPhysicalName(3, 11, "fluid")

        self.pv_model.addPhysicalGroup(2, self.dom_tags["left"], self.x_min_marker)
        self.pv_model.setPhysicalName(2, self.x_min_marker, "left")
        self.pv_model.addPhysicalGroup(2, self.dom_tags["right"], self.x_max_marker)
        self.pv_model.setPhysicalName(2, self.x_max_marker, "right")
        self.pv_model.addPhysicalGroup(2, self.dom_tags["front"], self.y_min_marker)
        self.pv_model.setPhysicalName(2, self.y_min_marker, "front")
        self.pv_model.addPhysicalGroup(2, self.dom_tags["back"], self.y_max_marker)
        self.pv_model.setPhysicalName(2, self.y_max_marker, "back")
        self.pv_model.addPhysicalGroup(2, self.dom_tags["bottom"], self.z_min_marker)
        self.pv_model.setPhysicalName(2, self.z_min_marker, "bottom")
        self.pv_model.addPhysicalGroup(2, self.dom_tags["top"], self.z_max_marker)
        self.pv_model.setPhysicalName(2, self.z_max_marker, "top")
        self.pv_model.addPhysicalGroup(
            2, self.dom_tags["panel_surface"], self.internal_surface_marker
        )
        self.pv_model.setPhysicalName(2, self.internal_surface_marker, "panel_surface")

    def _mark_surfaces_mod(self):

        fluid_marker = 1
        if self.mesh_comm.rank == self.model_rank:
            volumes = gmsh.model.getEntities(dim=3)
            assert len(volumes) == 1
            gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
            gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

        (
            self.inlet_marker,
            self.outlet_marker,
            self.wall_z_marker,
            self.obstacle_marker,
            self.wall_y_marker,
        ) = (2, 3, 4, 5, 6)
        self.inflow, self.outflow, self.walls_z, self.walls_y, self.obstacle = (
            [],
            [],
            [],
            [],
            [],
        )
        (
            inflow_b_found,
            outflow_b_found,
            wallz_b_found,
            wally_b_found,
            obstacle_b_found,
        ) = (False, False, False, False, False)
        if self.mesh_comm.rank == self.model_rank:
            boundaries = gmsh.model.getBoundary(volumes, oriented=False)
            for boundary in boundaries:
                center_of_mass = gmsh.model.occ.getCenterOfMass(
                    boundary[0], boundary[1]
                )
                center_x = (
                    self.params.domain.x_min
                    + abs(self.params.domain.x_max - self.params.domain.x_min) / 2
                )
                center_y = (
                    self.params.domain.y_min
                    + abs(self.params.domain.y_max - self.params.domain.y_min) / 2
                )
                center_z = (
                    self.params.domain.z_min
                    + abs(self.params.domain.z_max - self.params.domain.z_min) / 2
                )
                # if np.allclose(center_of_mass, [self.params.domain.x_min, center_y, center_z]):
                #     self.inflow.append(boundary[1])
                #     inflow_b_found = True
                if np.allclose(center_of_mass[0], [self.params.domain.x_min]):
                    self.inflow.append(boundary[1])
                    inflow_b_found = True
                elif np.allclose(center_of_mass[0], [self.params.domain.x_max]):
                    self.outflow.append(boundary[1])
                    outflow_b_found = True
                elif np.allclose(
                    center_of_mass[1], [self.params.domain.y_max]
                ) or np.allclose(center_of_mass[1], [self.params.domain.y_min]):
                    self.walls_y.append(boundary[1])
                    wally_b_found = True
                elif np.allclose(
                    center_of_mass[2], [self.params.domain.z_max]
                ) or np.allclose(center_of_mass[2], [self.params.domain.z_min]):
                    self.walls_z.append(boundary[1])
                    wallz_b_found = True
                else:
                    self.obstacle.append(boundary[1])
                    obstacle_b_found = True

            if (
                inflow_b_found
                and outflow_b_found
                and wallz_b_found
                and wally_b_found
                and obstacle_b_found
            ) == True:
                print("all boundaries are found ")
            else:
                print("boundary missing")
                quit
            gmsh.model.addPhysicalGroup(1, self.walls_z, self.wall_z_marker)
            gmsh.model.addPhysicalGroup(1, self.walls_y, self.wall_y_marker)
            gmsh.model.setPhysicalName(1, self.wall_z_marker, "Walls_z")
            gmsh.model.setPhysicalName(1, self.wall_y_marker, "Walls_y")
            gmsh.model.addPhysicalGroup(1, self.inflow, self.inlet_marker)
            gmsh.model.setPhysicalName(1, self.inlet_marker, "Inlet")
            gmsh.model.addPhysicalGroup(1, self.outflow, self.outlet_marker)
            gmsh.model.setPhysicalName(1, self.outlet_marker, "Outlet")
            gmsh.model.addPhysicalGroup(1, self.obstacle, self.obstacle_marker)
            gmsh.model.setPhysicalName(1, self.obstacle_marker, "Obstacle")

    def _set_length_scales(self):

        # Set size scales for the mesh
        # eps = 0.1

        # Set the mesh size at each point on the panel
        # panel_bbox = self.pv_model.getEntitiesInBoundingBox(self.params.domain.x_min+eps, self.params.domain.y_min-eps, self.params.domain.z_min+eps,
        #                                                  self.params.domain.x_max-eps, self.params.domain.y_max+eps, self.params.domain.z_max-eps)
        # self.pv_model.mesh.setSize(panel_bbox, self.params.domain.l_char)

        # Define a distance field from the bottom of the domain
        self.pv_model.mesh.field.add("Distance", 1)
        self.pv_model.mesh.field.setNumbers(1, "FacesList", self.obstacle)
        # self.pv_model.mesh.field.setNumbers(1, "FacesList", self.dom_tags["bottom"])

        self.pv_model.mesh.field.add("Threshold", 2)
        self.pv_model.mesh.field.setNumber(2, "IField", 1)
        self.pv_model.mesh.field.setNumber(2, "LcMin", 2.0 * self.params.domain.l_char)
        # self.pv_model.mesh.field.setNumber(2, 'LcMin', self.params.domain.l_char)
        self.pv_model.mesh.field.setNumber(2, "LcMax", 6.0 * self.params.domain.l_char)
        self.pv_model.mesh.field.setNumber(2, "DistMin", 4.5)
        self.pv_model.mesh.field.setNumber(2, "DistMax", 1.0 * self.params.domain.z_max)

        # Define a distance field from the immersed panels
        self.pv_model.mesh.field.add("Distance", 3)
        self.pv_model.mesh.field.setNumbers(3, "FacesList", self.obstacle)

        self.pv_model.mesh.field.add("Threshold", 4)
        self.pv_model.mesh.field.setNumber(4, "IField", 3)
        self.pv_model.mesh.field.setNumber(4, "LcMin", 0.2 * self.params.domain.l_char)
        self.pv_model.mesh.field.setNumber(4, "LcMax", 6.0 * self.params.domain.l_char)
        self.pv_model.mesh.field.setNumber(4, "DistMin", 0.5)
        self.pv_model.mesh.field.setNumber(4, "DistMax", 0.6 * self.params.domain.z_max)

        self.pv_model.mesh.field.add("Min", 5)
        self.pv_model.mesh.field.setNumbers(5, "FieldsList", [2, 4])

        threshold_field = gmsh.model.mesh.field.add("Threshold")
        distance_field = gmsh.model.mesh.field.add("Distance")
        self.pv_model.mesh.field.setNumbers(distance_field, "EdgesList", self.obstacle)
        self.pv_model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        self.pv_model.mesh.field.setNumber(
            threshold_field, "LcMin", self.params.domain.l_char
        )
        self.pv_model.mesh.field.setNumber(
            threshold_field, "LcMax", 0.25 * self.params.domain.z_max
        )
        self.pv_model.mesh.field.setNumber(
            threshold_field, "DistMin", self.params.domain.l_char
        )
        self.pv_model.mesh.field.setNumber(
            threshold_field, "DistMax", 2 * self.params.domain.z_max
        )
        min_field = gmsh.model.mesh.field.add("Min")
        self.pv_model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
        self.pv_model.mesh.field.setAsBackgroundMesh(min_field)

        # self.pv_model.mesh.field.setAsBackgroundMesh(5)

        # self.pv_model.mesh.field.setAsBackgroundMesh(5)

    def _set_length_scales_mod(self):
        res_min = self.params.domain.l_char
        if self.mesh_comm.rank == self.model_rank:
            # Define a distance field from the bottom of the domain
            self.pv_model.mesh.field.add("Distance", 1)
            self.pv_model.mesh.field.setNumbers(1, "FacesList", self.dom_tags["bottom"])

            self.pv_model.mesh.field.add("Threshold", 2)
            self.pv_model.mesh.field.setNumber(2, "IField", 1)
            self.pv_model.mesh.field.setNumber(
                2, "LcMin", 2.0 * self.params.domain.l_char
            )
            # self.pv_model.mesh.field.setNumber(2, 'LcMin', self.params.domain.l_char)
            self.pv_model.mesh.field.setNumber(
                2, "LcMax", 6.0 * self.params.domain.l_char
            )
            self.pv_model.mesh.field.setNumber(2, "DistMin", 4.5)
            self.pv_model.mesh.field.setNumber(
                2, "DistMax", 1.0 * self.params.domain.z_max
            )

            # Define a distance field from the immersed panels
            self.pv_model.mesh.field.add("Distance", 3)
            self.pv_model.mesh.field.setNumbers(
                3, "FacesList", self.dom_tags["panel_surface"]
            )

            self.pv_model.mesh.field.add("Threshold", 4)
            self.pv_model.mesh.field.setNumber(4, "IField", 3)

            self.pv_model.mesh.field.setNumber(
                4, "LcMin", self.params.domain.l_char * 0.5
            )
            self.pv_model.mesh.field.setNumber(
                4, "LcMax", 5 * self.params.domain.l_char
            )
            self.pv_model.mesh.field.setNumber(4, "DistMin", 0.5)
            self.pv_model.mesh.field.setNumber(
                4, "DistMax", 0.6 * self.params.domain.z_max
            )

            self.pv_model.mesh.field.add("Min", 5)
            self.pv_model.mesh.field.setNumbers(5, "FieldsList", [2, 4])
            self.pv_model.mesh.field.setAsBackgroundMesh(5)

            # gmsh.model.mesh.field.add("Distance", 1)
            # gmsh.model.mesh.field.setNumbers(1, "FacesList", self.dom_tags["panel_surface"])
            # r = res_min
            # resolution = r
            # gmsh.model.mesh.field.add("Threshold", 2)
            # gmsh.model.mesh.field.setNumber(2, "IField", 1)
            # gmsh.model.mesh.field.setNumber(2, "LcMin", resolution)
            # gmsh.model.mesh.field.setNumber(2, "LcMax", 2*resolution)
            # gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5*r)
            # gmsh.model.mesh.field.setNumber(2, "DistMax", r)

            # # We add a similar threshold at the inlet to ensure fully resolved inlet flow

            # gmsh.model.mesh.field.add("Distance", 3)
            # gmsh.model.mesh.field.setNumbers(3, "FacesList", self.dom_tags['back'] )
            # gmsh.model.mesh.field.setNumbers(3, "FacesList", self.dom_tags['front'] )
            # gmsh.model.mesh.field.setNumbers(3, "FacesList", self.dom_tags['top'] )
            # gmsh.model.mesh.field.setNumbers(3, "FacesList", self.dom_tags['bottom'] )
            # gmsh.model.mesh.field.setNumbers(3, "FacesList", self.dom_tags['left'] )
            # gmsh.model.mesh.field.setNumbers(3, "FacesList", self.dom_tags['right'] )
            # gmsh.model.mesh.field.add("Threshold", 4)
            # gmsh.model.mesh.field.setNumber(4, "IField", 3)
            # gmsh.model.mesh.field.setNumber(4, "LcMin", 5*resolution)
            # gmsh.model.mesh.field.setNumber(4, "LcMax", 10*resolution)
            # gmsh.model.mesh.field.setNumber(4, "DistMin", 0.1)
            # gmsh.model.mesh.field.setNumber(4, "DistMax", 0.5)

            # # We combine these fields by using the minimum field
            # gmsh.model.mesh.field.add("Min", 5)
            # gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2, 4])
            # gmsh.model.mesh.field.setAsBackgroundMesh(5)

        if self.mesh_comm.rank == self.model_rank:
            gmsh.option.setNumber("Mesh.Algorithm", 5)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            # gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.setOrder(2)
            gmsh.model.mesh.optimize("Netgen")

        # self.msh, self._, self.ft = gmshio.model_to_mesh(gmsh.model, self.mesh_comm, self.model_rank, gdim=2)
        # Set size scales for the mesh
        # eps = 0.1

        # Set the mesh size at each point on the panel
        # panel_bbox = self.pv_model.getEntitiesInBoundingBox(self.params.domain.x_min+eps, self.params.domain.y_min-eps, self.params.domain.z_min+eps,
        #                                                  self.params.domain.x_max-eps, self.params.domain.y_max+eps, self.params.domain.z_max-eps)
        # self.pv_model.mesh.setSize(panel_bbox, self.params.domain.l_char)

        # Define a distance field from the bottom of the domain
        # self.pv_model.mesh.field.add("Distance", 1)
        # self.pv_model.mesh.field.setNumbers(1, "FacesList", self.dom_tags["bottom"])

        # self.pv_model.mesh.field.add("Threshold", 2)
        # self.pv_model.mesh.field.setNumber(2, "IField", 1)
        # self.pv_model.mesh.field.setNumber(2, "LcMin", 2.0 * self.params.domain.l_char)
        # # self.pv_model.mesh.field.setNumber(2, 'LcMin', self.params.domain.l_char)
        # self.pv_model.mesh.field.setNumber(2, "LcMax", 6.0 * self.params.domain.l_char)
        # self.pv_model.mesh.field.setNumber(2, "DistMin", 4.5)
        # self.pv_model.mesh.field.setNumber(2, "DistMax", 1.0 * self.params.domain.z_max)

        # # Define a distance field from the immersed panels
        # self.pv_model.mesh.field.add("Distance", 3)
        # self.pv_model.mesh.field.setNumbers(
        #     3, "FacesList", self.dom_tags["panel_surface"]
        # )

        # self.pv_model.mesh.field.add("Threshold", 4)
        # self.pv_model.mesh.field.setNumber(4, "IField", 3)
        # self.pv_model.mesh.field.setNumber(4, "LcMin", self.params.domain.l_char)
        # self.pv_model.mesh.field.setNumber(4, "LcMax", 6.0 * self.params.domain.l_char)
        # self.pv_model.mesh.field.setNumber(4, "DistMin", 0.5)
        # self.pv_model.mesh.field.setNumber(4, "DistMax", 0.6 * self.params.domain.z_max)

        # self.pv_model.mesh.field.add("Min", 5)
        # self.pv_model.mesh.field.setNumbers(5, "FieldsList", [2, 4])
        # self.pv_model.mesh.field.setAsBackgroundMesh(5)

        # self.pv_model.mesh.field.setAsBackgroundMesh(5)

    def _enforce_periodicity(self):

        # TODO: Make this a generic mapping depending on which walls are marked for peridic BCs
        # TODO: Copy code to enforce periodicity from old generate_and_convert_3d_meshes.py

        # Mark the front/back walls for periodic mapping
        front_back_translation = [
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            self.y_span,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
        ]

        self.pv_model.mesh.setPeriodic(
            2, self.dom_tags["back"], self.dom_tags["front"], front_back_translation
        )

        # Mark the left/right walls for periodic mapping
        left_right_translation = [
            1,
            0,
            0,
            self.x_span,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
        ]

        self.pv_model.mesh.setPeriodic(
            2, self.dom_tags["right"], self.dom_tags["left"], left_right_translation
        )

    def _generate_mesh(self):
        if self.rank == 0:
            print("Starting mesh generation... ", end="")

        # Generate the mesh
        tic = time.time()
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        # gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        self.pv_model.mesh.generate(3)
        self.pv_model.mesh.setOrder(2)
        self.pv_model.mesh.optimize("Netgen")
        self.pv_model.mesh.generate(3)
        toc = time.time()
        if self.rank == 0:
            print("Finished.")
            print(f"Total meshing time = {toc-tic:.1f} s")

    def write_mesh_file(self):
        if self.rank == 0:
            # Save the *.msh file and *.vtk file (the latter is for visualization only)
            print(
                "Writing Mesh to %s... " % (self.params.general.output_dir_mesh), end=""
            )
            gmsh.write("%s/mesh.msh" % (self.params.general.output_dir_mesh))
            gmsh.write("%s/mesh.vtk" % (self.params.general.output_dir_mesh))

            def create_mesh(mesh, clean_points, cell_type):
                cells = mesh.get_cells_type(cell_type)
                cell_data = mesh.get_cell_data("gmsh:physical", cell_type)

                out_mesh = meshio.Mesh(
                    points=clean_points,
                    cells={cell_type: cells},
                    cell_data={"name_to_read": [cell_data]},
                )
                return out_mesh

            mesh_from_file = meshio.read(
                f"{self.params.general.output_dir_mesh}/mesh.msh"
            )
            pts = mesh_from_file.points
            tetra_mesh = create_mesh(mesh_from_file, pts, "tetra")
            tri_mesh = create_mesh(mesh_from_file, pts, "triangle")

            meshio.write(f"{self.params.general.output_dir_mesh}/mesh.xdmf", tetra_mesh)
            meshio.write(
                f"{self.params.general.output_dir_mesh}/mesh_mf.xdmf", tri_mesh
            )
            print("Done.")

    def test_mesh_functionspace(self):

        P2 = ufl.VectorElement("Lagrange", self.msh.ufl_cell(), 2)
        P1 = ufl.FiniteElement("Lagrange", self.msh.ufl_cell(), 1)
        V = FunctionSpace(self.msh, P2)
        Q = FunctionSpace(self.msh, P1)

        local_rangeV = V.dofmap.index_map.local_range
        dofsV = np.arange(*local_rangeV)

        local_rangeQ = Q.dofmap.index_map.local_range
        dofsQ = np.arange(*local_rangeQ)

        self.ndim = self.msh.topology.dim

        # coords = self.mesh.coordinates()

        nodes_dim = 0
        self.msh.topology.create_connectivity(nodes_dim, 0)
        num_nodes_owned_by_proc = self.msh.topology.index_map(nodes_dim).size_local
        geometry_entitites = cppmesh.entities_to_geometry(
            self.msh,
            nodes_dim,
            np.arange(num_nodes_owned_by_proc, dtype=np.int32),
            False,
        )
        points = self.msh.geometry.x

        coords = points[:]

        print(f"Rank {self.rank} owns {num_nodes_owned_by_proc} nodes\n{coords}")

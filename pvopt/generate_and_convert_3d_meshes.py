# purpose : generate mesh using gmsh api and convert to format readable by fenics
# boundaries are defined here using markers

# mesh generated use adaptive somehow, might need more refinement outflow inflow

# input: variable from file (bypassed)
# hard coded: every variable in yaml
# output: xdmf file for mesh and cdmf for boundary

# objective: use yaml input, use adaptivity inflow out-flow, periodic vs non-periodic
# longterm terrain

# from dolfin import *
# import meshio
# import gmsh

# import numpy as np
# import os
# import time
# import sys
# import yaml


# import sys
import os
import numpy as np
import time

# try:
#     import gmsh
# except ImportError:
#     print("This demo requires gmsh to be installed")
#     sys.exit(0)
import yaml
import gmsh
from dolfinx.io import XDMFFile, gmshio

from dolfinx import *
from mpi4py import MPI


# ================================================================


def get_domain_tags(model, x_range, y_range, z_range):

    # Loop through all surfaces to find periodic tags
    surf_ids = model.occ.getEntities(2)

    xm = 0.5 * (x_range[0] + x_range[1])
    ym = 0.5 * (y_range[0] + y_range[1])
    zm = 0.5 * (z_range[0] + z_range[1])

    dom_tags = {}

    for surf in surf_ids:
        tag = surf[1]

        com = model.occ.getCenterOfMass(2, tag)

        if np.isclose(com[0], x_range[0]):
            dom_tags["left"] = [tag]

        elif np.allclose(com[0], x_range[1]):
            dom_tags["right"] = [tag]

        elif np.allclose(com[1], y_range[0]):
            dom_tags["front"] = [tag]

        elif np.allclose(com[1], y_range[1]):
            dom_tags["back"] = [tag]

        elif np.allclose(com[2], z_range[0]):
            dom_tags["bottom"] = [tag]

        elif np.allclose(com[2], z_range[1]):
            dom_tags["top"] = [tag]

        else:
            if "panel_surface" in dom_tags:
                dom_tags["panel_surface"].append(tag)
            else:
                dom_tags["panel_surface"] = [tag]
    print(dom_tags)
    return dom_tags


# ================================================================


def write_xdmf_mesh_file(msh, mt, ft, proj_dir, save_boundary_mesh=True, verbose=False):
    output_mesh_name = "%s/mesh.xdmf" % (proj_dir)
    with XDMFFile(msh.comm, output_mesh_name, "w") as file:
        file.write_mesh(msh)
        msh.topology.create_connectivity(2, 3)
        file.write_meshtags(
            mt, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )
        file.write_meshtags(
            ft, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )


# ================================================================
def generate_periodic_panel_gmsh(
    panel, dom, num_panel_rows, theta_deg, proj_dir, comm, lc_fac=1.0
):
    num_procs = comm.Get_size()
    rank = comm.Get_rank()
    model_rank = 0
    # print('rank',rank)

    if rank == model_rank:
        # model = generate_periodic_panel_gmsh(panel, dom, num_panel_rows, theta_deg, proj_dir, lc_fac=3.0)

        # lc_fac=3.0
        lc_fac = 3.0

        print("characteristic length set to", lc_fac)

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        # Set the file format, version 2.2
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        # Name the *.geo file
        gmsh.model.add("multi_panel_mesh")

        # Convert to radians
        theta_rad = np.radians(theta_deg)

        # Calculate some useful dimensions
        x_range = [-0.5 * dom["length"], 0.5 * dom["length"]]
        y_range = [-0.5 * dom["span"], 0.5 * dom["span"]]
        z_range = [0, dom["height"]]
        panel_loc = panel["spacing"] * np.arange(num_panel_rows)
        panel_loc = panel_loc - np.mean(panel_loc)

        # Create the computational domain
        print("Create the computational domain")
        domain = gmsh.model.occ.addBox(
            x_range[0],
            y_range[0],
            z_range[0],
            dom["length"],
            dom["span"],
            dom["height"],
        )

        for z in range(num_panel_rows):
            panel_box = gmsh.model.occ.addBox(
                -0.5 * panel["length"],
                y_range[0],
                -0.5 * panel["thickness"],
                panel["length"],
                dom["span"],
                panel["thickness"],
            )

            # Rotate the panel currently centered at (0, 0, 0)
            gmsh.model.occ.rotate([(3, panel_box)], 0, 0, 0, 0, -1, 0, theta_rad)

            # Translate the panel [panel_loc, 0, elev]
            gmsh.model.occ.translate(
                [(3, panel_box)], panel_loc[z], 0, panel["elevation"]
            )

            # Remove each panel from the overall domain
            gmsh.model.occ.cut([(3, domain)], [(3, panel_box)])

        gmsh.model.occ.synchronize()

        # Get the surface tags for the domain and panel walls
        print("Get the surface tags for the domain and panel walls")
        dom_tags = get_domain_tags(gmsh.model, x_range, y_range, z_range)

        gmsh.model.addPhysicalGroup(3, [1], 11)
        gmsh.model.setPhysicalName(3, 11, "fluid")

        gmsh.model.addPhysicalGroup(2, dom_tags["left"], 1)
        gmsh.model.setPhysicalName(2, 1, "test")
        gmsh.model.addPhysicalGroup(2, dom_tags["right"], 2)
        gmsh.model.setPhysicalName(2, 2, "right")
        gmsh.model.addPhysicalGroup(2, dom_tags["front"], 3)
        gmsh.model.setPhysicalName(2, 3, "front")
        gmsh.model.addPhysicalGroup(2, dom_tags["back"], 4)
        gmsh.model.setPhysicalName(2, 4, "back")
        gmsh.model.addPhysicalGroup(2, dom_tags["bottom"], 5)
        gmsh.model.setPhysicalName(2, 5, "bottom")
        gmsh.model.addPhysicalGroup(2, dom_tags["top"], 6)
        gmsh.model.setPhysicalName(2, 6, "top")
        gmsh.model.addPhysicalGroup(2, dom_tags["panel_surface"], 7)
        gmsh.model.setPhysicalName(2, 7, "panel_surface")

        # Mark the front/back walls for periodic mapping
        front_back_translation = [
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            dom["span"],
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
        ]

        gmsh.model.mesh.setPeriodic(
            2, dom_tags["back"], dom_tags["front"], front_back_translation
        )

        # Mark the left/right walls for periodic mapping
        left_right_translation = [
            1,
            0,
            0,
            dom["length"],
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

        gmsh.model.mesh.setPeriodic(
            2, dom_tags["right"], dom_tags["left"], left_right_translation
        )

        # Set size scales for the mesh
        lc = lc_fac * panel["thickness"]
        eps = 0.1

        # Set the mesh size at each point on the panel
        # panel_bbox = gmsh.model.getEntitiesInBoundingBox(x_range[0]+eps, y_range[0]-eps, z_range[0]+eps,
        #                                                  x_range[1]-eps, y_range[1]+eps, z_range[1]-eps)
        # gmsh.model.mesh.setSize(panel_bbox, lc)

        print("creation of the mesh")
        # Define a distance field from the bottom of the domain
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "FacesList", dom_tags["bottom"])

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", 2.0 * lc)
        # gmsh.model.mesh.field.setNumber(2, 'LcMin', lc)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 6.0 * lc)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 4.5)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 1.0 * z_range[1])

        # Define a distance field from the immersed panels
        gmsh.model.mesh.field.add("Distance", 3)
        gmsh.model.mesh.field.setNumbers(3, "FacesList", dom_tags["panel_surface"])

        gmsh.model.mesh.field.add("Threshold", 4)
        gmsh.model.mesh.field.setNumber(4, "IField", 3)
        gmsh.model.mesh.field.setNumber(4, "LcMin", lc)
        gmsh.model.mesh.field.setNumber(4, "LcMax", 6.0 * lc)
        gmsh.model.mesh.field.setNumber(4, "DistMin", 0.5)
        gmsh.model.mesh.field.setNumber(4, "DistMax", 0.6 * z_range[1])

        gmsh.model.mesh.field.add("Min", 5)
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2, 4])
        gmsh.model.mesh.field.setAsBackgroundMesh(5)

        gmsh.model.mesh.field.setAsBackgroundMesh(5)

        print("Meshing... ", end="")

        # Generate the mesh
        tic = time.time()
        gmsh.model.mesh.generate(3)
        toc = time.time()

        print("Finished.")
        print("Total Meshing Time = %.1f s" % (toc - tic))

    return gmsh


# ================================================================


def main(theta_deg=30.0, mod=None):
    comm = MPI.COMM_WORLD
    num_procs = comm.Get_size()
    rank = comm.Get_rank()

    # Set the dimensions of the panel (all in meters)
    panel = {}
    with open("sim_params.yaml") as fp:
        output_dic = yaml.load(fp, Loader=yaml.FullLoader)

    # Specify the number of panels to mesh
    num_panel_rows = output_dic["num_panel_rows"]

    # panel['spacing'] = 7.0
    panel["spacing"] = output_dic["spacing"]
    panel["elevation"] = output_dic["elevation"]
    panel["length"] = output_dic["panel_length"]
    panel["thickness"] = output_dic["thickness"]

    # Set the dimensions of the domain (all in meters)
    dom = {}
    dom["length"] = num_panel_rows * panel["spacing"]
    dom["span"] = 1.0 * panel["spacing"]
    dom["height"] = 10.0 * panel["elevation"]

    # Create a project directory for this run
    proj_dir = "periodic_3_panel_alt/theta_%04.1f" % (theta_deg)
    os.makedirs("%s" % (proj_dir), exist_ok=True)

    # earlier code generated the mesh when np = 1 and computed dofs when np ~= 1
    # with dolfinx we can use any number of np, dolfinx should generate the mesh
    # on rank 1 and pass the information to all other ranks without having to write and read the mesh

    gmsh = generate_periodic_panel_gmsh(
        panel, dom, num_panel_rows, theta_deg, proj_dir, comm, lc_fac=3.0
    )

    msh, mt, ft = gmshio.model_to_mesh(gmsh.model, comm, 0)

    # print(rank,msh)

    msh.name = "cmpt_domain"
    mt.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_facets"

    write_xdmf_mesh_file(msh, mt, ft, proj_dir)

    mesh = msh

    # V = VectorFunctionSpace(mesh, 'P', 2)
    V = fem.FunctionSpace(msh, ("Lagrange", 2))
    Q = fem.FunctionSpace(msh, ("Lagrange", 1))

    local_rangeV = V.dofmap.index_map.local_range
    dofsV = np.arange(*local_rangeV)

    local_rangeQ = Q.dofmap.index_map.local_range
    dofsQ = np.arange(*local_rangeQ)

    size = np.size(dofsQ)

    print("Finished.")
    print("Total Dofs = ", (size))
    print("Total Dofs = ", (dofsV))
    print("Total Dofs = ", (dofsQ))


# mod would be the perturbation around .1
def run_all_angles(mod=None):  # run multiple angle
    for k in range(10):
        print("Found mod", mod)

    theta_deg_list = np.array(
        [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70],
        dtype=np.float64,
    )
    if mod is not None:
        theta_deg_list += mod

    for theta_deg in theta_deg_list:
        # Specify the rotation of the panels
        print(theta_deg)

        main(theta_deg=theta_deg, mod=mod)


# run_all_angles()
# run_all_angles(mod=float(sys.argv[1]))
main()

import gmsh
import math
import numpy as np
import scipy
import time

start_time = time.time()
gmsh.initialize()

gmsh.model.add("mountain")

###########################
# read x-y-z data in
###########################
print("read x-y-z data in")

imported_data = np.loadtxt("fake_terrain.csv", delimiter=",", usecols=range(3))
imported_x = imported_data[:, 0]
imported_y = imported_data[:, 1]
imported_z = imported_data[:, 2]


# interpolate the data to z = f(x,y)
interpolate_func = scipy.interpolate.LinearNDInterpolator(
    list(zip(imported_x, imported_y)), imported_z
)  # interpolate the data to z = f(x,y)

#####################################################################
# compute the coordinates of points used to construct terrain shape
#####################################################################
print("compute the coordinates of points used to construct terrain shape")

nx = 101  # number of points along x
ny = 101  # number of points along y

x_array = np.linspace(min(imported_x), max(imported_x), nx)
y_array = np.linspace(min(imported_y), max(imported_y), ny)

x_array, y_array = np.meshgrid(
    x_array, y_array
)  # y = [[0,0...],[].....[100,100,100...]], x = [[0,...100],[0,...100],[]...[0,...100]], x[i], y[j] coordinates of points on jth row.
z_array = interpolate_func(x_array, y_array)

#############################
# create points on mountain
#############################
print("create points on mountain")

pt_tag = 1

pt_tag_on_xy_plane = []

for I in range(ny):
    for J in range(nx):
        x = x_array[I][J]
        y = y_array[I][J]
        z = z_array[I][J]
        gmsh.model.occ.addPoint(x, y, z, tag=pt_tag + I * nx + J)

        if z == 0:
            pt_tag_on_xy_plane.append(pt_tag + I * nx + J)

#############################
# create mountain surface
#############################
print("create mountain surface")

line_on_xy_plane = []

# create horizontal lines

line_hor = pt_tag + (ny - 1) * nx + nx

for I in range(ny):
    for J in range(nx - 1):
        gmsh.model.occ.addLine(
            pt_tag + I * nx + J,
            pt_tag + I * nx + J + 1,
            tag=line_hor + (nx - 1) * I + J,
        )
        if (
            pt_tag + I * nx + J in pt_tag_on_xy_plane
            and pt_tag + I * nx + J + 1 in pt_tag_on_xy_plane
        ):
            line_on_xy_plane.append(line_hor + (nx - 1) * I + J)

line_ver = line_hor + (nx - 1) * (ny - 1) + nx - 2 + 1

for J in range(nx):
    for I in range(ny - 1):
        gmsh.model.occ.addLine(
            pt_tag + I * nx + J,
            pt_tag + (I + 1) * nx + J,
            tag=line_ver + (ny - 1) * J + I,
        )
        if (
            pt_tag + I * nx + J in pt_tag_on_xy_plane
            and pt_tag + (I + 1) * nx + J in pt_tag_on_xy_plane
        ):
            line_on_xy_plane.append(line_ver + (ny - 1) * J + I)

# create the diagnal lines

line_dia = line_ver + (ny - 1) * (nx - 1) + ny - 2 + 1

for I in range(ny - 1):
    for J in range(nx - 1):
        gmsh.model.occ.addLine(
            pt_tag + I * nx + J,
            pt_tag + (I + 1) * nx + J + 1,
            tag=line_dia + (nx - 1) * I + J,
        )
        if (
            pt_tag + I * nx + J in pt_tag_on_xy_plane
            and pt_tag + (I + 1) * nx + J + 1 in pt_tag_on_xy_plane
        ):
            line_on_xy_plane.append(line_dia + (nx - 1) * I + J)

# create terrain surfaces (construct triangles)
# create line loops

cl_ter = line_dia + (nx - 1) * (ny - 2) + nx - 2 + 1

s_ter_on_xy_plane = []

for I in range(ny - 1):
    for J in range(nx - 1):
        gmsh.model.occ.addCurveLoop(
            [
                line_hor + (nx - 1) * I + J,
                line_ver + (ny - 1) * (J + 1) + I,
                -line_dia - (nx - 1) * I - J,
            ],
            tag=cl_ter + I * (2 * (nx - 1)) + 2 * J + 0,
        )
        gmsh.model.occ.addCurveLoop(
            [
                line_ver + (ny - 1) * J + I,
                line_hor + (nx - 1) * (I + 1) + J,
                -line_dia - (nx - 1) * I - J,
            ],
            tag=cl_ter + I * (2 * (nx - 1)) + 2 * J + 1,
        )

        # create terrain surfaces

s_ter = cl_ter + (ny - 2) * (2 * (nx - 1)) + 2 * (nx - 2) + 1 + 1

for I in range(ny - 1):
    for J in range(nx - 1):
        gmsh.model.occ.addPlaneSurface(
            [cl_ter + I * (2 * (nx - 1)) + 2 * J + 0],
            tag=s_ter + I * (2 * (nx - 1)) + 2 * J + 0,
        )
        if (
            line_hor + (nx - 1) * I + J in line_on_xy_plane
            and line_ver + (ny - 1) * (J + 1) + I in line_on_xy_plane
            and line_dia + (nx - 1) * I + J in line_on_xy_plane
        ):
            s_ter_on_xy_plane.append(s_ter + I * (2 * (nx - 1)) + 2 * J + 0)
        gmsh.model.occ.addPlaneSurface(
            [cl_ter + I * (2 * (nx - 1)) + 2 * J + 1],
            tag=s_ter + I * (2 * (nx - 1)) + 2 * J + 1,
        )
        if (
            line_ver + (ny - 1) * J + I in line_on_xy_plane
            and line_hor + (nx - 1) * (I + 1) + J in line_on_xy_plane
            and line_dia + (nx - 1) * I + J in line_on_xy_plane
        ):
            s_ter_on_xy_plane.append(s_ter + I * (2 * (nx - 1)) + 2 * J + 1)

##############################
# create bottome surface
##############################
print("create bottome surface")
# create line loop

bottom_list = []
bottom_list.extend(list(range(line_hor, line_hor + nx - 2 + 1)))
bottom_list.extend(
    list(
        range(
            line_ver + (ny - 1) * (nx - 1), line_ver + (ny - 1) * (nx - 1) + ny - 2 + 1
        )
    )
)
bottom_list.extend(
    list(
        range(
            -line_hor - (nx - 1) * (ny - 1) - nx + 2,
            -line_hor - (nx - 1) * (ny - 1) + 1,
        )
    )
)
bottom_list.extend(list(range(-line_ver - (ny - 2), -line_ver + 1)))

cl_bottom = s_ter + (ny - 2) * (2 * (nx - 1)) + 2 * (nx - 2) + 2
gmsh.model.occ.addCurveLoop(bottom_list, tag=cl_bottom)

s_bottom = cl_bottom + 1
gmsh.model.occ.addPlaneSurface([cl_bottom], tag=s_bottom)

# gmsh.model.occ.synchronize()
# cut the bottom surface to get rid of overlaped facets
remove_list = []
for I in s_ter_on_xy_plane:
    remove_list.append((2, I))
# gmsh.model.occ.synchronize()

aaa = gmsh.model.occ.cut(
    [(2, s_bottom)], remove_list, tag=s_bottom + 1, removeObject=True, removeTool=True
)

###############################
# create terrain volume
###############################
print("create terrain volume")

# create surface loop

sl_mountain = s_bottom + 2
surface_list = list([s_bottom + 1])
surface_list.extend(
    list(range(s_ter, s_ter + (ny - 2) * (2 * (nx - 1)) + 2 * (nx - 2) + 2))
)

for I in s_ter_on_xy_plane:
    surface_list.remove(I)

gmsh.model.occ.addSurfaceLoop(surface_list, tag=sl_mountain)

v_m = sl_mountain + 1

gmsh.model.occ.addVolume([sl_mountain], tag=v_m)


###########################
# add PV volume (flat box)
###########################

pv_domain = gmsh.model.occ.addBox(40, 40, 10, 20, 20, 1)


##########################
# create whole box domain
##########################

print("create whole box domain")

whole_domain = gmsh.model.occ.addBox(0, 0, 0, 100, 100, 20)


print("cut")
domain = gmsh.model.occ.cut([(3, whole_domain)], [(3, v_m), (3, pv_domain)])

gmsh.model.occ.synchronize()
gmsh.write("fake_terrain_geo.brep")

gmsh.option.setNumber("Mesh.MeshSizeMin", 4)
gmsh.option.setNumber("Mesh.MeshSizeMax", 4)

finish_geometry_time = time.time()

##############################
# generate mesh
##############################

print("generate mesh")

gmsh.model.mesh.generate(3)
gmsh.write("fake_terrain.vtk")

gmsh.finalize()

finish_mesh_time = time.time()

print("total running time = " + "%s seconds" % (finish_mesh_time - start_time))
print("time to create geometry = " + "%s seconds" % (finish_geometry_time - start_time))
print("time to mesh = " + "%s seconds" % (finish_mesh_time - finish_geometry_time))

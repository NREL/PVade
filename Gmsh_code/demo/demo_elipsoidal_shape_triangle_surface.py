import gmsh
import math
import numpy as np

gmsh.initialize()

gmsh.model.add("mountain")

#########################
# Define parameters
#########################

nx = 20
ny = 20

x_min = 10
x_max = 40

y_center = 30

#############################
# create points on mountain
#############################

pt_tag = 1

p_left = gmsh.model.occ.addPoint(x_min, y_center, 0, tag=pt_tag)

p_center = pt_tag + 1

for I in range(1, nx):
    for J in range(0, ny + 1):
        x = x_min + (x_max - x_min) / nx * I
        y = (
            30
            - (20**2 - 20**2 / 15**2 * (x - 25) ** 2) ** 0.5
            + 2 * (20**2 - 20**2 / 15**2 * (x - 25) ** 2) ** 0.5 / ny * J
        )
        if 1 - (x - 25) ** 2 / 15**2 - (y - 30) ** 2 / 20**2 < 0:
            z = 0
        else:
            z = 20 * (1 - (x - 25) ** 2 / 15**2 - (y - 30) ** 2 / 20**2) ** 0.5
        gmsh.model.occ.addPoint(x, y, z, tag=p_center + (ny + 1) * (I - 1) + J)

p_right = gmsh.model.occ.addPoint(x_max, y_center, 0)


#############################
# create mountain surface
#############################

# create lines along y-axis at different x locations
line_ver = p_right + 1
for I in range(1, nx):
    for J in range(0, ny):
        gmsh.model.occ.addLine(
            p_center + (ny + 1) * (I - 1) + J,
            p_center + (ny + 1) * (I - 1) + J + 1,
            tag=line_ver + ny * (I - 1) + J,
        )

# create lines at most left surface (connect p_left with p_center+0....p_center+ny)
line_left = line_ver + ny * (nx - 1 - 1) + ny
for J in range(ny + 1):
    gmsh.model.occ.addLine(p_left, p_center + J, tag=line_left + J)

# create curve loops at most left surface
cl_left = line_left + ny + 1
for J in range(ny):
    gmsh.model.occ.addCurveLoop(
        [line_left + J, line_ver + J, -line_left - J - 1], tag=cl_left + J
    )

# create most left surfaces
s_left = cl_left + ny
for J in range(ny):
    gmsh.model.occ.addPlaneSurface([cl_left + J], tag=s_left + J)

# create lines at center surfaces
line_center = s_left + ny
for I in range(1, nx - 1):
    for J in range(2 * ny + 1):
        if J % 2 == 0:
            gmsh.model.occ.addLine(
                p_center + (ny + 1) * (I - 1) + int(J / 2),
                p_center + (ny + 1) * (I) + int(J / 2),
                tag=line_center + (2 * ny + 1) * (I - 1) + J,
            )
        else:
            gmsh.model.occ.addLine(
                p_center + (ny + 1) * (I - 1) + int((J + 1) / 2),
                p_center + (ny + 1) * (I) + int((J - 1) / 2),
                tag=line_center + (2 * ny + 1) * (I - 1) + J,
            )

# create line loop at center surfaces
cl_center_left = line_center + (2 * ny + 1) * (nx - 2 - 1) + 2 * ny + 1
for I in range(1, nx - 1):
    for J in range(ny):
        gmsh.model.occ.addCurveLoop(
            [
                line_ver + ny * (I - 1) + J,
                line_center + (2 * ny + 1) * (I - 1) + 2 * J + 1,
                -(line_center + (2 * ny + 1) * (I - 1) + 2 * J),
            ],
            tag=cl_center_left + ny * (I - 1) + J,
        )


s_center_left = cl_center_left + ny * (nx - 2 - 1) + ny
for I in range(1, nx - 1):
    for J in range(ny):
        gmsh.model.occ.addPlaneSurface(
            [cl_center_left + ny * (I - 1) + J], tag=s_center_left + ny * (I - 1) + J
        )

cl_center_right = s_center_left + ny * (nx - 2 - 1) + ny
for I in range(1, nx - 1):
    for J in range(ny):
        gmsh.model.occ.addCurveLoop(
            [
                line_center + (2 * ny + 1) * (I - 1) + 2 * J + 1,
                line_ver + ny * (I) + J,
                -(line_center + (2 * ny + 1) * (I - 1) + 2 * J + 2),
            ],
            tag=cl_center_right + ny * (I - 1) + J,
        )

s_center_right = cl_center_right + ny * (nx - 2 - 1) + ny
for I in range(1, nx - 1):
    for J in range(ny):
        gmsh.model.occ.addPlaneSurface(
            [cl_center_right + ny * (I - 1) + J], tag=s_center_right + ny * (I - 1) + J
        )

# create lines at most right surface

line_right = s_center_right + ny * (nx - 2 - 1) + ny
for J in range(ny + 1):
    gmsh.model.occ.addLine(
        p_right, p_center + (ny + 1) * (nx - 2) + J, tag=line_right + J
    )

# create curve loops at most right surface

cl_right = line_right + ny + 1
for J in range(ny):
    gmsh.model.occ.addCurveLoop(
        [line_right + J, line_ver + ny * (nx - 2) + J, -line_right - J - 1],
        tag=cl_right + J,
    )

# create most right surfaces

s_right = cl_right + ny
for J in range(ny):
    gmsh.model.occ.addPlaneSurface([cl_right + J], tag=s_right + J)


# create bottom surface

bottom_list = list([line_left])
bottom_list.extend(
    list(range(line_center, line_center + (2 * ny + 1) * (nx - 2), 2 * ny + 1))
)
bottom_list.append(-line_right)
bottom_list.append(line_right + ny)
bottom_list.extend(
    list(
        range(
            -(line_center + (2 * ny + 1) * (nx - 3) + 2 * ny),
            -(line_center + 2 * ny) + 2 * ny + 1,
            2 * ny + 1,
        )
    )
)
bottom_list.append(-line_left - ny)

cl_bottom = s_right + ny
gmsh.model.occ.addCurveLoop(bottom_list, tag=cl_bottom)

s_bottom = cl_bottom + 1
gmsh.model.occ.addPlaneSurface([cl_bottom], tag=s_bottom)

# create mountain volume

sc_mountain = s_bottom + 1
surface_list = list([s_bottom])
surface_list.extend(list(range(s_left, s_left + ny)))
surface_list.extend(list(range(s_center_left, s_center_left + ny * (nx - 2 - 1) + ny)))
surface_list.extend(
    list(range(s_center_right, s_center_right + ny * (nx - 2 - 1) + ny))
)
surface_list.extend(list(range(s_right, s_right + ny)))

gmsh.model.occ.addSurfaceLoop(surface_list, tag=sc_mountain)

v_m = sc_mountain + 1
gmsh.model.occ.addVolume([sc_mountain], tag=v_m)

gmsh.model.occ.synchronize()

#####################
# create whole box domain
#####################

whole_domain = gmsh.model.occ.addBox(0, 0, 0, 50, 100, 50)

# print(v_m, whole_domain)

domain = gmsh.model.occ.cut([(3, whole_domain)], [(3, v_m)])

gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMin", 1.5)
gmsh.option.setNumber("Mesh.MeshSizeMax", 1.5)

gmsh.model.mesh.generate(3)
gmsh.write("mountain.vtk")

gmsh.finalize()

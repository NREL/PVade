//+
SetFactory("OpenCASCADE");

/* Define parameters */

nx = 20;
ny = 20;

x_min = 10;
x_max = 40;

y_center = 30;

/* create points on mountain */

p_left = newp;
Point(p_left) = {x_min, y_center, 0};

p_center = newp;

For I In {1:nx-1}
    For J In {0:ny}
        x = x_min+(x_max-x_min)/nx*I;
        y = 30-(20^2-20^2/15^2*(x-25)^2)^0.5+2*(20^2-20^2/15^2*(x-25)^2)^0.5/ny*J;
        If (1-(x-25)^2/15^2-(y-30)^2/20^2<0)
            z = 0;
        Else
            z = 20*(1-(x-25)^2/15^2-(y-30)^2/20^2)^0.5;
        EndIf
        Point(p_center+(ny+1)*(I-1)+J) = {x, y, z};
    EndFor
EndFor

p_right = newp;
Point(p_right) = {x_max, y_center, 0};

/* Create surface of mountain */

// create lines along y-axis at different x locations

line_ver = newc;

For I In {1:nx-1}
    For J In {0:ny-1}
        Line(line_ver+ny*(I-1)+J) = {p_center+(ny+1)*(I-1)+J, p_center+(ny+1)*(I-1)+J+1};
    EndFor
EndFor

// create lines at most left surface (connect p_left with p_center+0....p_center+ny)

line_left = newc;
For J In {0:ny}
    Line(line_left+J) = {p_left, p_center+J};
EndFor

// create curve loops at most left surface

cl_left = newcl;
For J In {0:ny-1}
    Curve Loop(cl_left+J) = {line_left+J, line_ver+J, -line_left-J-1};
EndFor

// create most left surfaces

s_left = news;
For J In {0:ny-1}
    Surface(s_left+J) = {cl_left+J};
EndFor

// create lines at center surfaces

line_center = newc;
For I In {1:nx-2}
    For J In {0:2*ny}
        If (J%2==0)
           Line(line_center+(2*ny+1)*(I-1)+J) = {p_center+(ny+1)*(I-1)+J/2, p_center+(ny+1)*(I)+J/2};
        Else
           Line(line_center+(2*ny+1)*(I-1)+J) = {p_center+(ny+1)*(I-1)+(J+1)/2, p_center+(ny+1)*(I)+(J-1)/2};
        EndIf
    EndFor
EndFor

// create line loop at center surfaces

cl_center_left = newcl;
For I In {1:nx-2}
    For J In {0:ny-1}
        Curve Loop(cl_center_left+ny*(I-1)+J) = {line_ver+ny*(I-1)+J, line_center+(2*ny+1)*(I-1)+2*J+1, -(line_center+(2*ny+1)*(I-1)+2*J)};
    EndFor
EndFor

s_center_left = news;
For I In {1:nx-2}
    For J In {0:ny-1}
        Surface(s_center_left+ny*(I-1)+J) = {cl_center_left+ny*(I-1)+J};
    EndFor
EndFor


cl_center_right = newcl;
For I In {1:nx-2}
    For J In {0:ny-1}
        Curve Loop(cl_center_right+ny*(I-1)+J) = {line_center+(2*ny+1)*(I-1)+2*J+1, line_ver+ny*(I)+J, -(line_center+(2*ny+1)*(I-1)+2*J+2)};
    EndFor
EndFor

s_center_right = news;
For I In {1:nx-2}
    For J In {0:ny-1}
        Surface(s_center_right+ny*(I-1)+J) = {cl_center_right+ny*(I-1)+J};
    EndFor
EndFor

// create lines at most right surface

line_right = newc;
For J In {0:ny}
    Line(line_right+J) = {p_right, p_center+(ny+1)*(nx-2)+J};
EndFor

// create curve loops at most right surface

cl_right = newcl;
For J In {0:ny-1}
    Curve Loop(cl_right+J) = {line_right+J, line_ver+ny*(nx-2)+J, -line_right-J-1};
EndFor

// create most right surfaces

s_right = news;
For J In {0:ny-1}
    Surface(s_right+J) = {cl_right+J};
EndFor

// create bottom surface
/*
Printf('%g', line_left);
Printf('%g', line_center);
Printf('%g', line_center+(2*ny+1)*(nx-3));
Printf('%g', -line_right);
Printf('%g', line_right+ny);
Printf('%g', -(line_center+(2*ny+1)*(nx-3)+2*ny));
Printf('%g', -(line_center+2*ny));
Printf('%g', -line_left-ny);
*/


cl_bottom = newcl;
Curve Loop(cl_bottom) = {line_left,  line_center:line_center+(2*ny+1)*(nx-3):2*ny+1, -line_right, line_right+ny, -(line_center+(2*ny+1)*(nx-3)+2*ny):-(line_center+2*ny):2*ny+1, -line_left-ny};

s_bottom = news;
Surface(s_bottom) = {cl_bottom};

// create mountain volume

sc_mountain = news;
Surface Loop(sc_mountain) = {s_bottom, s_left:s_left+ny-1, s_center_left:s_center_left+ny*(nx-2-1)+ny-1, s_center_right:s_center_right+ny*(nx-2-1)+ny-1, s_right:s_right+ny-1};
v_m = newv;
Volume(v_m) = {sc_mountain};

/* create whole box domain */
v_m_box = newv;
Box(v_m_box) = {0, 0, 0, 50, 100, 50};

BooleanDifference{ Volume{v_m_box}; Delete; }{ Volume{v_m}; Delete; }

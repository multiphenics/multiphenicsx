# Copyright (C) 2016-2020 by the multiphenics authors
#
# This file is part of multiphenics.
#
# multiphenics is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

import pygmsh
from numpy import cos, pi, sin, tan
from dolfinx import *
from dolfinx.cpp.mesh import CellType, GhostMode
from multiphenics import *
from multiphenics.io import XDMFFile
from multiphenics.utils.mesh import generate_boundary_restriction

"""
This file generates the mesh which is used in the following examples:
    7b_stokes_neumann_control
The test case is from section 5.5 of
F. Negri. Reduced basis method for parametrized optimal control problems governed by PDEs. Master thesis, Politecnico di Milano, 2010-2011.
"""

# Geometrical parameters
mu1 = 1.0
mu2 = pi/5.0
mu3 = pi/6.0
mu4 = 1.0
mu5 = 1.7
mu6 = 2.2
lcar = 0.05
# ... and related quantities
Y = 1.0
X = -Y
L = 3.0
B = Y - mu1
H_1 = B + tan(mu2)*mu5
H_2 = B - tan(mu3)*mu6
L_1 = mu1*cos(mu2)*sin(mu2)
L_2 = (B-X)*cos(mu3)*sin(mu3)
N = mu1*cos(mu2)*cos(mu2)
M = - (B-X)*cos(mu3)*cos(mu3)

# Create mesh
geom = pygmsh.built_in.Geometry()
p0 = geom.add_point([0.0, X, 0.0], lcar)
p1 = geom.add_point([L - mu4, X, 0.0], lcar)
p2 = geom.add_point([L, X, 0.0], lcar)
p3 = geom.add_point([L + mu6 - L_2, H_2 + M, 0.0], lcar)
p4 = geom.add_point([L + mu6, H_2, 0.0], lcar)
p5 = geom.add_point([L, B, 0.0], lcar)
p6 = geom.add_point([L + mu5, H_1, 0.0], lcar)
p7 = geom.add_point([L + mu5 - L_1, H_1 + N, 0.0], lcar)
p8 = geom.add_point([L, Y, 0.0], lcar)
p9 = geom.add_point([L - mu4, Y, 0.0], lcar)
p10 = geom.add_point([0.0, Y, 0.0], lcar)
l0 = geom.add_line(p0, p1)
l1 = geom.add_line(p1, p2)
l2 = geom.add_line(p2, p3)
l3 = geom.add_line(p3, p4)
l4 = geom.add_line(p4, p5)
l5 = geom.add_line(p5, p6)
l6 = geom.add_line(p6, p7)
l7 = geom.add_line(p7, p8)
l8 = geom.add_line(p8, p9)
l9 = geom.add_line(p9, p10)
l10 = geom.add_line(p10, p0)
l11 = geom.add_line(p1, p9)
l12 = geom.add_line(p2, p5)
l13 = geom.add_line(p5, p8)
geom.add_physical([l10], label=1)
geom.add_physical([l0, l1, l2, l4, l5, l7, l8, l9], label=2)
geom.add_physical([l3, l6], label=3)
geom.add_physical(l11, label=4)
line_loop_rectangle_left = geom.add_line_loop([l0, l11, l9, l10])
line_loop_rectangle_right = geom.add_line_loop([l1, l12, l13, l8, -l11])
line_loop_bifurcation_top = geom.add_line_loop([l5, l6, l7, -l13])
line_loop_bifurcation_bottom = geom.add_line_loop([l2, l3, l4, -l12])
rectangle_left = geom.add_plane_surface(line_loop_rectangle_left)
rectangle_right = geom.add_plane_surface(line_loop_rectangle_right)
bifurcation_top = geom.add_plane_surface(line_loop_bifurcation_top)
bifurcation_bottom = geom.add_plane_surface(line_loop_bifurcation_bottom)
geom.add_physical(rectangle_left, label=11)
geom.add_physical(rectangle_right, label=12)
geom.add_physical(bifurcation_top, label=13)
geom.add_physical(bifurcation_bottom, label=14)
pygmsh_mesh = pygmsh.generate_mesh(geom)
mesh = Mesh(MPI.comm_world, CellType.triangle, pygmsh_mesh.points[:, :2], pygmsh_mesh.cells["triangle"], [], GhostMode.none)

# Create subdomains
subdomains_mvc = MeshValueCollection("size_t", mesh, mesh.topology.dim, pygmsh_mesh.cells["triangle"], pygmsh_mesh.cell_data["triangle"]["gmsh:physical"] - 10)
subdomains = MeshFunction("size_t", mesh, subdomains_mvc, 0)

# Create boundaries
boundaries_mvc = MeshValueCollection("size_t", mesh, mesh.topology.dim - 1, pygmsh_mesh.cells["line"], pygmsh_mesh.cell_data["line"]["gmsh:physical"])
boundaries = MeshFunction("size_t", mesh, boundaries_mvc, 0)

# Create restrictions
control_restriction = generate_boundary_restriction(mesh, boundaries, 3)

# Save
with XDMFFile(MPI.comm_world, "bifurcation.xdmf") as output:
    output.write(mesh)
with XDMFFile(MPI.comm_world, "bifurcation_subdomains.xdmf") as output:
    output.write(subdomains)
with XDMFFile(MPI.comm_world, "bifurcation_boundaries.xdmf") as output:
    output.write(boundaries)
with XDMFFile(MPI.comm_world, "bifurcation_restriction_control.rtc.xdmf") as output:
    output.write(control_restriction)

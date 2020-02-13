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
from dolfinx import *
from dolfinx.cpp.mesh import CellType, GhostMode
from multiphenics import *
from multiphenics.io import XDMFFile
from multiphenics.utils.mesh import generate_boundary_restriction, generate_interface_restriction, generate_subdomain_restriction

# Geometrical parameters
r = 3
lcar = 1./4.

# Create mesh
geom = pygmsh.built_in.Geometry()
p0 = geom.add_point([0.0, 0.0, 0.0], lcar)
p1 = geom.add_point([0.0, +r, 0.0], lcar)
p2 = geom.add_point([0.0, -r, 0.0], lcar)
c0 = geom.add_circle_arc(p1, p0, p2)
c1 = geom.add_circle_arc(p2, p0, p1)
l0 = geom.add_line(p2, p1)
geom.add_physical([c0, c1], label=1)
geom.add_physical(l0, label=2)
line_loop_left = geom.add_line_loop([c0, l0])
line_loop_right = geom.add_line_loop([c1, -l0])
semicircle_left = geom.add_plane_surface(line_loop_left)
semicircle_right = geom.add_plane_surface(line_loop_right)
geom.add_physical(semicircle_left, label=11)
geom.add_physical(semicircle_right, label=12)
pygmsh_mesh = pygmsh.generate_mesh(geom)
mesh = Mesh(MPI.comm_world, CellType.triangle, pygmsh_mesh.points[:, :2], pygmsh_mesh.cells["triangle"], [], GhostMode.none)

# Create subdomains
subdomains_mvc = MeshValueCollection("size_t", mesh, mesh.topology.dim, pygmsh_mesh.cells["triangle"], pygmsh_mesh.cell_data["triangle"]["gmsh:physical"] - 10)
subdomains = MeshFunction("size_t", mesh, subdomains_mvc, 0)

# Create boundaries
boundaries_mvc = MeshValueCollection("size_t", mesh, mesh.topology.dim - 1, pygmsh_mesh.cells["line"], pygmsh_mesh.cell_data["line"]["gmsh:physical"])
boundaries = MeshFunction("size_t", mesh, boundaries_mvc, 0)

# Create restrictions
left_restriction = generate_subdomain_restriction(mesh, subdomains, 1)
right_restriction = generate_subdomain_restriction(mesh, subdomains, 2)
boundary_restriction = generate_boundary_restriction(mesh, boundaries, 1)
interface_restriction = generate_interface_restriction(mesh, subdomains, {1, 2})

# Save
with XDMFFile(MPI.comm_world, "circle.xdmf") as output:
    output.write(mesh)
with XDMFFile(MPI.comm_world, "circle_subdomains.xdmf") as output:
    output.write(subdomains)
with XDMFFile(MPI.comm_world, "circle_boundaries.xdmf") as output:
    output.write(boundaries)
with XDMFFile(MPI.comm_world, "circle_restriction_left.rtc.xdmf") as output:
    output.write(left_restriction)
with XDMFFile(MPI.comm_world, "circle_restriction_right.rtc.xdmf") as output:
    output.write(right_restriction)
with XDMFFile(MPI.comm_world, "circle_restriction_boundary.rtc.xdmf") as output:
    output.write(boundary_restriction)
with XDMFFile(MPI.comm_world, "circle_restriction_interface.rtc.xdmf") as output:
    output.write(interface_restriction)

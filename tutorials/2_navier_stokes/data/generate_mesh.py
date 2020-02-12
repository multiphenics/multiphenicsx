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
from dolfinx.io import XDMFFile

# Geometrical parameters
pre_step_length = 4.
after_step_length = 14.
pre_step_height = 3.
after_step_height = 5.
lcar = 1./5.

# Create mesh
geom = pygmsh.built_in.Geometry()
p0 = geom.add_point([0.0, after_step_height - pre_step_height, 0.0], lcar)
p1 = geom.add_point([pre_step_length, after_step_height - pre_step_height, 0.0], lcar)
p2 = geom.add_point([pre_step_length, 0.0, 0.0], lcar)
p3 = geom.add_point([pre_step_length + after_step_length, 0.0, 0.0], lcar)
p4 = geom.add_point([pre_step_length + after_step_length, after_step_height, 0.0], lcar)
p5 = geom.add_point([0.0, after_step_height, 0.0], lcar)
l0 = geom.add_line(p0, p1)
l1 = geom.add_line(p1, p2)
l2 = geom.add_line(p2, p3)
l3 = geom.add_line(p3, p4)
l4 = geom.add_line(p4, p5)
l5 = geom.add_line(p5, p0)
geom.add_physical([l5], label=1)
geom.add_physical([l0, l1, l2, l4], label=2)
line_loop = geom.add_line_loop([l0, l1, l2, l3, l4, l5])
domain = geom.add_plane_surface(line_loop)
geom.add_physical(domain, label=0)
pygmsh_mesh = pygmsh.generate_mesh(geom)
mesh = Mesh(MPI.comm_world, CellType.triangle, pygmsh_mesh.points[:, :2], pygmsh_mesh.cells["triangle"], [], GhostMode.none)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, mesh.topology.dim, 0)

# Create boundaries
boundaries_mvc = MeshValueCollection("size_t", mesh, mesh.topology.dim - 1, pygmsh_mesh.cells["line"], pygmsh_mesh.cell_data["line"]["gmsh:physical"])
boundaries = MeshFunction("size_t", mesh, boundaries_mvc, 0)

# Save
with XDMFFile(MPI.comm_world, "backward_facing_step.xdmf") as output:
    output.write(mesh)
with XDMFFile(MPI.comm_world, "backward_facing_step_subdomains.xdmf") as output:
    output.write(subdomains)
with XDMFFile(MPI.comm_world, "backward_facing_step_boundaries.xdmf") as output:
    output.write(boundaries)

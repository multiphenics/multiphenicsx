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
from multiphenics.utils.mesh import generate_boundary_restriction

# Geometrical parameters
r = 3
lcar = 1./4.

# Create mesh
geom = pygmsh.built_in.Geometry()
c0 = geom.add_circle([0.0, 0.0, 0.0], r, lcar)
geom.add_physical(c0.plane_surface, label=0)
geom.add_physical(c0.line_loop.lines, label=1)
pygmsh_mesh = pygmsh.generate_mesh(geom)
mesh = Mesh(MPI.comm_world, CellType.triangle, pygmsh_mesh.points[:, :2], pygmsh_mesh.cells["triangle"], [], GhostMode.none)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, mesh.topology.dim, 0)

# Create boundaries
boundaries_mvc = MeshValueCollection("size_t", mesh, mesh.topology.dim - 1, pygmsh_mesh.cells["line"], pygmsh_mesh.cell_data["line"]["gmsh:physical"])
boundaries = MeshFunction("size_t", mesh, boundaries_mvc, 0)

# Create restrictions
boundary_restriction = generate_boundary_restriction(mesh, boundaries, 1)

# Save
with XDMFFile(MPI.comm_world, "circle.xdmf") as output:
    output.write(mesh)
with XDMFFile(MPI.comm_world, "circle_subdomains.xdmf") as output:
    output.write(subdomains)
with XDMFFile(MPI.comm_world, "circle_boundaries.xdmf") as output:
    output.write(boundaries)
with XDMFFile(MPI.comm_world, "circle_restriction_boundary.rtc.xdmf") as output:
    output.write(boundary_restriction)

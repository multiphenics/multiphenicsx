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

from dolfinx import *
from mshr import *
from multiphenics import *

# Create mesh
domain = Circle(Point(0., 0.), 3.)
mesh = generate_mesh(domain, 15)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, mesh.topology.dim, 0)

# Create boundaries
class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

boundaries = MeshFunction("size_t", mesh, mesh.topology.dim - 1, 0)
on_boundary = OnBoundary()
on_boundary.mark(boundaries, 1)

# Create restrictions
boundary_restriction = MeshRestriction(mesh, on_boundary)

# Save
XDMFFile("circle.xdmf").write(mesh)
XDMFFile("circle_physical_region.xdmf").write(subdomains)
XDMFFile("circle_facet_region.xdmf").write(boundaries)
XDMFFile("circle_restriction_boundary.rtc.xdmf").write(boundary_restriction)

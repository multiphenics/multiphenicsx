# Copyright (C) 2016-2017 by the multiphenics authors
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

from dolfin import *
from mshr import *
from multiphenics import *

# Create mesh
domain = Circle(Point(0., 0.), 3.)
domain_left = domain and Rectangle(Point(-3.0, -3.0), Point(0.0, 3.0))
domain_right = domain and Rectangle(Point(0.0, -3.0), Point(3.0, 3.0))
domain.set_subdomain(1, domain_left)
domain.set_subdomain(2, domain_right)
mesh = generate_mesh(domain, 15)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())

# Create boundaries
class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
class OnInterface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.)

boundaries = FacetFunction("size_t", mesh)
on_boundary = OnBoundary()
on_boundary.mark(boundaries, 1)
on_interface = OnInterface()
on_interface.mark(boundaries, 2)

# Create restrictions
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= 0.
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= 0.
        
boundary_restriction = MeshRestriction(mesh, on_boundary)
interface_restriction = MeshRestriction(mesh, on_interface)
left = Left()
left_restriction = MeshRestriction(mesh, left)
right = Right()
right_restriction = MeshRestriction(mesh, right)

# Save
File("circle.xml") << mesh
File("circle_physical_region.xml") << subdomains
File("circle_facet_region.xml") << boundaries
File("circle_restriction_boundary.rtc.xml") << boundary_restriction
File("circle_restriction_interface.rtc.xml") << interface_restriction
File("circle_restriction_left.rtc.xml") << left_restriction
File("circle_restriction_right.rtc.xml") << right_restriction
XDMFFile("circle.xdmf").write(mesh)
XDMFFile("circle_physical_region.xdmf").write(subdomains)
XDMFFile("circle_facet_region.xdmf").write(boundaries)
XDMFFile("circle_restriction_boundary.rtc.xdmf").write(boundary_restriction)
XDMFFile("circle_restriction_interface.rtc.xdmf").write(interface_restriction)
XDMFFile("circle_restriction_left.rtc.xdmf").write(left_restriction)
XDMFFile("circle_restriction_right.rtc.xdmf").write(right_restriction)

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

"""
This file generates the mesh which is used in the following examples:
    1b_poisson
The test case is from section 5.1 of
F. Negri, G. Rozza, A. Manzoni and A. Quarteroni. Reduced Basis Method for Parametrized Elliptic Optimal Control Problems. SIAM Journal on Scientific Computing, 35(5): A2316-A2340, 2013.
"""

# Create mesh
square = Rectangle(Point(0, 0), Point(1, 1))
rectangle = Rectangle(Point(1, 0), Point(4, 1))
domain = square + rectangle
domain.set_subdomain(1, square)
domain.set_subdomain(2, rectangle)
mesh = generate_mesh(domain, 64)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())

# Create boundaries
class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
        
boundaries = FacetFunction("size_t", mesh)
boundary = Boundary()
boundary.mark(boundaries, 1)

# Save
File("rectangle.xml") << mesh
File("rectangle_physical_region.xml") << subdomains
File("rectangle_facet_region.xml") << boundaries
XDMFFile("rectangle.xdmf").write(mesh)
XDMFFile("rectangle_physical_region.xdmf").write(subdomains)
XDMFFile("rectangle_facet_region.xdmf").write(boundaries)

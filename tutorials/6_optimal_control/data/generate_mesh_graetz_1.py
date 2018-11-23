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

from dolfin import *
from mshr import *
from multiphenics import *

"""
This file generates the mesh which is used in the following examples:
    2b_advection_diffusion_reaction
The test case is from section 5.2 of
F. Negri, G. Rozza, A. Manzoni and A. Quarteroni. Reduced Basis Method for Parametrized Elliptic Optimal Control Problems. SIAM Journal on Scientific Computing, 35(5): A2316-A2340, 2013.
"""

# Create mesh
domain = Rectangle(Point(0, 0), Point(2.5, 1))
square_o = Rectangle(Point(0, 0), Point(1, 1))
rectangle_o = Rectangle(Point(1, 0), Point(2.5, 1))
square_i = Rectangle(Point(0.2, 0.3), Point(0.8, 0.7))
rectangle_i = Rectangle(Point(1.2, 0.3), Point(2.5, 0.7))
domain.set_subdomain(1, square_i)
domain.set_subdomain(2, rectangle_i)
domain.set_subdomain(3, square_o - square_i)
domain.set_subdomain(4, rectangle_o - rectangle_i)
mesh = generate_mesh(domain, 48)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, mesh.topology.dim, mesh.domains())

# Create boundaries
class Boundary_D_1(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
                and
            (
                (x[0] >= 0 and x[0] <= 1 and x[1] > -DOLFIN_EPS and x[1] < DOLFIN_EPS)
                    or
                (x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and x[1] >= 0 and x[1] <= 1)
                    or
                (x[0] >= 0 and x[0] <= 1 and x[1] > 1-DOLFIN_EPS and x[1] < 1+DOLFIN_EPS)
            )
        )
        
class Boundary_D_2(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
                and
            (
                (x[0] >= 1 and x[0] <= 2.5 and x[1] < DOLFIN_EPS and x[1] > -DOLFIN_EPS)
                    or
                (x[0] >= 1 and x[0] <= 2.5 and x[1] > 1-DOLFIN_EPS and x[1] < 1+DOLFIN_EPS)
            )
        )
        
class Boundary_N(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
                and
            (x[0] > 2.5 - DOLFIN_EPS and x[0] < 2.5 + DOLFIN_EPS and x[1] >= 0 and x[1] <= 1)
        )
        
boundaries = MeshFunction("size_t", mesh, mesh.topology.dim - 1)
boundary_D_1 = Boundary_D_1()
boundary_D_1.mark(boundaries, 1)
boundary_D_2 = Boundary_D_2()
boundary_D_2.mark(boundaries, 2)
boundary_N = Boundary_N()
boundary_N.mark(boundaries, 3)

# Save
File("graetz_1.xml") << mesh
File("graetz_1_physical_region.xml") << subdomains
File("graetz_1_facet_region.xml") << boundaries
XDMFFile("graetz_1.xdmf").write(mesh)
XDMFFile("graetz_1_physical_region.xdmf").write(subdomains)
XDMFFile("graetz_1_facet_region.xdmf").write(boundaries)

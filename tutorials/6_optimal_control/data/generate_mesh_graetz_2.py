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
    3b_advection_diffusion_reaction_neumann_control
The test case is from section 5.3 of
F. Negri, G. Rozza, A. Manzoni and A. Quarteroni. Reduced Basis Method for Parametrized Elliptic Optimal Control Problems. SIAM Journal on Scientific Computing, 35(5): A2316-A2340, 2013.
"""

# Create mesh
square = Rectangle(Point(0.0, 0.0), Point(1.0, 1.0))
rectangle = Rectangle(Point(1.0, 0.0), Point(3.0, 1.0))
rectangle_i = Rectangle(Point(1.0, 0.3), Point(3.0, 0.7))
obs_subdomain = rectangle - rectangle_i
domain =  square + rectangle
domain.set_subdomain(1, square)
domain.set_subdomain(2, rectangle_i)
domain.set_subdomain(3, obs_subdomain)
mesh = generate_mesh(domain, 48)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())

# Create boundaries
class Boundary_D(SubDomain):
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
        
class Boundary_C(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
                and
            (
                (x[0] >= 1 and x[0] <= 3 and x[1] < DOLFIN_EPS and x[1] > -DOLFIN_EPS)
                    or
                (x[0] >= 1 and x[0] <= 3 and x[1] > 1-DOLFIN_EPS and x[1] < 1+DOLFIN_EPS)
            )
        )
        
class Boundary_N(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
                and
            (x[0] > 3 - DOLFIN_EPS and x[0] < 3 + DOLFIN_EPS and x[1] >= 0 and x[1] <= 1)
        )
        
boundaries = FacetFunction("size_t", mesh)
boundary_D = Boundary_D()
boundary_D.mark(boundaries, 1)
boundary_C = Boundary_C()
boundary_C.mark(boundaries, 2)
boundary_N = Boundary_N()
boundary_N.mark(boundaries, 3)

# Create restrictions
control_restriction = MeshRestriction(mesh, boundary_C)

# Save
File("graetz_2.xml") << mesh
File("graetz_2_physical_region.xml") << subdomains
File("graetz_2_facet_region.xml") << boundaries
File("graetz_2_restriction_control.rtc.xml") << control_restriction
XDMFFile("graetz_2.xdmf").write(mesh)
XDMFFile("graetz_2_physical_region.xdmf").write(subdomains)
XDMFFile("graetz_2_facet_region.xdmf").write(boundaries)
XDMFFile("graetz_2_restriction_control.rtc.xdmf").write(control_restriction)

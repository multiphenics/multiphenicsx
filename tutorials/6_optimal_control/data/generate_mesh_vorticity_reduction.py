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
    7a_stokes_dirichlet_control
The test case is from section 5 of
F. Negri, A. Manzoni and G. Rozza. Reduced basis approximation of parametrized optimal flow control problems for the Stokes equations. Computer and Mathematics with Applications, 69(4):319-336, 2015.
"""

# Geometrical parameters
H = 0.9
h = 0.35
r = 0.1

# Create mesh
rectangle_1 = Rectangle(Point(0.0, 0.0), Point(H, 1.0))
rectangle_2 = Rectangle(Point(H, 0.0), Point(H + h, 1.0))
rectangle_3 = Rectangle(Point(H + h, 0.0), Point(2.0, 1.0))
domain_1 = rectangle_1 + rectangle_2 + rectangle_3
rectangle_i = Rectangle(Point(H, 0.4), Point(H + h, 0.6))
circle_i = Circle(Point(H, 0.5), r, 32)
domain = domain_1 - (rectangle_i + circle_i)
observation = Polygon([Point(H + h, 0.4), Point(1.8, 0.2), Point(1.8, 0.8), Point(H + h, 0.6)])
domain.set_subdomain(1, rectangle_1)
domain.set_subdomain(2, rectangle_2)
domain.set_subdomain(3, rectangle_3 - observation)
domain.set_subdomain(4, observation)
mesh = generate_mesh(domain, 64)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())

# Create boundaries
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < DOLFIN_EPS

class Boundary_s(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[1] < DOLFIN_EPS or x[1] > 1 - DOLFIN_EPS)

class Boundary_N(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] > 2.0 - DOLFIN_EPS

class Boundary_c(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
                and
            (
                (x[0] >= H and x[0] <= H + h and x[1] < 0.4 + DOLFIN_EPS and x[1] > 0.4 - DOLFIN_EPS)
                    or
                (x[0] >= H and x[0] <= H + h and x[1] < 0.6 + DOLFIN_EPS and x[1] > 0.6 - DOLFIN_EPS)
            )
        )

class Boundary_w(SubDomain):
    def inside(self, x, on_boundary):
        r_x = sqrt((x[0] - H)**2 + (x[1] - 0.5)**2)
        return (
            on_boundary
                and
            (
                (x[0] < H + h + DOLFIN_EPS and x[0] > H + h - DOLFIN_EPS and x[1] >= 0.4 and x[1] <= 0.6)
                    or
                r_x < r + DOLFIN_EPS
            )
        )
        
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
inlet = Inlet()
inlet.mark(boundaries, 1) # Gamma_in
boundary_s = Boundary_s()
boundary_s.mark(boundaries, 2) # Gamma_S
boundary_N = Boundary_N()
boundary_N.mark(boundaries, 3) # Gamma_N
boundary_c = Boundary_c()
boundary_c.mark(boundaries, 4) # Gamma_C
boundary_w = Boundary_w()
boundary_w.mark(boundaries, 5) # Gamma_W

# Create restrictions
control_restriction = MeshRestriction(mesh, boundary_c)

# Save
File("vorticity_reduction.xml") << mesh
File("vorticity_reduction_physical_region.xml") << subdomains
File("vorticity_reduction_facet_region.xml") << boundaries
File("vorticity_reduction_restriction_control.rtc.xml") << control_restriction
XDMFFile("vorticity_reduction.xdmf").write(mesh)
XDMFFile("vorticity_reduction_physical_region.xdmf").write(subdomains)
XDMFFile("vorticity_reduction_facet_region.xdmf").write(boundaries)
XDMFFile("vorticity_reduction_restriction_control.rtc.xdmf").write(control_restriction)

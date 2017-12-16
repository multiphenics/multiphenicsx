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
rectangle_1 = Rectangle(Point(0.0, X), Point(L - mu4, Y))
rectangle_2 = Rectangle(Point(L - mu4, X), Point(L, Y))
bifurcation_1 = Polygon([Point(L, B), Point(L + mu5, H_1), Point(L + mu5 - L_1, H_1 + N), Point(L, Y)])
bifurcation_2 = Polygon([Point(L, X), Point(L + mu6 - L_2, H_2 + M), Point(L + mu6, H_2), Point(L, B)])
domain = rectangle_1 + rectangle_2 + bifurcation_1 + bifurcation_2
domain.set_subdomain(1, rectangle_1)
domain.set_subdomain(2, rectangle_2)
domain.set_subdomain(3, bifurcation_1)
domain.set_subdomain(4, bifurcation_2)
mesh = generate_mesh(domain, 80)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())

# Create boundaries
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < DOLFIN_EPS

class Boundary_D(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
                and
            (
                (x[1] > X - DOLFIN_EPS and x[1] < X+DOLFIN_EPS and x[0] >= 0.0 and x[0] <= L)
                    or
                (x[0] >= 0.0 and x[0] <= L and x[1] > Y - DOLFIN_EPS and x[1] < Y + DOLFIN_EPS)
                    or
                (x[0] >= L and x[0] <= L+mu5 and x[1] >= B and x[1] <= H_1)
                    or
                (x[0] > L and x[0] <= L+mu5-L_1 and x[1] >= Y and x[1] <= H_1+N)
                    or
                (x[0] > L and x[0] <= L+mu6 and x[1] >= H_2 and x[1] <= B)
                    or
                (x[0] > L and x[0] <= L+mu6-L_2 and x[1] >= H_2 + M and x[1] <= B)
            )
        )

class Boundary_C(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
                and
            (
                (x[0] <= L+mu5 and x[0] >= L+mu5-L_1 and x[1] >= H_1 and x[1] <= H_1+N)
                    or
                (x[0] >= L+mu6-L_2 and x[0] <= L+mu6 and x[1] >= H_2+M and x[1] <= H_2)
            )
        )
        
class Boundary_Obs(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < (L-mu4 + DOLFIN_EPS) and x[0] > (L-mu4 - DOLFIN_EPS) and x[1] >= X and x[1] <= Y
        
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
inlet = Inlet()
inlet.mark(boundaries, 1)
boundary_d = Boundary_D()
boundary_d.mark(boundaries, 2)
boundary_c = Boundary_C()
boundary_c.mark(boundaries, 3)
boundary_obs = Boundary_Obs()
boundary_obs.mark(boundaries, 4)

# Create restrictions
control_restriction = MeshRestriction(mesh, boundary_c)

# Save
File("bifurcation.xml") << mesh
File("bifurcation_physical_region.xml") << subdomains
File("bifurcation_facet_region.xml") << boundaries
File("bifurcation_restriction_control.rtc.xml") << control_restriction
XDMFFile("bifurcation.xdmf").write(mesh)
XDMFFile("bifurcation_physical_region.xdmf").write(subdomains)
XDMFFile("bifurcation_facet_region.xdmf").write(boundaries)
XDMFFile("bifurcation_restriction_control.rtc.xdmf").write(control_restriction)

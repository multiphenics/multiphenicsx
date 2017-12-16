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
    1a_poisson
    2a_advection_diffusion_reaction
    3a_advection_diffusion_reaction_neumann_control
    4a_poisson_dirichlet_control
    4b_poisson_dirichlet_control
    4c_poisson_neumann_control_boundary_observation
    5_stokes
    6_navier_stokes
    8_navier_stokes_neumann_control
"""

# Create mesh
domain = Rectangle(Point(0., 0.), Point(1., 1.))
mesh = generate_mesh(domain, 32)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)

# Create boundaries
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < DOLFIN_EPS

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 1.) < DOLFIN_EPS

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1]) < DOLFIN_EPS
                
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 1.) < DOLFIN_EPS
        
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
bottom = Bottom()
bottom.mark(boundaries, 1)
left = Left()
left.mark(boundaries, 2)
top = Top()
top.mark(boundaries, 3)
right = Right()
right.mark(boundaries, 4)

# Create restrictions
left_restriction = MeshRestriction(mesh, left)

# Save
File("square.xml") << mesh
File("square_physical_region.xml") << subdomains
File("square_facet_region.xml") << boundaries
File("square_restriction_boundary_2.rtc.xml") << left_restriction
XDMFFile("square.xdmf").write(mesh)
XDMFFile("square_physical_region.xdmf").write(subdomains)
XDMFFile("square_facet_region.xdmf").write(boundaries)
XDMFFile("square_restriction_boundary_2.rtc.xdmf").write(left_restriction)

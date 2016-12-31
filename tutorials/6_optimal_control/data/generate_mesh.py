# Copyright (C) 2015-2017 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import *
from mshr import *

## 1. INTERIOR MESH ##
# Create mesh
domain = Rectangle(Point(0., 0.), Point(1., 1.))
mesh = generate_mesh(domain, 32)
plot(mesh)
interactive()

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)
plot(subdomains)
interactive()

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
        
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
bottom = Bottom()
bottom.mark(boundaries, 1)
left = Left()
left.mark(boundaries, 2)
top = Top()
top.mark(boundaries, 3)
right = Right()
right.mark(boundaries, 4)
plot(boundaries)
interactive()

# Save
File("square.xml") << mesh
File("square.pvd") << mesh
File("square_physical_region.xml") << subdomains
File("square_facet_region.xml") << boundaries

## 2. BOUNDARY MESHES ##
boundary_mesh = BoundaryMesh(mesh, "exterior")

class LeftOnBoundaryMesh(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0]) < DOLFIN_EPS

class RightOnBoundaryMesh(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.) < DOLFIN_EPS

class BottomOnBoundaryMesh(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1]) < DOLFIN_EPS
                
class TopOnBoundaryMesh(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1] - 1.) < DOLFIN_EPS
        
boundaries_on_boundary_mesh = CellFunction("size_t", boundary_mesh)
boundaries_on_boundary_mesh.set_all(0)
bottom_on_boundary_mesh = BottomOnBoundaryMesh()
bottom_on_boundary_mesh.mark(boundaries_on_boundary_mesh, 1)
left_on_boundary_mesh = LeftOnBoundaryMesh()
left_on_boundary_mesh.mark(boundaries_on_boundary_mesh, 2)
top_on_boundary_mesh = TopOnBoundaryMesh()
top_on_boundary_mesh.mark(boundaries_on_boundary_mesh, 3)
right_on_boundary_mesh = RightOnBoundaryMesh()
right_on_boundary_mesh.mark(boundaries_on_boundary_mesh, 4)

for idx in (1, 2, 3, 4):
    sub_boundary_mesh = SubMesh(boundary_mesh, boundaries_on_boundary_mesh, idx)
    File("boundary_square_" + str(idx) + ".xml") << sub_boundary_mesh
    File("boundary_square_" + str(idx) + ".pvd") << sub_boundary_mesh
    

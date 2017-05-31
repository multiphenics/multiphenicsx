# Copyright (C) 2015-2017 by the RBniCS authors
# Copyright (C) 2016-2017 by the multiphenics authors
#
# This file is part of the RBniCS interface to multiphenics.
#
# RBniCS and multiphenics are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and multiphenics are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import *
from mshr import *

def generate_mesh1():
    pass # Uses the same mesh as tutorial 04
    
def generate_mesh2():
    pass # Copy the one from rbnics
    
def generate_mesh3():
    # Create mesh
    rectangle = Rectangle(Point(0., 0.), Point(2., 1.))
    subdomain = dict()
    subdomain[1] = Rectangle(Point(0., 0.), Point(1., 1.))
    subdomain[2] = Rectangle(Point(1., 0.2), Point(2., 0.8))
    subdomain[3] = rectangle - subdomain[1] - subdomain[2]
    domain = rectangle
    for i, s in subdomain.iteritems():
        domain.set_subdomain(i, subdomain[i])
    mesh = generate_mesh(domain, 64)
    plot(mesh)
    interactive()

    # Create subdomains
    subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())
    plot(subdomains)
    interactive()
    
    # Create boundaries
    class Left(SubDomain):
        def __init__(self):
            SubDomain.__init__(self)
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[0] - 0.) < DOLFIN_EPS 
       
    class Right(SubDomain):
        def __init__(self):
            SubDomain.__init__(self)
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[0] - 2.) < DOLFIN_EPS 
            
    class Bottom(SubDomain):
        def __init__(self, x_min, x_max):
            SubDomain.__init__(self)
            self.x_min = x_min
            self.x_max = x_max
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[1] - 0.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max
            
    class Top(SubDomain):
        def __init__(self, x_min, x_max):
            SubDomain.__init__(self)
            self.x_min = x_min
            self.x_max = x_max
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[1] - 1.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max

            
    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)
    left = Left()
    left.mark(boundaries, 1)
    bottom1 = Bottom(0., 1.) 
    bottom1.mark(boundaries, 1)
    top1 = Top(0., 1.)
    top1.mark(boundaries, 1)
    bottom2 = Bottom(1., 2.) 
    bottom2.mark(boundaries, 2)
    top2 = Top(1., 2.) 
    top2.mark(boundaries, 2)
    right = Right()
    right.mark(boundaries, 3)
    plot(boundaries)
    interactive()

    # Create control mesh
    boundary_mesh = BoundaryMesh(mesh, "exterior")
    
    class BottomOnBoundaryMesh(SubDomain):
        def __init__(self, x_min, x_max):
            SubDomain.__init__(self)
            self.x_min = x_min
            self.x_max = x_max
        def inside(self, x, on_boundary):
            return abs(x[1] - 0.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max
            
    class TopOnBoundaryMesh(SubDomain):
        def __init__(self, x_min, x_max):
            SubDomain.__init__(self)
            self.x_min = x_min
            self.x_max = x_max
        def inside(self, x, on_boundary):
            return abs(x[1] - 1.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max
            
    boundaries_on_boundary_mesh = CellFunction("size_t", boundary_mesh)
    boundaries_on_boundary_mesh.set_all(0)
    bottom2_on_boundary_mesh = BottomOnBoundaryMesh(1., 2.)
    bottom2_on_boundary_mesh.mark(boundaries_on_boundary_mesh, 1)
    top2_on_boundary_mesh = TopOnBoundaryMesh(1., 2.)
    top2_on_boundary_mesh.mark(boundaries_on_boundary_mesh, 1)
    
    control_mesh = SubMesh(boundary_mesh, boundaries_on_boundary_mesh, 1)
    plot(control_mesh)
    interactive()
    
    # Save
    File("mesh3.xml") << mesh
    File("mesh3.pvd") << mesh
    File("mesh3_physical_region.xml") << subdomains
    File("mesh3_control_region.xml") << control_mesh
    File("mesh3_control_region.pvd") << control_mesh
    File("mesh3_facet_region.xml") << boundaries
    
generate_mesh1()
generate_mesh2()
generate_mesh3()

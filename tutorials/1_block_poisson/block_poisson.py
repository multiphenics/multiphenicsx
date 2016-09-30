# Copyright (C) 2016 by the block_ext authors
#
# This file is part of block_ext.
#
# block_ext is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# block_ext is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with block_ext. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import *
from block_ext import *

"""
In this tutorial we first solve the problem

-u'' = f    in Omega = [0, 1]
 u   = 0    on Gamma = {0, 1}
 
using standard FEniCS code.

Then we use block_ext to solve the system

-   w_1'' - 2 w_2'' = 3 f    in Omega
- 3 w_1'' - 4 w_2'' = 7 f    in Omega

subject to

 w_1 = 0    on Gamma = {0, 1}
 w_2 = 0    on Gamma = {0, 1}
 
By construction the solution of the system is
    (w_1, w_2) = (u, u)

We then compare the solution provided by block_ext
to the one provided by standard FEniCS.
"""

# Mesh generation
mesh = UnitIntervalMesh(32)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 0.) < DOLFIN_EPS

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 1.) < DOLFIN_EPS
        
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
left = Left()
left.mark(boundaries, 1)
right = Right()
right.mark(boundaries, 1)

# Standard assembly
V = FunctionSpace(mesh, "Lagrange", 2)
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v))*dx + u*v*dx
x0 = SpatialCoordinate(mesh)[0]
f = 100*sin(20*x0)*v*dx
bc = DirichletBC(V, Constant(0.), boundaries, 1)

# Standard solve
A = assemble(a)
bc.apply(A)
F = assemble(f)
bc.apply(F)
U = Function(V)
solve(A, U.vector(), F)

# Create the block matrix for the LHS
AA = block_assemble([[1.*a, 2.*a],
                     [3.*a, 4.*a]])
# Create the block vector for the RHS
FF = block_assemble([3.*f, 
                     7.*f])
# Add block BCs
bcs = BlockDirichletBC([bc, 
                        bc])
bcs.apply(AA)
bcs.apply(FF)

# Block solve
UU = BlockFunction([V, V])
block_solve(AA, UU.block_vector(), FF)
UU1, UU2 = UU

#plot(U, title="0")
#plot(UU1, title="1")
#plot(UU2, title="2")
plot(U - UU1, title="e1")
plot(U - UU2, title="e2")
interactive()

# Export the matrix/vector to file
block_matlab_export(AA, "AA", FF, "FF")



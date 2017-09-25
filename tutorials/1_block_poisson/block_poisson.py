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

from numpy import isclose
from dolfin import *
import matplotlib.pyplot as plt
from multiphenics import *

"""
In this tutorial we first solve the problem

-u'' = f    in Omega = [0, 1]
 u   = 0    on Gamma = {0, 1}
 
using standard FEniCS code.

Then we use multiphenics to solve the system

-   w_1'' - 2 w_2'' = 3 f    in Omega
- 3 w_1'' - 4 w_2'' = 7 f    in Omega

subject to

 w_1 = 0    on Gamma = {0, 1}
 w_2 = 0    on Gamma = {0, 1}
 
By construction the solution of the system is
    (w_1, w_2) = (u, u)

We then compare the solution provided by multiphenics
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

x0 = SpatialCoordinate(mesh)[0]

def run_standard():
    # Define a function space
    V = FunctionSpace(mesh, "Lagrange", 2)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Create the matrix for the LHS
    a = inner(grad(u), grad(v))*dx + u*v*dx
    A = assemble(a)

    # Create the vector for the RHS
    f = 100*sin(20*x0)*v*dx
    F = assemble(f)

    # Apply boundary conditions
    bc = DirichletBC(V, Constant(0.), boundaries, 1)
    bc.apply(A)
    bc.apply(F)

    # Solve the linear system
    U = Function(V)
    solve(A, U.vector(), F)
    
    # Return the solution
    return U
    
U = run_standard()

def run_block():
    # Define a block function space
    V = FunctionSpace(mesh, "Lagrange", 2)
    VV = BlockFunctionSpace([V, V])
    uu = BlockTrialFunction(VV)
    vv = BlockTestFunction(VV)
    (u1, u2) = block_split(uu)
    (v1, v2) = block_split(vv)

    # Create the block matrix for the block LHS
    aa = [[1*inner(grad(u1), grad(v1))*dx + 1*u1*v1*dx, 2*inner(grad(u2), grad(v1))*dx + 2*u2*v1*dx],
          [3*inner(grad(u1), grad(v2))*dx + 3*u1*v2*dx, 4*inner(grad(u2), grad(v2))*dx + 4*u2*v2*dx]]
    AA = block_assemble(aa)
    
    # Create the block vector for the block RHS
    ff = [300*sin(20*x0)*v1*dx,
          700*sin(20*x0)*v2*dx]
    FF = block_assemble(ff)
    
    # Apply block boundary conditions
    bc1 = DirichletBC(VV.sub(0), Constant(0.), boundaries, 1)
    bc2 = DirichletBC(VV.sub(1), Constant(0.), boundaries, 1)
    bcs = BlockDirichletBC([bc1,
                            bc2])
    bcs.apply(AA)
    bcs.apply(FF)
    
    # Solve the block linear system
    UU = BlockFunction(VV)
    block_solve(AA, UU.block_vector(), FF)
    UU1, UU2 = UU
    
    # Return the block solution
    return UU1, UU2
    
UU1, UU2 = run_block()

# plt.figure()
# plot(U, title="0")
# plt.figure()
# plot(UU1, title="1")
# plt.figure()
# plot(UU2, title="2")
plt.figure()
plot(U - UU1, title="e1")
plt.figure()
plot(U - UU2, title="e2")
plt.show()

U_norm = sqrt(assemble(inner(grad(U), grad(U))*dx))
err_1_norm = sqrt(assemble(inner(grad(U - UU1), grad(U - UU1))*dx))
err_2_norm = sqrt(assemble(inner(grad(U - UU2), grad(U - UU2))*dx))
print("Relative error for first component is equal to", err_1_norm/U_norm)
print("Relative error for second component is equal to", err_2_norm/U_norm)
assert isclose(err_1_norm/U_norm, 0., atol=1.e-10)
assert isclose(err_2_norm/U_norm, 0., atol=1.e-10)

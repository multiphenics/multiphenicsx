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
import matplotlib.pyplot as plt
from mshr import *
from multiphenics import *

"""
In this example we solve a Laplace problem with non-homogeneous
Dirichlet boundary conditions. To impose them we use Lagrange multipliers.
Note that standard FEniCS code does not easily support Lagrange multipliers,
because FEniCS does not support subdomain/boundary restricted function spaces,
and thus one would have to declare the Lagrange multiplier on the entire
domain and constrain it in the interior. This procedure would require
the definition of suitable FacetFunction-s to constrain the additional DOFs, 
resulting in a (1) cumbersome mesh definition for the user and (2) unnecessarily
large linear system.
Instead, using multiphenics, we provide a BlockDiscardDOFs class that carries out
automatically the discard of unnecessary DOFs, also reducing the linear system.
"""

## MESHES ##
# Create mesh
domain = Circle(Point(0., 0.), 3.)
mesh = generate_mesh(domain, 15)
# SubDomain definition for boundary restriction
class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
on_boundary = OnBoundary()

## FUNCTION SPACES ##
# Function space
V = FunctionSpace(mesh, "Lagrange", 2)
# Block function space
W = BlockFunctionSpace([V, V], restrict=[None, on_boundary])

## TRIAL/TEST FUNCTIONS ##
ul = BlockTrialFunction(W)
(u, l) = block_split(ul)
vm = BlockTestFunction(W)
(v, m) = block_split(vm)

## MEASURES ##
dx = Measure("dx")(domain=mesh)
ds = Measure("ds")(domain=mesh)

## ASSEMBLE ##
g = Expression("sin(3*x[0] + 1)*sin(3*x[1] + 1)", element=V.ufl_element())
a = [[inner(grad(u),grad(v))*dx, l*v*ds], 
     [u*m*ds                   , 0     ]]
f =  [v*dx                     , g*m*ds]

## SOLVE ##
A = block_assemble(a)
F = block_assemble(f)
block_matlab_export(A, "A")
block_matlab_export(F, "F")

U = BlockFunction(W)
block_solve(A, U.block_vector(), F)

#plt.figure(); plot(U[0])
#plt.figure(); plot(U[1])
#plt.show()

## ERROR ##
A_ex = assemble(a[0][0])
F_ex = assemble(f[0])
bc_ex = DirichletBC(V, g, on_boundary)
bc_ex.apply(A_ex)
bc_ex.apply(F_ex)
U_ex = Function(V)
solve(A_ex, U_ex.vector(), F_ex)
plt.figure(); plot(U_ex)
err = Function(V)
err.vector().add_local(+ U_ex.vector().array())
err.vector().add_local(- U[0].vector().array())
err.vector().apply("")
plt.figure(); plot(err)
plt.show()

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
from mshr import *
from block_ext import *

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
Instead, using block_ext, we provide a BlockDiscardDOFs class that carries out
automatically the discard of unnecessary DOFs, also reducing the linear system.
"""

## MESHES ##
# Create interior mesh
domain = Circle(Point(0., 0.), 3.)
mesh = generate_mesh(domain, 15)
# Create boundary mesh
boundary_mesh = BoundaryMesh(mesh, "exterior")

## FUNCTION SPACES ##
# For interior meshes
V = FunctionSpace(mesh, "Lagrange", 2)
# For boundary
boundary_V = FunctionSpace(boundary_mesh, "Lagrange", 2)

## TRIAL/TEST FUNCTIONS ##
# For interior meshes
u = TrialFunction(V)
v = TestFunction(V)

## EXTENDEND TRIAL/TEST FUNCTIONS ##
# For boundary
l = TrialFunction(V)
m = TestFunction(V)

## MEASURES ##
# For interior meshes
dx = Measure("dx")(domain=mesh)
ds = Measure("ds")(domain=mesh)

## ASSEMBLE ##
g = Expression("sin(3*x[0] + 1)*sin(3*x[1] + 1)", element=V.ufl_element())
a = [[inner(grad(u),grad(v))*dx, l*v*ds               ], 
     [u*m*ds                   , Constant(0.)*l*m*ds]]
f = [v*dx                      , g*m*ds                ]
discard_dofs = BlockDiscardDOFs(
    [V, boundary_V],
    [V,           V]
)

## SOLVE ##
A = block_assemble(a)
F = block_assemble(f)
block_matlab_export(A, "A", F, "F", discard_dofs)

U = BlockFunction([V, boundary_V])
block_solve(A, U.block_vector(), F, discard_dofs)

#plot(U[0])
#plot(U[1])
#interactive()

## ERROR ##
A_ex = assemble(a[0][0])
F_ex = assemble(f[0])
def boundary(x, on_boundary):
	return on_boundary
bc_ex = DirichletBC(V, g, boundary)
bc_ex.apply(A_ex)
bc_ex.apply(F_ex)
U_ex = Function(V)
solve(A_ex, U_ex.vector(), F_ex)
plot(U_ex)
err = Function(V)
err.vector().add_local(+ U_ex.vector().array())
err.vector().add_local(- U[0].vector().array())
err.vector().apply("")
plot(err)
interactive()

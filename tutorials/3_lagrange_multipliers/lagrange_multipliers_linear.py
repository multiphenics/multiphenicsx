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
# import matplotlib.pyplot as plt
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
This task is more easily handled by multiphenics by providing a restriction
in the definition of the (block) function space. Such restriction (which is
basically a collection of MeshFunction-s) can be generated from a SubDomain object,
see data/generate_mesh.py
"""

# MESHES #
# Mesh
mesh = Mesh("data/circle.xml")
subdomains = MeshFunction("size_t", mesh, "data/circle_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/circle_facet_region.xml")
# Dirichlet boundary
boundary_restriction = MeshRestriction(mesh, "data/circle_restriction_boundary.rtc.xml")

# FUNCTION SPACES #
# Function space
V = FunctionSpace(mesh, "Lagrange", 2)
# Block function space
W = BlockFunctionSpace([V, V], restrict=[None, boundary_restriction])

# TRIAL/TEST FUNCTIONS #
ul = BlockTrialFunction(W)
(u, l) = block_split(ul)
vm = BlockTestFunction(W)
(v, m) = block_split(vm)

# MEASURES #
dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)

# ASSEMBLE #
g = Expression("sin(3*x[0] + 1)*sin(3*x[1] + 1)", element=V.ufl_element())
a = [[inner(grad(u), grad(v))*dx, l*v*ds],
     [u*m*ds                    , 0     ]]
f =  [v*dx                      , g*m*ds]

# SOLVE #
A = block_assemble(a)
F = block_assemble(f)

U = BlockFunction(W)
block_solve(A, U.block_vector(), F)

# plt.figure()
# plot(U[0])
# plt.figure()
# plot(U[1])
# plt.show()

# ERROR #
A_ex = assemble(a[0][0])
F_ex = assemble(f[0])
bc_ex = DirichletBC(V, g, boundaries, 1)
bc_ex.apply(A_ex)
bc_ex.apply(F_ex)
U_ex = Function(V)
solve(A_ex, U_ex.vector(), F_ex)
# plt.figure()
# plot(U_ex)
# plt.show()
err = Function(V)
err.vector().add_local(+ U_ex.vector().get_local())
err.vector().add_local(- U[0].vector().get_local())
err.vector().apply("")
# plt.figure()
# plot(err)
# plt.show()
U_ex_norm = sqrt(assemble(inner(grad(U_ex), grad(U_ex))*dx))
err_norm = sqrt(assemble(inner(grad(err), grad(err))*dx))
print("Relative error is equal to", err_norm/U_ex_norm)
assert isclose(err_norm/U_ex_norm, 0., atol=1.e-10)

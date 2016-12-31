# Copyright (C) 2016-2017 by the block_ext authors
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
In this tutorial we compute the inf-sup constant
of the saddle point problem resulting from a Laplace problem with non-homogeneous
Dirichlet boundary conditions imposed by Lagrange multipliers.
"""

## MESHES ##
# Create interior mesh
domain = Circle(Point(0., 0.), 3.)
mesh = generate_mesh(domain, 15)
# Create boundary mesh
boundary_mesh = BoundaryMesh(mesh, "exterior")

## FUNCTION SPACES ##
# Interior space
V = FunctionSpace(mesh, "Lagrange", 2)
# Boundary space
boundary_V = FunctionSpace(boundary_mesh, "Lagrange", 2)
# For block problem definition
W = BlockFunctionSpace([V, V], keep=[V, boundary_V])

## TRIAL/TEST FUNCTIONS ##
ul = BlockTrialFunction(W)
(u, l) = block_split(ul)
vm = BlockTestFunction(W)
(v, m) = block_split(vm)

## MEASURES ##
dx = Measure("dx")(domain=mesh)
ds = Measure("ds")(domain=mesh)

## ASSEMBLE ##
a = [[inner(grad(u),grad(v))*dx , - l*v*ds], 
     [- u*m*ds                  , 0       ]]
b = [[0                         , 0       ], 
     [0                         , - l*m*ds]]

## SOLVE ##
A = block_assemble(a)
B = block_assemble(b)
eigensolver = BlockSLEPcEigenSolver(A, B)
eigensolver.parameters["problem_type"] = "gen_non_hermitian"
eigensolver.parameters["spectrum"] = "smallest real"
eigensolver.parameters["spectral_transform"] = "shift-and-invert"
eigensolver.parameters["spectral_shift"] = 1.e-5
eigensolver.solve(1)
r, c = eigensolver.get_eigenvalue(0)
assert abs(c) < 1.e-10
assert r > 0., "r = " + str(r) + " is not positive"
print "Inf-sup constant: ", sqrt(r)

# Export matrices to MATLAB format to double check the result.
# You will need to convert the matrix to dense storage and use eig()
block_matlab_export(A, "A")
block_matlab_export(B, "B")

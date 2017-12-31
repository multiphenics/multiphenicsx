# Copyright (C) 2016-2018 by the multiphenics authors
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
from multiphenics import *

"""
In this tutorial we compute the inf-sup constant
of the saddle point problem resulting from a Laplace problem with non-homogeneous
Dirichlet boundary conditions imposed by Lagrange multipliers.
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
a = [[inner(grad(u), grad(v))*dx, - l*v*ds],
     [- u*m*ds                  , 0       ]]
b = [[0                         , 0       ],
     [0                         , - l*m*ds]]

# SOLVE #
A = block_assemble(a)
B = block_assemble(b)
eigensolver = BlockSLEPcEigenSolver(A, B)
eigensolver.parameters["problem_type"] = "gen_non_hermitian"
eigensolver.parameters["spectrum"] = "target real"
eigensolver.parameters["spectral_transform"] = "shift-and-invert"
eigensolver.parameters["spectral_shift"] = 1.e-5
eigensolver.solve(1)
r, c = eigensolver.get_eigenvalue(0)
assert abs(c) < 1.e-10
assert r > 0., "r = " + str(r) + " is not positive"
print("Inf-sup constant: ", sqrt(r))
assert isclose(sqrt(r), 0.088385)

# Export matrices to MATLAB format to double check the result.
# You will need to convert the matrix to dense storage and use eig()
block_matlab_export(A, "A")
block_matlab_export(B, "B")

# Copyright (C) 2016-2020 by the multiphenics authors
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
from petsc4py import PETSc
from ufl import *
from dolfinx import *
from dolfinx.cpp.mesh import GhostMode
from multiphenics import *
from multiphenics.fem import block_assemble
from multiphenics.io import XDMFFile

"""
In this tutorial we compute the inf-sup constant
of the saddle point problem resulting from a Laplace problem with non-homogeneous
Dirichlet boundary conditions imposed by Lagrange multipliers.
"""

# MESHES #
# Mesh
mesh = XDMFFile(MPI.comm_world, "data/circle.xdmf").read_mesh(GhostMode.none)
subdomains = XDMFFile(MPI.comm_world, "data/circle_physical_region.xdmf").read_mf_size_t(mesh)
boundaries = XDMFFile(MPI.comm_world, "data/circle_facet_region.xdmf").read_mf_size_t(mesh)
# Dirichlet boundary
boundary_restriction = XDMFFile(MPI.comm_world, "data/circle_restriction_boundary.rtc.xdmf").read_mesh_restriction(mesh)

# FUNCTION SPACES #
# Function space
V = FunctionSpace(mesh, ("Lagrange", 2))
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
A = block_assemble(a)
B = block_assemble(b)

# SOLVE #
options = PETSc.Options()
options_prefix = "multiphenics_eigensolver_"
options.setValue(options_prefix + "eps_gen_non_hermitian", "")
options.setValue(options_prefix + "eps_target_real", "")
options.setValue(options_prefix + "eps_target", 1.e-5)
options.setValue(options_prefix + "st_type", "sinvert")
options.setValue(options_prefix + "st_ksp_type", "preonly")
options.setValue(options_prefix + "st_pc_type", "lu")
options.setValue(options_prefix + "st_pc_factor_mat_solver_type", "mumps")
eigensolver = BlockSLEPcEigenSolver(A, B)
eigensolver.set_options_prefix(options_prefix)
eigensolver.set_from_options()
eigensolver.solve(1)
eigv = eigensolver.get_eigenvalue(0)
r, c = eigv.real, eigv.imag
assert abs(c) < 1.e-10
assert r > 0., "r = " + str(r) + " is not positive"
print("Inf-sup constant: ", sqrt(r))
assert isclose(sqrt(r), 0.088385)

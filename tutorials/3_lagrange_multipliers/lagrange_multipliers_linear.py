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

import numpy
from numpy import isclose, where
from petsc4py import PETSc
from ufl import *
from dolfinx import *
from dolfinx.cpp.mesh import GhostMode
from dolfinx.fem import assemble_scalar, locate_dofs_topological
from multiphenics import *
from multiphenics.io import XDMFFile

"""
In this example we solve a Laplace problem with non-homogeneous
Dirichlet boundary conditions. To impose them we use Lagrange multipliers.
Note that standard FEniCS code does not easily support Lagrange multipliers,
because FEniCS does not support subdomain/boundary restricted function spaces,
and thus one would have to declare the Lagrange multiplier on the entire
domain and constrain it in the interior. This procedure would require
the definition of suitable MeshFunction-s to constrain the additional DOFs,
resulting in a (1) cumbersome mesh definition for the user and (2) unnecessarily
large linear system.
This task is more easily handled by multiphenics by providing a restriction
in the definition of the (block) function space. Such restriction (which is
basically a collection of MeshFunction-s) can be generated from a SubDomain object,
see data/generate_mesh.py
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
g = Function(V)
g.interpolate(lambda x: numpy.sin(3*x[0] + 1)*numpy.sin(3*x[1] + 1))
a = [[inner(grad(u), grad(v))*dx, l*v*ds],
     [u*m*ds                    , 0     ]]
f =  [v*dx                      , g*m*ds]

# SOLVE #
ul = BlockFunction(W)
solver_parameters = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
block_solve(a, ul, f, petsc_options=solver_parameters)

# ERROR #
boundaries_1 = where(boundaries.values == 1)[0]
bdofs_V_1 = locate_dofs_topological(V, mesh.topology.dim - 1, boundaries_1)
bc_ex = DirichletBC(g, bdofs_V_1)
u_ex = Function(V)
solve(a[0][0] == f[0], u_ex, bc_ex, petsc_options=solver_parameters)
u_ex.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
u_ex_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(u_ex), grad(u_ex))*dx)))
err_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(u_ex - ul[0]), grad(u_ex - ul[0]))*dx)))
print("Relative error is equal to", err_norm/u_ex_norm)
assert isclose(err_norm/u_ex_norm, 0., atol=1.e-10)

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
from ufl import *
from dolfin import *
from dolfin.cpp.mesh import GhostMode
from dolfin.fem import assemble_scalar
from multiphenics import *
from multiphenics.io import XDMFFile

"""
This tutorial is very similar to tutorial 3, as it concerns a weak imposition of
a Dirichlet boundary conditions through a Lagrange multiplier.
The main difference with respect to previous tutorials is in the mesh (and restrictions)
generation stage, available in the file data/generate_restrictions.py.
In this tutorial we use gmsh to generate a data/mesh.msh file starting from the geometry
described in data/mesh.geo, which represents a sphere embedded in a box.
Mesh and subdomain files, in xml format, can then be obtained
through the FEniCS script dolfin-convert. As dolfin-convert does not automatically provide
restrictions as output, two auxiliary methods are provided in data/generate_restrictions.py
to automatically generate subdomain restrictions and interface restrictions based on
subdomain ids. These restrictions are then employed in this script to solve a Poisson problem
on the sphere (with manufactured solution g), discarding degrees of freedom on its complement.
"""

# MESHES #
# Mesh
if MPI.size(MPI.comm_world) > 1:
    mesh_ghost_mode = GhostMode.shared_facet # shared_facet ghost mode is required by dS
else:
    mesh_ghost_mode = GhostMode.none
mesh = XDMFFile(MPI.comm_world, "data/mesh.xdmf").read_mesh(mesh_ghost_mode)
subdomains = XDMFFile(MPI.comm_world, "data/mesh_physical_region.xdmf").read_mf_size_t(mesh)
boundaries = XDMFFile(MPI.comm_world, "data/mesh_facet_region.xdmf").read_mf_size_t(mesh)
# Restrictions
sphere_restriction = XDMFFile(MPI.comm_world, "data/mesh_sphere_restriction.rtc.xdmf").read_mesh_restriction(mesh)
interface_restriction = XDMFFile(MPI.comm_world, "data/mesh_interface_restriction.rtc.xdmf").read_mesh_restriction(mesh)

# FUNCTION SPACES #
# Function space
V = FunctionSpace(mesh, ("Lagrange", 2))
# Block function space
W = BlockFunctionSpace([V, V], restrict=[sphere_restriction, interface_restriction])

# TRIAL/TEST FUNCTIONS #
ul = BlockTrialFunction(W)
(u, l) = block_split(ul)
vm = BlockTestFunction(W)
(v, m) = block_split(vm)

# MEASURES #
dx = Measure("dx")(subdomain_data=subdomains)
dS = Measure("dS")(subdomain_data=boundaries)

# ASSEMBLE #
x = SpatialCoordinate(mesh)
f = 27*sin(3*x[0])*sin(3*x[1])*sin(3*x[2])
g = sin(3*x[0])*sin(3*x[1])*sin(3*x[2])
a = [[inner(grad(u), grad(v))*dx(2), l("+")*v("+")*dS(1)],
     [u("+")*m("+")*dS(1)          , 0                  ]]
f =  [f*v*dx(2)                    , g("+")*m("+")*dS(1)]

# SOLVE #
u = BlockFunction(W)
solver_parameters = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
block_solve(a, u, f, petsc_options=solver_parameters)

# ERROR #
g_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(g, g)*dx(2, domain=mesh))))
err_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(u[0] - g, u[0] - g)*dx(2))))
print("Relative error is equal to", err_norm/g_norm)
assert isclose(err_norm/g_norm, 0., atol=1.e-4)

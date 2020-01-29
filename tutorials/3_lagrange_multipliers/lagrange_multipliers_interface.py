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

from numpy import isclose, where
from petsc4py import PETSc
from ufl import *
from dolfinx import *
from dolfinx.cpp.mesh import GhostMode
from dolfinx.fem import assemble_scalar
from multiphenics import *
from multiphenics.io import XDMFFile

r"""
In this example we solve
    - \Delta u = f      in \Omega
             u = 0      in \partial \Omega
using a domain decomposition approach for \Omega = \Omega_1 \cup \Omega_2,
and introducing a lagrange multiplier to handle the continuity of the solution across
the interface \Gamma between \Omega_1 and \Omega_2.

The resulting weak formulation is:
    find u_1 \in V(\Omega_1), u_2 \in V(\Omega_2), \eta \in E(\Gamma)
s.t.
    \int_{\Omega_1} grad(u_1) \cdot grad(v_1) \dx_1 +
    \int_{\Omega_2} grad(u_2) \cdot grad(v_2) \dx_2 +
    \int_{\Gamma} \lambda (v_1 - v_2) \ds = 0,
        \forall v_1 \in V(\Omega_1), v_2 \in V(\Omega_2)
and
    \int_{\Gamma} \eta  (u_1 - u_2) \ds = 0,
        \forall \eta \in E(\Gamma)
where boundary conditions on \partial\Omega are embedded in V(.)
"""

# MESHES #
# Mesh
if MPI.size(MPI.comm_world) > 1:
    mesh_ghost_mode = GhostMode.shared_facet # shared_facet ghost mode is required by dS
else:
    mesh_ghost_mode = GhostMode.none
mesh = XDMFFile(MPI.comm_world, "data/circle.xdmf").read_mesh(mesh_ghost_mode)
subdomains = XDMFFile(MPI.comm_world, "data/circle_physical_region.xdmf").read_mf_size_t(mesh)
boundaries = XDMFFile(MPI.comm_world, "data/circle_facet_region.xdmf").read_mf_size_t(mesh)
# Restrictions
left = XDMFFile(MPI.comm_world, "data/circle_restriction_left.rtc.xdmf").read_mesh_restriction(mesh)
right = XDMFFile(MPI.comm_world, "data/circle_restriction_right.rtc.xdmf").read_mesh_restriction(mesh)
interface = XDMFFile(MPI.comm_world, "data/circle_restriction_interface.rtc.xdmf").read_mesh_restriction(mesh)

# FUNCTION SPACES #
# Function space
V = FunctionSpace(mesh, ("Lagrange", 2))
# Block function space
W = BlockFunctionSpace([V, V, V], restrict=[left, right, interface])

# TRIAL/TEST FUNCTIONS #
u1u2l = BlockTrialFunction(W)
(u1, u2, l) = block_split(u1u2l)
v1v2m = BlockTestFunction(W)
(v1, v2, m) = block_split(v1v2m)

# MEASURES #
dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)
dS = Measure("dS")(subdomain_data=boundaries)
dS = dS(2) # restrict to the interface, which has facet ID equal to 2

# ASSEMBLE #
a = [[inner(grad(u1), grad(v1))*dx(1), 0                              ,   l("-")*v1("-")*dS ],
     [0                              , inner(grad(u2), grad(v2))*dx(2), - l("+")*v2("+")*dS ],
     [m("-")*u1("-")*dS              , - m("+")*u2("+")*dS            , 0                   ]]
f =  [v1*dx(1)                       , v2*dx(2)                       , 0                   ]

zero = Function(V)
boundaries_1 = where(boundaries.values == 1)[0]
bc1 = DirichletBC(W.sub(0), zero, boundaries_1)
bc2 = DirichletBC(W.sub(1), zero, boundaries_1)
bcs = BlockDirichletBC([bc1,
                        bc2,
                        None])

# SOLVE #
uu = BlockFunction(W)
solver_parameters = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
block_solve(a, uu, f, bcs, petsc_options=solver_parameters)

# ERROR #
u = TrialFunction(V)
v = TestFunction(V)
a_ex = inner(grad(u), grad(v))*dx
f_ex = v*dx
bc_ex = DirichletBC(V, zero, boundaries_1)
u_ex = Function(V)
solve(a_ex == f_ex, u_ex, bc_ex, petsc_options=solver_parameters)
u_ex.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
u_ex1_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(u_ex, u_ex)*dx(1))))
u_ex2_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(u_ex, u_ex)*dx(2))))
err1_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(u_ex - uu[0], u_ex - uu[0])*dx(1))))
err2_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(u_ex - uu[1], u_ex - uu[1])*dx(2))))
print("Relative error on subdomain 1", err1_norm/u_ex1_norm)
print("Relative error on subdomain 2", err2_norm/u_ex2_norm)
assert isclose(err1_norm/u_ex1_norm, 0., atol=1.e-10)
assert isclose(err2_norm/u_ex2_norm, 0., atol=1.e-10)

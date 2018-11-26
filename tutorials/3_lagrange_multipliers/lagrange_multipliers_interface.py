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
from dolfin import *
from dolfin import function
from dolfin.cpp.mesh import GhostMode
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
mesh = XDMFFile(MPI.comm_world, "data/circle.xdmf").read_mesh(MPI.comm_world, GhostMode.shared_facet) # shared_facet ghost mode is required by dS
subdomains = XDMFFile(MPI.comm_world, "data/circle_physical_region.xdmf").read_mf_size_t(mesh)
boundaries = XDMFFile(MPI.comm_world, "data/circle_facet_region.xdmf").read_mf_size_t(mesh)
# Restrictions
left = XDMFFile(MPI.comm_world, "data/circle_restriction_left.rtc.xdmf").read_mesh_restriction(mesh)
right = XDMFFile(MPI.comm_world, "data/circle_restriction_right.rtc.xdmf").read_mesh_restriction(mesh)
interface = XDMFFile(MPI.comm_world, "data/circle_restriction_interface.rtc.xdmf").read_mesh_restriction(mesh)

# FUNCTION SPACES #
# Function space
V = FunctionSpace(mesh, "Lagrange", 2)
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

@function.expression.numba_eval
def zero_eval(values, x, cell):
    values[:] = 0.0
zero = interpolate(Expression(zero_eval), V)
bc1 = DirichletBC(W.sub(0), zero, boundaries, 1)
bc2 = DirichletBC(W.sub(1), zero, boundaries, 1)
bcs = BlockDirichletBC([bc1,
                        bc2,
                        None])

# SOLVE #
u = BlockFunction(W)
solver_parameters = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
block_solve(a, u.block_vector(), f, bcs, petsc_options=solver_parameters)

# ERROR #
u = TrialFunction(V)
v = TestFunction(V)
A_ex = assemble(inner(grad(u), grad(v))*dx)
F_ex = assemble(v*dx)
bc_ex = DirichletBC(V, zero, boundaries, 1)
bc_ex.apply(A_ex)
bc_ex.apply(F_ex)
u_ex = Function(V)
solve(A_ex, u_ex.vector(), F_ex)
err1 = Function(V)
err1.vector().add_local(+ u_ex.vector().get_local())
err1.vector().add_local(- u[0].vector().get_local())
err1.vector().apply("")
err2 = Function(V)
err2.vector().add_local(+ u_ex.vector().get_local())
err2.vector().add_local(- u[1].vector().get_local())
err2.vector().apply("")
err1_norm = sqrt(assemble(err1*err1*dx(1))/assemble(u_ex*u_ex*dx(1)))
err2_norm = sqrt(assemble(err2*err2*dx(2))/assemble(u_ex*u_ex*dx(2)))
print("Relative error on subdomain 1", err1_norm)
print("Relative error on subdomain 2", err2_norm)
assert isclose(err1_norm, 0., atol=1.e-10)
assert isclose(err2_norm, 0., atol=1.e-10)

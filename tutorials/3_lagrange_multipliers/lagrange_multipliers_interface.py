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
parameters["ghost_mode"] = "shared_facet" # required by dS

"""
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
mesh = Mesh("data/circle.xml")
subdomains = MeshFunction("size_t", mesh, "data/circle_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/circle_facet_region.xml")
# Restrictions
left = MeshRestriction(mesh, "data/circle_restriction_left.rtc.xml")
right = MeshRestriction(mesh, "data/circle_restriction_right.rtc.xml")
interface = MeshRestriction(mesh, "data/circle_restriction_interface.rtc.xml")

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

bc1 = DirichletBC(W.sub(0), Constant(0.), boundaries, 1)
bc2 = DirichletBC(W.sub(1), Constant(0.), boundaries, 1)
bcs = BlockDirichletBC([bc1,
                        bc2,
                        None])

# SOLVE #
A = block_assemble(a)
F = block_assemble(f)
bcs.apply(A)
bcs.apply(F)

U = BlockFunction(W)
block_solve(A, U.block_vector(), F)

# plt.figure()
# plot(U[0])
# plt.figure()
# plot(U[1])
# plt.figure()
# plot(U[2])
# plt.show()

# ERROR #
u = TrialFunction(V)
v = TestFunction(V)
A_ex = assemble(inner(grad(u), grad(v))*dx)
F_ex = assemble(v*dx)
bc_ex = DirichletBC(V, Constant(0.), boundaries, 1)
bc_ex.apply(A_ex)
bc_ex.apply(F_ex)
U_ex = Function(V)
solve(A_ex, U_ex.vector(), F_ex)
# plt.figure()
# plot(U_ex)
# plt.show()
err1 = Function(V)
err1.vector().add_local(+ U_ex.vector().get_local())
err1.vector().add_local(- U[0].vector().get_local())
err1.vector().apply("")
err2 = Function(V)
err2.vector().add_local(+ U_ex.vector().get_local())
err2.vector().add_local(- U[1].vector().get_local())
err2.vector().apply("")
err1_norm = sqrt(assemble(err1*err1*dx(1))/assemble(U_ex*U_ex*dx(1)))
err2_norm = sqrt(assemble(err2*err2*dx(2))/assemble(U_ex*U_ex*dx(2)))
print("Relative error on subdomain 1", err1_norm)
print("Relative error on subdomain 2", err2_norm)
assert isclose(err1_norm, 0., atol=1.e-10)
assert isclose(err2_norm, 0., atol=1.e-10)

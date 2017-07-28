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

from dolfin import *
from multiphenics import *

"""
In this tutorial we solve the optimal control problem

min J(y, u) = 1/2 \int_{\Omega} (y - y_d)^2 dx + \alpha/2 \sum_{i=1}^3 \int_{\Gamma_2} u_i^2 ds
s.t.
      - \Delta y = f                        in \Omega
    \partial_n y = 0                        on \Gamma_1
               y = u1*l1 + u2*l2 + u3*l3    on \Gamma_2
    \partial_n y = 0                        on \Gamma_3
               y = 0                        on \Gamma_4
             
where
    \Omega                               unit square
    u1, u2, u3 in IR                     control variables
    l1, l2, l3 \in L^2(\Gamma_2)         known boundary profiles
    y \in H^1_0(\Omega)                  state variable
    \alpha > 0                           penalization parameter
    y_d                                  desired state
    f                                    forcing term
    
using an adjoint formulation solved by a one shot approach
"""

## MESH ##
# Mesh
mesh = Mesh("data/square.xml")
boundaries = MeshFunction("size_t", mesh, "data/square_facet_region.xml")
# Dirichlet boundary
left = MeshRestriction(mesh, "data/square_restriction_boundary_2.rtc.xml")

## FUNCTION SPACES ##
Y = FunctionSpace(mesh, "Lagrange", 2)
U = FunctionSpace(mesh, "R", 0)
L = FunctionSpace(mesh, "Lagrange", 2)
Q = Y
W = BlockFunctionSpace([Y, U, U, U, L, Q], restrict=[None, None, None, None, left, None])

## PROBLEM DATA ##
alpha = Constant(1.e-5)
y_d = Constant(1.)
f = Expression("10*sin(2*pi*x[0])*sin(2*pi*x[1])", element=W.sub(0).ufl_element())
l1 = Constant(1.)
l2 = Expression("sin(2*pi*x[1])", element=W.sub(0).ufl_element())
l3 = Expression("cos(2*pi*x[1])", element=W.sub(0).ufl_element())

## TRIAL/TEST FUNCTIONS ##
yulp = BlockTrialFunction(W)
(y, u1, u2, u3, l, p) = block_split(yulp)
zvmq = BlockTestFunction(W)
(z, v1, v2, v3, m, q) = block_split(zvmq)

## MEASURES ##
ds = Measure("ds")(subdomain_data=boundaries)

## OPTIMALITY CONDITIONS ##
alpha_u_v_ds = [[alpha*u1*v1*ds(2), 0                , 0                ],
                [0                , alpha*u2*v2*ds(2), 0                ],
                [0                , 0                , alpha*u3*v3*ds(2)]]
minus_l_v_ds = [[- l1*l*v1*ds(2)],
                [- l2*l*v2*ds(2)],
                [- l3*l*v3*ds(2)]]
minus_u_m_ds = [[- l1*u1*m*ds(2)  , - l2*u2*m*ds(2)  , - l3*u3*m*ds(2)]]
a = [[y*z*dx                    , 0           , l*z*ds(2)   , inner(grad(p), grad(z))*dx], 
     [0                         , alpha_u_v_ds, minus_l_v_ds, 0                         ],
     [y*m*ds(2)                 , minus_u_m_ds, 0           , 0                         ],
     [inner(grad(y), grad(q))*dx, 0           , 0           , 0                         ]]
f =  [y_d*z*dx,
      0       ,
      0       ,
      f*q*dx   ]
bc = BlockDirichletBC([[DirichletBC(W.sub(0), Constant(0.), boundaries, 4)],
                       [],
                       [],
                       [],
                       [],
                       [DirichletBC(W.sub(5), Constant(0.), boundaries, idx) for idx in (2, 4)]])

## SOLUTION ##
yulp = BlockFunction(W)
(y, u1, u2, u3, l, p) = block_split(yulp)

## FUNCTIONAL ##
J = 0.5*inner(y - y_d, y - y_d)*dx + 0.5*alpha*inner(u1, u1)*ds(2) + 0.5*alpha*inner(u2, u2)*ds(2) + 0.5*alpha*inner(u3, u3)*ds(2)

## UNCONTROLLED FUNCTIONAL VALUE ##
A_state = assemble(a[3][0])
F_state = assemble(f[3])
bc_state = [DirichletBC(W.sub(0), Constant(0.), boundaries, idx) for idx in (2, 4)]
[bc_state_.apply(A_state) for bc_state_ in bc_state]
[bc_state_.apply(F_state)  for bc_state_ in bc_state]
solve(A_state, y.vector(), F_state)
print "Uncontrolled J =", assemble(J)
plot(y, title="uncontrolled state")
interactive()

## OPTIMAL CONTROL ##
A = block_assemble(a)
F = block_assemble(f)
bc.apply(A)
bc.apply(F)
block_solve(A, yulp.block_vector(), F)
print "Optimal J =", assemble(J)
plot(y, title="state")
plot(l, title="lambda")
plot(p, title="adjoint")
print "u1", u1.vector().array()
print "u2", u2.vector().array()
print "u3", u3.vector().array()
interactive()


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
import matplotlib.pyplot as plt
from multiphenics import *

"""
In this tutorial we solve the optimal control problem

min J(y, u) = 1/2 \int_{\Omega} (y - y_d)^2 dx + \alpha/2 \int_{\Omega} u^2 dx
s.t.
    - \epsilon \Delta y + \beta \cdot \nabla y + \sigma y = f + u       in \Omega
                                                        y = g_D         on \partial \Omega
             
where
    \Omega                      unit square
    u \in L^2(\Omega)           control variable
    y \in H^1_0(\Omega)         state variable
    \alpha > 0                  penalization parameter
    y_d = piecewise constant    desired state
    \epsilon > 0                diffusion coefficient
    \beta in IR^2               advection field
    \sigma > 0                  reaction coefficient
    f                           forcing term
    g_D = piecewise constant    non homogeneous Dirichlet BC
    
using an adjoint formulation solved by a one shot approach
"""

## MESH ##
mesh = Mesh("data/graetz_1.xml")
subdomains = MeshFunction("size_t", mesh, "data/graetz_1_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/graetz_1_facet_region.xml")

## FUNCTION SPACES ##
Y = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
U = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Q = Y
W_el = BlockElement(Y, U, Q)
W = BlockFunctionSpace(mesh, W_el)

## PROBLEM DATA ##
alpha = Constant(0.01)
y_d_1 = Constant(0.6)
y_d_2 = Constant(1.8)
epsilon = Constant(1./15.)
beta = Expression(("x[1]*(1-x[1])", "0"), element=VectorElement(Y))
sigma = Constant(0.)
f = Constant(0.)

## TRIAL/TEST FUNCTIONS ##
yup = BlockTrialFunction(W)
(y, u, p) = block_split(yup)
zvq = BlockTestFunction(W)
(z, v, q) = block_split(zvq)

## MEASURES ##
dx = Measure("dx")(subdomain_data=subdomains)

## OPTIMALITY CONDITIONS ##
state_operator = epsilon*inner(grad(y), grad(q))*dx + inner(beta, grad(y))*q*dx + sigma*y*q*dx
adjoint_operator = epsilon*inner(grad(p), grad(z))*dx - inner(beta, grad(p))*z*dx + sigma*p*z*dx
a = [[y*z*(dx(1) + dx(2)), 0           , adjoint_operator], 
     [0                  , alpha*u*v*dx, - p*v*dx        ],
     [state_operator     , - u*q*dx    , 0               ]]
f =  [y_d_1*z*dx(1) + y_d_2*z*dx(2),
      0                            ,
      f*q*dx                        ]
bc = BlockDirichletBC([[DirichletBC(W.sub(0), Constant(idx), boundaries, idx) for idx in (1, 2)],
                       [],
                       [DirichletBC(W.sub(2), Constant(0. ), boundaries, idx) for idx in (1, 2)]])

## SOLUTION ##
yup = BlockFunction(W)
(y, u, p) = block_split(yup)

## FUNCTIONAL ##
J = 0.5*inner(y - y_d_1, y - y_d_1)*dx(1) + 0.5*inner(y - y_d_2, y - y_d_2)*dx(2) + 0.5*alpha*inner(u, u)*dx

## UNCONTROLLED FUNCTIONAL VALUE ##
A_state = assemble(a[2][0])
F_state = assemble(f[2])
[bc_state.apply(A_state) for bc_state in bc[0]]
[bc_state.apply(F_state)  for bc_state in bc[0]]
solve(A_state, y.vector(), F_state)
print "Uncontrolled J =", assemble(J)
plt.figure(); plot(y, title="uncontrolled state")
plt.show()

## OPTIMAL CONTROL ##
A = block_assemble(a, keep_diagonal=True)
F = block_assemble(f)
bc.apply(A)
bc.apply(F)
block_solve(A, yup.block_vector(), F)
print "Optimal J =", assemble(J)
plt.figure(); plot(y, title="state")
plt.figure(); plot(u, title="control")
plt.figure(); plot(p, title="adjoint")
plt.show()


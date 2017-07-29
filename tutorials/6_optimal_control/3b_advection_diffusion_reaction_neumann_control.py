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

min J(y, u) = 1/2 \int_{\Omega} (y - y_d)^2 dx + \alpha/2 \int_{\Gamma_2} u^2 ds
s.t.
       - \epsilon \Delta y + \beta \cdot \nabla y + \sigma y = f    in \Omega
                                                           y = 1    on \Gamma_1
                                       \epsilon \partial_n y = u    on \Gamma_2
                                       \epsilon \partial_n y = 0    on \Gamma_3
             
where
    \Omega                      unit square
    u \in L^2(\Gamma_2)         control variable
    y \in H^1_0(\Omega)         state variable
    \alpha > 0                  penalization parameter
    y_d = piecewise constant    desired state
    \epsilon > 0                diffusion coefficient
    \beta in IR^2               advection field
    \sigma > 0                  reaction coefficient
    f                           forcing term
    
using an adjoint formulation solved by a one shot approach
"""

## MESH ##
# Mesh
mesh = Mesh("data/graetz_2.xml")
subdomains = MeshFunction("size_t", mesh, "data/graetz_2_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/graetz_2_facet_region.xml")
# Neumann boundary
control_boundary = MeshRestriction(mesh, "data/graetz_2_restriction_control.rtc.xml")

## FUNCTION SPACES ##
Y = FunctionSpace(mesh, "Lagrange", 2)
U = FunctionSpace(mesh, "Lagrange", 2)
Q = Y
W = BlockFunctionSpace([Y, U, Q], restrict=[None, control_boundary, None])

## PROBLEM DATA ##
alpha = Constant(0.07)
y_d = Constant(2.5)
epsilon = Constant(1./12.)
beta = Expression(("x[1]*(1-x[1])", "0"), element=VectorElement(Y.ufl_element()))
sigma = Constant(0.)
f = Constant(0.)

## TRIAL/TEST FUNCTIONS ##
yup = BlockTrialFunction(W)
(y, u, p) = block_split(yup)
zvq = BlockTestFunction(W)
(z, v, q) = block_split(zvq)

## MEASURES ##
dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)

## OPTIMALITY CONDITIONS ##
state_operator = epsilon*inner(grad(y), grad(q))*dx + inner(beta, grad(y))*q*dx + sigma*y*q*dx
adjoint_operator = epsilon*inner(grad(p), grad(z))*dx - inner(beta, grad(p))*z*dx + sigma*p*z*dx
a = [[y*z*dx(3)     , 0              , adjoint_operator], 
     [0             , alpha*u*v*ds(2), - p*v*ds(2)     ],
     [state_operator, - u*q*ds(2)    , 0               ]]
f =  [y_d*z*dx(3),
      0          ,
      f*q*dx      ]
bc = BlockDirichletBC([[DirichletBC(W.sub(0), Constant(1.), boundaries, 1)],
                       [],
                       [DirichletBC(W.sub(2), Constant(0.), boundaries, 1)]])

## SOLUTION ##
yup = BlockFunction(W)
(y, u, p) = block_split(yup)

## FUNCTIONAL ##
J = 0.5*inner(y - y_d, y - y_d)*dx(3) + 0.5*alpha*inner(u, u)*ds(2)

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


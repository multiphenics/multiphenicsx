# Copyright (C) 2016-2019 by the multiphenics authors
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
import matplotlib.pyplot as plt
from multiphenics import *

r"""
In this tutorial we solve the optimal control problem

min J(y, u) = 1/2 \int_{\Omega} (y - y_d)^2 dx + \alpha/2 \int_{\Omega} u^2 dx
s.t.
    - \Delta y = f + u   in \Omega
             y = 0       on \partial \Omega
             
where
    \Omega                      unit square
    u \in L^2(\Omega)           control variable
    y \in H^1_0(\Omega)         state variable
    \alpha > 0                  penalization parameter
    y_d                         desired state
    f                           forcing term
    
using an adjoint formulation solved by a one shot approach
"""

# MESH #
mesh = Mesh("data/square.xml")
boundaries = MeshFunction("size_t", mesh, "data/square_facet_region.xml")

# FUNCTION SPACES #
Y = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
U = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Q = Y
W_el = BlockElement(Y, U, Q)
W = BlockFunctionSpace(mesh, W_el)

# PROBLEM DATA #
alpha = Constant(1.e-5)
y_d = Expression("10*x[0]*(1-x[0])*x[1]*(1-x[1])", element=W.sub(0).ufl_element())
f = Constant(0.)

# TRIAL/TEST FUNCTIONS #
yup = BlockTrialFunction(W)
(y, u, p) = block_split(yup)
zvq = BlockTestFunction(W)
(z, v, q) = block_split(zvq)

# OPTIMALITY CONDITIONS #
a = [[y*z*dx                    , 0           , inner(grad(p), grad(z))*dx],
     [0                         , alpha*u*v*dx, - p*v*dx                  ],
     [inner(grad(y), grad(q))*dx, - u*q*dx    , 0                         ]]
f =  [y_d*z*dx,
      0       ,
      f*q*dx   ]
bc = BlockDirichletBC([[DirichletBC(W.sub(0), Constant(0.), boundaries, idx) for idx in (1, 2, 3, 4)],
                       [],
                       [DirichletBC(W.sub(2), Constant(0.), boundaries, idx) for idx in (1, 2, 3, 4)]])

# SOLUTION #
yup = BlockFunction(W)
(y, u, p) = block_split(yup)

# FUNCTIONAL #
J = 0.5*inner(y - y_d, y - y_d)*dx + 0.5*alpha*inner(u, u)*dx

# UNCONTROLLED FUNCTIONAL VALUE #
A_state = assemble(a[2][0])
F_state = assemble(f[2])
[bc_state.apply(A_state) for bc_state in bc[0]]
[bc_state.apply(F_state) for bc_state in bc[0]]
solve(A_state, y.vector(), F_state)
print("Uncontrolled J =", assemble(J))
assert isclose(assemble(J), 0.055411611)
plt.figure()
plot(y, title="uncontrolled state")
plt.show()

# OPTIMAL CONTROL #
A = block_assemble(a, keep_diagonal=True)
F = block_assemble(f)
bc.apply(A)
bc.apply(F)
block_solve(A, yup.block_vector(), F)
print("Optimal J =", assemble(J))
assert isclose(assemble(J), 0.0002322147)
plt.figure()
plot(y, title="state")
plt.figure()
plot(u, title="control")
plt.figure()
plot(p, title="adjoint")
plt.show()

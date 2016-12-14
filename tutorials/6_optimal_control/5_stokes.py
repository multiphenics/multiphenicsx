# Copyright (C) 2016 by the block_ext authors
#
# This file is part of block_ext.
#
# block_ext is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# block_ext is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with block_ext. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import *
from block_ext import *
from sympy import ccode, cos, symbols

"""
In this tutorial we solve the optimal control problem

min J(y, u) = 1/2 \int_{\Omega} (v - v_d)^2 dx + \alpha/2 \int_{\Omega} u^2 dx
s.t.
    - \Delta v + \nabla p = f + u   in \Omega
                    div v = 0       in \Omega
                        v = 0       on \partial\Omega
             
where
    \Omega                      unit square
    u \in [L^2(\Omega)]^2       control variable
    v \in [H^1_0(\Omega)]^2     state velocity variable
    p \in L^2(\Omega)           state pressure variable
    \alpha > 0                  penalization parameter
    v_d                         desired state
    f                           forcing term
    
using an adjoint formulation solved by a one shot approach
"""

## MESH ##
mesh = Mesh("data/square.xml")
boundaries = MeshFunction("size_t", mesh, "data/square_facet_region.xml")

## FUNCTION SPACES ##
Y_velocity = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Y_pressure = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
U = VectorElement("Lagrange", mesh.ufl_cell(), 2)
L = FiniteElement("R", mesh.ufl_cell(), 0)
Q_velocity = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q_pressure = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W_el = BlockElement(Y_velocity, Y_pressure, U, L, Q_velocity, Q_pressure)
W = BlockFunctionSpace(mesh, W_el)

## PROBLEM DATA ##
alpha = Constant(1.e-5)
x, y = symbols('x[0], x[1]')
psi_d = 10*(1-cos(0.8*pi*x))*(1-cos(0.8*pi*y))*(1-x)**2*(1-y)**2
v_d = Expression((ccode(psi_d.diff(y, 1)), ccode(-psi_d.diff(x, 1))), element=W.sub(0).ufl_element())
f = Constant((0., 0.))

## TRIAL/TEST FUNCTIONS ##
trial = BlockTrialFunction(W)
(v, p, u, l, z, b) = block_split(trial)
test = BlockTestFunction(W)
(w, q, r, m, s, d) = block_split(test)

## OPTIMALITY CONDITIONS ##
a = [[inner(v, s)*dx            , 0            , 0                   , 0     , inner(grad(z), grad(s))*dx, - b*div(s)*dx], 
     [0                         , 0            , 0                   , l*d*dx, - d*div(z)*dx             , 0            ],
     [0                         , 0            , alpha*inner(u, r)*dx, 0     , - inner(z, r)*dx          , 0            ],
     [0                         , p*m*dx       , 0                   , 0     , 0                         , 0            ],
     [inner(grad(v), grad(w))*dx, - p*div(w)*dx, - inner(u, w)*dx    , 0     , 0                         , 0            ],
     [- q*div(v)*dx             , 0            , 0                   , 0     , 0                         , 0            ]]
f =  [inner(v_d, s)*dx,
      0               ,
      0               ,
      0               ,
      inner(f, w)*dx  ,
      0                ]
bc = BlockDirichletBC([[DirichletBC(W.sub(0), Constant((0., 0.)), boundaries, idx) for idx in (1, 2, 3, 4)],
                       [],
                       [],
                       [],
                       [DirichletBC(W.sub(4), Constant((0., 0.)), boundaries, idx) for idx in (1, 2, 3, 4)],
                       []])

## SOLUTION ##
solution = BlockFunction(W)
(v, p, u, l, z, b) = block_split(solution)

## FUNCTIONAL ##
J = 0.5*inner(v - v_d, v - v_d)*dx + 0.5*alpha*inner(u, u)*dx

## UNCONTROLLED FUNCTIONAL VALUE ##
A_state = block_assemble([[a[4][0], a[4][1]],
                          [a[5][0], a[5][1]]])
F_state = block_assemble( [f[4],
                           f[5]])
bc_state = BlockDirichletBC([bc[4],
                             bc[5]])
bc_state.apply(A_state)
bc_state.apply(F_state)
solution_state = BlockFunction([v, p])
block_solve(A_state, solution_state.block_vector(), F_state)
print "Uncontrolled J =", assemble(J)
plot(v, title="uncontrolled state velocity")
plot(p, title="uncontrolled state pressure")
interactive()

## OPTIMAL CONTROL ##
A = block_assemble(a)
F = block_assemble(f)
bc.apply(A)
bc.apply(F)
block_solve(A, solution.block_vector(), F)
print "Optimal J =", assemble(J)
plot(v, title="state velocity")
plot(p, title="state pressure")
plot(u, title="control")
plot(l, title="lambda")
plot(z, title="adjoint velocity")
plot(b, title="adjoint pressure")
interactive()


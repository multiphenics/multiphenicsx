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
from sympy import ccode, cos, symbols

"""
In this tutorial we solve the optimal control problem

min J(y, u) = 1/2 \int_{\Omega} |curl v|^2 dx + \alpha/2 \int_{\Gamma_C} |\partial_t u|^2 ds
s.t.
- \nu \Delta v + \nabla p = f       in \Omega
                    div v = 0       in \Omega
                        v = 2.5 t   on \Gamma_{in} [id 1]
                        v = 0       on \Gamma_{w}  [id 5]
                v \cdot n = u       on \Gamma_{C}  [id 4]
                v \cdot t = 0       on \Gamma_{C}  [id 4]
                v \cdot n = 0       on \Gamma_{s}  [id 2]
 \nu \partial_n v \cdot t = 0       on \Gamma_{s}  [id 2]
   p n - \nu \partial_n v = 0       on \Gamma_{N}  [id 3]
       
             
where
    \Omega                      unit square
    u \in [L^2(Gamma_C)]^2      control variable
    v \in [H^1_0(\Omega)]^2     state velocity variable
    p \in L^2(\Omega)           state pressure variable
    \alpha > 0                  penalization parameter
    f                           forcing term
    
using an adjoint formulation solved by a one shot approach
"""

## MESH ##
# Mesh
mesh = Mesh("data/vorticity_reduction.xml")
subdomains = MeshFunction("size_t", mesh, "data/vorticity_reduction_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/vorticity_reduction_facet_region.xml")
# Dirichlet control boundary
control_boundary = MeshRestriction(mesh, "data/vorticity_reduction_restriction_control.rtc")
# Normal and tangent
n = FacetNormal(mesh)
t = as_vector([n[1], -n[0]])

## FUNCTION SPACES ##
Y_velocity = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Y_pressure = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
U = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
L = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
Q_velocity = Y_velocity
Q_pressure = Y_pressure
W_el = BlockElement(Y_velocity, Y_pressure, U, L, Q_velocity, Q_pressure)
W = BlockFunctionSpace(mesh, W_el, restrict=[None, None, control_boundary, control_boundary, None, None])

## PROBLEM DATA ##
nu = Constant(1.)
alpha = Constant(1.e-2)
f = Constant((0., 0.))

## TRIAL/TEST FUNCTIONS ##
trial = BlockTrialFunction(W)
(v, p, u, l, z, b) = block_split(trial)
test = BlockTestFunction(W)
(w, q, r, m, s, d) = block_split(test)

## MEASURES ##
dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)

## OPTIMALITY CONDITIONS ##
def vorticity(v, w):
    return inner(curl(v), curl(w))
def penalty(u, r):
    return alpha*inner(grad(u), t)*inner(grad(r), t)
a = [[vorticity(v, w)*dx(4)        , 0            , 0                  , l*inner(w, n)*ds(4), nu*inner(grad(z), grad(w))*dx, - b*div(w)*dx], 
     [0                            , 0            , 0                  , 0                  , - q*div(z)*dx                , 0            ],
     [0                            , 0            , penalty(u, r)*ds(4), -l*r*ds(4)         , 0                            , 0            ],
     [m*inner(v, n)*ds(4)          , 0            , -m*u*ds(4)         , 0                  , 0                            , 0            ],
     [nu*inner(grad(v), grad(s))*dx, - p*div(s)*dx, 0                  , 0                  , 0                            , 0            ],
     [- d*div(v)*dx                , 0            , 0                  , 0                  , 0                            , 0            ]]
f =  [0,
      0               ,
      0               ,
      0               ,
      inner(f, s)*dx  ,
      0                ]
bc = BlockDirichletBC([[
                            DirichletBC(W.sub(0), Constant((2.5, 0.0)), boundaries, 1),
                            DirichletBC(W.sub(0).sub(1), Constant(0.0), boundaries, 2),
                            DirichletBC(W.sub(0).sub(0), Constant(0.0), boundaries, 4),
                            DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 5), 
                        ],
                       [],
                       [],
                       [],
                       [
                            DirichletBC(W.sub(4), Constant((0.0, 0.0)), boundaries, 1),
                            DirichletBC(W.sub(4).sub(1), Constant(0.0), boundaries, 2),
                            DirichletBC(W.sub(4), Constant((0.0, 0.0)), boundaries, 4),
                            DirichletBC(W.sub(4), Constant((0.0, 0.0)), boundaries, 5), 
                        ],
                       []])

## SOLUTION ##
solution = BlockFunction(W)
(v, p, u, l, z, b) = block_split(solution)

## FUNCTIONAL ##
J = 0.5*vorticity(v, v)*dx(4) + 0.5*penalty(u, u)*ds(4)

## UNCONTROLLED FUNCTIONAL VALUE ##
W_state_trial = W.extract_block_sub_space((0, 1))
W_state_test  = W.extract_block_sub_space((4, 5))
a_state = block_restrict(a, [W_state_test, W_state_trial])
A_state = block_assemble(a_state)
f_state = block_restrict(f, W_state_test)
F_state = block_assemble(f_state)
bc_state = BlockDirichletBC([[
                                  DirichletBC(W_state_trial.sub(0), Constant((2.5, 0.0)), boundaries, 1),
                                  DirichletBC(W_state_trial.sub(0).sub(1), Constant(0.0), boundaries, 2),
                                  DirichletBC(W_state_trial.sub(0), Constant((0.0, 0.0)), boundaries, 4),
                                  DirichletBC(W_state_trial.sub(0), Constant((0.0, 0.0)), boundaries, 5), 
                              ],
                             []])
bc_state.apply(A_state)
bc_state.apply(F_state)
solution_state = block_restrict(solution, W_state_trial)
block_solve(A_state, solution_state.block_vector(), F_state)
print "Uncontrolled J =", assemble(J)
plot(v, title="uncontrolled state velocity")
plot(p, title="uncontrolled state pressure")
interactive()

## OPTIMAL CONTROL ##
A = block_assemble(a, keep_diagonal=True)
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


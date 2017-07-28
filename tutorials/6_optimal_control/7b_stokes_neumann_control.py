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

min J(y, u) = 1/2 \int_{\Omega} (v - v_d)^2 dx + \alpha1/2 \int_{\Gamma_C [id 4]} |\partial_t u|^2 ds + \alpha2/2 \int_{\Gamma_C [id 4]} |u|^2 ds
s.t.
- \nu \Delta v + \nabla p = f       in \Omega
                    div v = 0       in \Omega
                        v = g       on \Gamma_{in} [id 1]
                        v = 0       on \Gamma_{w}  [id 2]
   p n - \nu \partial_n v = u       on \Gamma_{N}  [id 3]
       
             
where
    \Omega                      unit square
    u \in [L^2(Gamma_C)]^2      control variable
    v \in [H^1_0(\Omega)]^2     state velocity variable
    p \in L^2(\Omega)           state pressure variable
    \alpha1, \alpha2 > 0        penalization parameters
    v_d                         desired state
    f                           forcing term
    g                           inlet profile
    
using an adjoint formulation solved by a one shot approach
"""

## MESH ##
# Mesh
mesh = Mesh("data/bifurcation.xml")
subdomains = MeshFunction("size_t", mesh, "data/bifurcation_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/bifurcation_facet_region.xml")
# Neumann control boundary
control_boundary = MeshRestriction(mesh, "data/bifurcation_restriction_control.rtc.xml")
# Normal and tangent
n = FacetNormal(mesh)
t = as_vector([n[1], -n[0]])

## FUNCTION SPACES ##
Y_velocity = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Y_pressure = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
U = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q_velocity = Y_velocity
Q_pressure = Y_pressure
W_el = BlockElement(Y_velocity, Y_pressure, U, Q_velocity, Q_pressure)
W = BlockFunctionSpace(mesh, W_el, restrict=[None, None, control_boundary, None, None])

## PROBLEM DATA ##
nu = Constant(0.04)
alpha_1 = Constant(0.001)
alpha_2 = 0.1*alpha_1
g = Expression(("10.0*a*(x[1] + 1.0)*(1.0 - x[1])", "0.0"), a=1.0, element=W.sub(0).ufl_element())
v_d = Expression(("a*(b*10.0*(pow(x[1], 3) - pow(x[1], 2) - x[1] + 1.0)) + ((1.0-b)*10.0*(-pow(x[1], 3) - pow(x[1], 2) + x[1] + 1.0))", "0.0"), a=1.0, b=0.8, element=W.sub(0).ufl_element())
f = Constant((0., 0.))

## TRIAL/TEST FUNCTIONS ##
trial = BlockTrialFunction(W)
(v, p, u, z, b) = block_split(trial)
test = BlockTestFunction(W)
(w, q, r, s, d) = block_split(test)

## MEASURES ##
dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)
dS = Measure("dS")(subdomain_data=boundaries)

## OPTIMALITY CONDITIONS ##
def tracking(v, w):
    return inner(v, w)('-')
def penalty(u, r):
    return alpha_1*inner(grad(u)*t, grad(r)*t) + alpha_2*inner(u, r)
a = [[tracking(v, w)*dS(4)         , 0            , 0                  , nu*inner(grad(z), grad(w))*dx, - b*div(w)*dx], 
     [0                            , 0            , 0                  , - q*div(z)*dx                , 0            ],
     [0                            , 0            , penalty(u, r)*ds(3), - inner(z, r)*ds(3)          , 0            ],
     [nu*inner(grad(v), grad(s))*dx, - p*div(s)*dx, - inner(u, s)*ds(3), 0                            , 0            ],
     [- d*div(v)*dx                , 0            , 0                  , 0                            , 0            ]]
f =  [tracking(v_d, w)*dS(4),
      0                     ,
      0                     ,
      inner(f, s)*dx        ,
      0                      ]
bc = BlockDirichletBC([[DirichletBC(W.sub(0), g, boundaries, 1), DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 2)],
                       [],
                       [],
                       [DirichletBC(W.sub(3), Constant((0.0, 0.0)), boundaries, idx) for idx in (1, 2)],
                       []])

## SOLUTION ##
solution = BlockFunction(W)
(v, p, u, z, b) = block_split(solution)

## FUNCTIONAL ##
J = 0.5*tracking(v - v_d, v - v_d)*dS(4) + 0.5*penalty(u, u)*ds(3)

## UNCONTROLLED FUNCTIONAL VALUE ##
W_state_trial = W.extract_block_sub_space((0, 1))
W_state_test  = W.extract_block_sub_space((3, 4))
a_state = block_restrict(a, [W_state_test, W_state_trial])
A_state = block_assemble(a_state)
f_state = block_restrict(f, W_state_test)
F_state = block_assemble(f_state)
bc_state = block_restrict(bc, W_state_trial)
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
plot(z, title="adjoint velocity")
plot(b, title="adjoint pressure")
interactive()


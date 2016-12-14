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

min J(y, u) = 1/2 \int_{\Omega} (v - v_d)^2 dx + \alpha/2 \int_{\Gamma_2} u^2 dx
s.t.
    - \nu \Delta v + v \cdot \nabla v + \nabla p = f       in \Omega
                                           div v = 0       in \Omega
                                               v = 0       on \Gamma_1
                           pn - \nu \partial_n v = u       on \Gamma_2
                                               v = 0       on \Gamma_3
                                               v = 0       on \Gamma_4
             
where
    \Omega                      unit square
    u \in [L^2(\Omega)]^2       control variable
    v \in [H^1_0(\Omega)]^2     state velocity variable
    p \in L^2(\Omega)           state pressure variable
    \alpha > 0                  penalization parameter
    v_d                         desired state
    \nu                         kinematic viscosity
    f                           forcing term
    
using an adjoint formulation solved by a one shot approach
"""

## MESH ##
# Interior mesh
mesh = Mesh("data/square.xml")
boundaries = MeshFunction("size_t", mesh, "data/square_facet_region.xml")
# Neumann boundary mesh
boundary_mesh = Mesh("data/boundary_square_2.xml")

## FUNCTION SPACES ##
# Interior spaces
Y_velocity = VectorFunctionSpace(mesh, "Lagrange", 2)
Y_pressure = FunctionSpace(mesh, "Lagrange", 1)
U = VectorFunctionSpace(mesh, "Lagrange", 2)
Q_velocity = VectorFunctionSpace(mesh, "Lagrange", 2)
Q_pressure = FunctionSpace(mesh, "Lagrange", 1)
# Boundary control space
boundary_U = VectorFunctionSpace(boundary_mesh, "Lagrange", 2)
# Block space
W = BlockFunctionSpace([Y_velocity, Y_pressure, U, Q_velocity, Q_pressure], keep=[Y_velocity, Y_pressure, boundary_U, Q_velocity, Q_pressure])

## PROBLEM DATA ##
alpha = Constant(1.e-5)
x, y = symbols("x[0], x[1]")
psi_d = 10*(1-cos(0.8*pi*x))*(1-cos(0.8*pi*y))*(1-x)**2*(1-y)**2
v_d = Expression((ccode(psi_d.diff(y, 1)), ccode(-psi_d.diff(x, 1))), element=W.sub(0).ufl_element())
nu = Constant(0.1)
f = Constant((0., 0.))

## NONLINEAR SOLVER PARAMETERS ##
snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "error_on_nonconvergence": True}}

## TRIAL/TEST FUNCTIONS AND SOLUTION ##
trial = BlockTrialFunction(W)
solution = BlockFunction(W)
(v, p, u, z, b) = block_split(solution)
test = BlockTestFunction(W)
(w, q, r, s, d) = block_split(test)

## MEASURES ##
ds = Measure("ds")(subdomain_data=boundaries)

## OPTIMALITY CONDITIONS  ##
r =  [nu*inner(grad(z), grad(s))*dx + inner(grad(s)*v, z)*dx + inner(grad(v)*s, z)*dx - b*div(s)*dx + inner(v - v_d, s)*dx,
      - d*div(z)*dx                                                                                                       ,
      alpha*inner(u, r)*ds(2) - inner(z, r)*ds(2)                                                                         ,
      nu*inner(grad(v), grad(w))*dx + inner(grad(v)*v, w)*dx - p*div(w)*dx - inner(f, w)*dx - inner(u, w)*ds(2)           ,
      - q*div(v)*dx                                                                                                        ]
dr = block_derivative(r, solution, trial)
bc = BlockDirichletBC([[DirichletBC(W.sub(0), Constant((0., 0.)), boundaries, idx) for idx in (1, 3, 4)],
                       [],
                       [],
                       [DirichletBC(W.sub(3), Constant((0., 0.)), boundaries, idx) for idx in (1, 3, 4)],
                       []])


## FUNCTIONAL ##
J = 0.5*inner(v - v_d, v - v_d)*dx + 0.5*alpha*inner(u, u)*ds(2)

## UNCONTROLLED FUNCTIONAL VALUE ##
r_state = [r[3],
           r[4]]
dr_state = [[dr[3, 0], dr[3, 1]],
            [dr[4, 0], dr[4, 1]]]
bc_state = BlockDirichletBC([bc[3],
                             bc[4]])
solution_state = BlockFunction([v, p])
problem_state = BlockNonlinearProblem(r_state, solution_state, bc_state, dr_state)
solver_state = BlockPETScSNESSolver(problem_state)
solver_state.parameters.update(snes_solver_parameters["snes_solver"])
solver_state.solve()
print "Uncontrolled J =", assemble(J)
plot(v, title="uncontrolled state velocity")
plot(p, title="uncontrolled state pressure")
interactive()

## OPTIMAL CONTROL ##
problem = BlockNonlinearProblem(r, solution, bc, dr)
solver = BlockPETScSNESSolver(problem)
solver.parameters.update(snes_solver_parameters["snes_solver"])
solver.solve()
print "Optimal J =", assemble(J)
plot(v, title="state velocity")
plot(p, title="state pressure")
plot(u, title="control")
plot(z, title="adjoint velocity")
plot(b, title="adjoint pressure")
interactive()


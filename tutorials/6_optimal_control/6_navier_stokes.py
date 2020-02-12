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

from numpy import isclose, isin, stack, where
from ufl import *
from dolfinx import *
from dolfinx.cpp.mesh import GhostMode
from dolfinx.fem import assemble_scalar, locate_dofs_topological
from dolfinx.io import XDMFFile
from dolfinx.plotting import plot
import matplotlib.pyplot as plt
from multiphenics import *
from sympy import cos, lambdify, symbols

r"""
In this tutorial we solve the optimal control problem

min J(y, u) = 1/2 \int_{\Omega} |v - v_d|^2 dx + \alpha/2 \int_{\Omega} |u|^2 dx
s.t.
    - \nu \Delta v + v \cdot \nabla v + \nabla p = f + u   in \Omega
                                           div v = 0       in \Omega
                                               v = 0       on \partial\Omega
             
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

# MESH #
mesh = XDMFFile(MPI.comm_world, "data/square.xdmf").read_mesh(GhostMode.none)
boundaries = XDMFFile(MPI.comm_world, "data/square_facet_region.xdmf").read_mf_size_t(mesh)
boundaries_1234 = where(isin(boundaries.values, (1, 2, 3, 4)))[0]

# FUNCTION SPACES #
Y_velocity = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Y_pressure = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
U = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q_velocity = Y_velocity
Q_pressure = Y_pressure
W_el = BlockElement(Y_velocity, Y_pressure, U, Q_velocity, Q_pressure)
W = BlockFunctionSpace(mesh, W_el)

# PROBLEM DATA #
alpha = 1.e-5
epsilon = 1.e-5
x, y = symbols("x[0], x[1]")
psi_d = 10*(1-cos(0.8*pi*x))*(1-cos(0.8*pi*y))*(1-x)**2*(1-y)**2
v_d_x = lambdify([x, y], psi_d.diff(y, 1))
v_d_y = lambdify([x, y], -psi_d.diff(x, 1))
v_d = Function(W.sub(0))
v_d.interpolate(lambda x: stack((v_d_x(x[0], x[1]), v_d_y(x[0], x[1])), axis=0))
nu = 0.01
f = Constant(mesh, (0., 0.))
bc0 = Function(W.sub(0))

# TRIAL/TEST FUNCTIONS AND SOLUTION #
trial = BlockTrialFunction(W)
solution = BlockFunction(W)
(v, p, u, z, b) = block_split(solution)
test = BlockTestFunction(W)
(w, q, r, s, d) = block_split(test)

# OPTIMALITY CONDITIONS  #
r =  [nu*inner(grad(z), grad(w))*dx + inner(grad(w)*v, z)*dx + inner(grad(v)*w, z)*dx - b*div(w)*dx + inner(v - v_d, w)*dx,
      - q*div(z)*dx + epsilon*b*q*dx                                                                                      ,
      alpha*inner(u, r)*dx - inner(z, r)*dx                                                                               ,
      nu*inner(grad(v), grad(s))*dx + inner(grad(v)*v, s)*dx - p*div(s)*dx - inner(u + f, s)*dx                           ,
      - d*div(v)*dx + epsilon*p*d*dx                                                                                       ]
dr = block_derivative(r, solution, trial)
bdofs_W0_1234 = locate_dofs_topological((W.sub(0), W.sub(0)), mesh.topology.dim - 1, boundaries_1234)
bdofs_W3_1234 = locate_dofs_topological((W.sub(3), W.sub(0)), mesh.topology.dim - 1, boundaries_1234)
bc = BlockDirichletBC([[DirichletBC(bc0, bdofs_W0_1234, W.sub(0))],
                       [],
                       [],
                       [DirichletBC(bc0, bdofs_W3_1234, W.sub(3))],
                       []])


# FUNCTIONAL #
J = 0.5*inner(v - v_d, v - v_d)*dx + 0.5*alpha*inner(u, u)*dx

# UNCONTROLLED FUNCTIONAL VALUE #
W_state_trial = W.extract_block_sub_space((0, 1))
W_state_test = W.extract_block_sub_space((3, 4))
r_state = block_restrict(r, W_state_test)
dr_state = block_restrict(dr, [W_state_test, W_state_trial])
bc_state = block_restrict(bc, W_state_trial)
solution_state = block_restrict(solution, W_state_trial)
problem_state = BlockNonlinearProblem(r_state, solution_state, bc_state, dr_state)
solver_state = BlockNewtonSolver(mesh.mpi_comm())
solver_state.max_it = 20
solver_state.solve(problem_state, solution_state.block_vector)
J_uncontrolled = MPI.sum(mesh.mpi_comm(), assemble_scalar(J))
print("Uncontrolled J =", J_uncontrolled)
assert isclose(J_uncontrolled, 0.1784542)
plt.figure()
plot(v, title="uncontrolled state velocity")
plt.figure()
plot(p, title="uncontrolled state pressure")
plt.show()

# OPTIMAL CONTROL #
problem = BlockNonlinearProblem(r, solution, bc, dr)
solver = BlockNewtonSolver(mesh.mpi_comm())
solver.max_it = 20
solver.solve(problem, solution.block_vector)
J_controlled = MPI.sum(mesh.mpi_comm(), assemble_scalar(J))
print("Optimal J =", J_controlled)
assert isclose(J_controlled, 9.9523191e-7)
plt.figure()
plot(v, title="state velocity")
plt.figure()
plot(p, title="state pressure")
plt.figure()
plot(u, title="control")
plt.figure()
plot(z, title="adjoint velocity")
plt.figure()
plot(b, title="adjoint pressure")
plt.show()

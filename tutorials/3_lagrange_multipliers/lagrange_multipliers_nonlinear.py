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
from mshr import *
from block_ext import *

"""
In this example we solve a nonlinear Laplace problem associated to
    min E(u)
    s.t. u = g on \partial \Omega
where
    E(u) = \int_\Omega { (1 + u^2) |grad u|^2 - u } dx
using a Lagrange multiplier to handle non-homogeneous Dirichlet boundary conditions. 
"""

snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "error_on_nonconvergence": False}}

## MESHES ##
# Create interior mesh
domain = Circle(Point(0., 0.), 3.)
mesh = generate_mesh(domain, 15)
# Create boundary mesh
boundary_mesh = BoundaryMesh(mesh, "exterior")

## FUNCTION SPACES ##
# Interior space
V = FunctionSpace(mesh, "Lagrange", 2)
# Boundary space
boundary_V = FunctionSpace(boundary_mesh, "Lagrange", 2)
# For block problem definition
W = BlockFunctionSpace([V, V], keep=[V, boundary_V])

## TRIAL/TEST FUNCTIONS ##
dul = BlockTrialFunction(W)
(du, dl) = block_split(dul)
ul = BlockFunction(W)
(u, l) = block_split(ul)
vm = BlockTestFunction(W)
(v, m) = block_split(vm)

## MEASURES ##
dx = Measure("dx")(domain=mesh)
ds = Measure("ds")(domain=mesh)

## ASSEMBLE ##
g = Expression("sin(3*x[0] + 1)*sin(3*x[1] + 1)", element=V.ufl_element())
F = [inner((1+u**2)*grad(u),grad(v))*dx + u*v*inner(grad(u),grad(u))*dx + l*v*ds - v*dx, 
     u*m*ds - g*m*ds]
J = block_derivative(F, ul, dul)
bc = BlockDirichletBC([[], []])

## SOLVE ##
problem = BlockNonlinearProblem(F, ul, bc, J)
solver = BlockPETScSNESSolver(problem)
solver.parameters.update(snes_solver_parameters["snes_solver"])
solver.solve()

z = Function(V, assemble(F[0]))
plot(z, interactive=True)

plot(u)
plot(l)
interactive()

## ERROR ##
u_ex = Function(V)
F_ex = replace(F[0], {u: u_ex})
J_ex = derivative(F_ex, u_ex, du)
def boundary(x, on_boundary):
	return on_boundary
bc_ex = DirichletBC(V, g, boundary)
problem_ex = NonlinearVariationalProblem(F_ex, u_ex, bc_ex, J_ex)
solver_ex = NonlinearVariationalSolver(problem_ex)
solver_ex.parameters.update(snes_solver_parameters)
solver_ex.solve()
plot(u_ex)
err = Function(V)
err.vector().add_local(+ u_ex.vector().array())
err.vector().add_local(- u.vector().array())
err.vector().apply("")
plot(err)
interactive()

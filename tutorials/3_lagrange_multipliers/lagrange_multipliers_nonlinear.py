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

from numpy import isclose
from ufl import replace
from dolfin import *
# import matplotlib.pyplot as plt
from multiphenics import *

r"""
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

# MESHES #
# Mesh
mesh = Mesh("data/circle.xml")
subdomains = MeshFunction("size_t", mesh, "data/circle_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/circle_facet_region.xml")
# Dirichlet boundary
boundary_restriction = MeshRestriction(mesh, "data/circle_restriction_boundary.rtc.xml")

# FUNCTION SPACES #
# Function space
V = FunctionSpace(mesh, "Lagrange", 2)
# Block function space
W = BlockFunctionSpace([V, V], restrict=[None, boundary_restriction])

# TRIAL/TEST FUNCTIONS #
dul = BlockTrialFunction(W)
(du, dl) = block_split(dul)
ul = BlockFunction(W)
(u, l) = block_split(ul)
vm = BlockTestFunction(W)
(v, m) = block_split(vm)

# MEASURES #
dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)

# ASSEMBLE #
x = SpatialCoordinate(mesh)
g = sin(3*x[0] + 1)*sin(3*x[1] + 1)
F = [inner((1+u**2)*grad(u), grad(v))*dx + u*v*inner(grad(u), grad(u))*dx + l*v*ds - v*dx,
     u*m*ds - g*m*ds]
J = block_derivative(F, ul, dul)

# SOLVE #
problem = BlockNonlinearProblem(F, ul, None, J)
solver = BlockPETScSNESSolver(problem)
solver.parameters.update(snes_solver_parameters["snes_solver"])
solver.solve()

# (u, l) = ul.block_split()
# plt.figure()
# plot(u)
# plt.figure()
# plot(l)
# plt.show()

# ERROR #
u_ex = Function(V)
F_ex = replace(F[0], {u: u_ex})
J_ex = derivative(F_ex, u_ex, du)
bc_ex = DirichletBC(V, g, boundaries, 1)
problem_ex = NonlinearVariationalProblem(F_ex, u_ex, bc_ex, J_ex)
solver_ex = NonlinearVariationalSolver(problem_ex)
solver_ex.parameters.update(snes_solver_parameters)
solver_ex.solve()
# plt.figure()
# plot(u_ex)
# plt.show()
err = Function(V)
err.vector().add_local(+ u_ex.vector().get_local())
err.vector().add_local(- u.vector().get_local())
err.vector().apply("")
# plt.figure()
# plot(err)
# plt.show()
u_ex_norm = sqrt(assemble(inner(grad(u_ex), grad(u_ex))*dx))
err_norm = sqrt(assemble(inner(grad(err), grad(err))*dx))
print("Relative error is equal to", err_norm/u_ex_norm)
assert isclose(err_norm/u_ex_norm, 0., atol=1.e-9)

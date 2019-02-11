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

import numpy
from numpy import isclose, where
from ufl import replace
from petsc4py import PETSc
from dolfin import *
from dolfin.cpp.mesh import GhostMode
from dolfin.fem import assemble_matrix, assemble_scalar, assemble_vector
from multiphenics import *
from multiphenics.fem import DirichletBCLegacy
from multiphenics.io import XDMFFile

r"""
In this example we solve a nonlinear Laplace problem associated to
    min E(u)
    s.t. u = g on \partial \Omega
where
    E(u) = \int_\Omega { (1 + u^2) |grad u|^2 - u } dx
using a Lagrange multiplier to handle non-homogeneous Dirichlet boundary conditions.
"""

# MESHES #
# Mesh
mesh = XDMFFile(MPI.comm_world, "data/circle.xdmf").read_mesh(MPI.comm_world, GhostMode.none)
subdomains = XDMFFile(MPI.comm_world, "data/circle_physical_region.xdmf").read_mf_size_t(mesh)
boundaries = XDMFFile(MPI.comm_world, "data/circle_facet_region.xdmf").read_mf_size_t(mesh)
# Dirichlet boundary
boundary_restriction = XDMFFile(MPI.comm_world, "data/circle_restriction_boundary.rtc.xdmf").read_mesh_restriction(mesh)

# FUNCTION SPACES #
# Function space
V = FunctionSpace(mesh, ("Lagrange", 2))
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
@function.expression.numba_eval
def g_eval(values, x, cell):
    values[:, 0] = numpy.sin(3*x[:, 0] + 1)*numpy.sin(3*x[:, 1] + 1)
g = interpolate(Expression(g_eval), V)
F = [inner((1+u**2)*grad(u), grad(v))*dx + u*v*inner(grad(u), grad(u))*dx + l*v*ds - v*dx,
     u*m*ds - g*m*ds]
J = block_derivative(F, ul, dul)

# SOLVE #
def set_solver_parameters(solver):
    solver.max_it = 20
    
problem = BlockNonlinearProblem(F, ul, None, J)
solver = BlockNewtonSolver(mesh.mpi_comm())
set_solver_parameters(solver)
solver.solve(problem, ul.block_vector())

# ERROR #
# Class for interfacing with the Newton solver
class LagrangeMultipliersNonlinearProblem(NonlinearProblem):
    def __init__(self, F, up, bc, J):
        NonlinearProblem.__init__(self)
        self._F = F
        self._up = up
        self._bc = bc
        self._J = J
        self._F_vec = None
        self._J_mat = None
        
    def form(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, _):
        if self._F_vec is None:
            self._F_vec = assemble_vector(self._F)
        else:
            with self._F_vec.localForm() as f_local:
                f_local.set(0.0)
            assemble_vector(self._F_vec, self._F)
        self._F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        DirichletBCLegacy.apply(self._bc, self._F_vec, self._up.vector())
        return self._F_vec

    def J(self, _):
        if self._J_mat is None:
            self._J_mat = assemble_matrix(self._J)
        else:
            self._J_mat.zeroEntries()
            assemble_matrix(self._J_mat, self._J)
        self._J_mat.assemble()
        DirichletBCLegacy.apply(self._bc, self._J_mat, 1.0)
        return self._J_mat
        
u_ex = Function(V)
F_ex = replace(F[0], {u: u_ex, l: 0})
J_ex = derivative(F_ex, u_ex, du)
boundaries_1 = where(boundaries.array() == 1)[0]
bc_ex = [DirichletBC(V, g, boundaries_1)]
problem_ex = LagrangeMultipliersNonlinearProblem(F_ex, u_ex, bc_ex, J_ex)
solver_ex = NewtonSolver(mesh.mpi_comm())
set_solver_parameters(solver_ex)
solver_ex.solve(problem_ex, u_ex.vector())
u_ex_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(u_ex), grad(u_ex))*dx)))
err_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(u_ex - ul[0]), grad(u_ex - ul[0]))*dx)))
print("Relative error is equal to", err_norm/u_ex_norm)
assert isclose(err_norm/u_ex_norm, 0., atol=1.e-9)

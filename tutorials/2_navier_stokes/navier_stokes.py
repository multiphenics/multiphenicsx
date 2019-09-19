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

from numpy import isclose, where
from petsc4py import PETSc
from dolfin import *
from dolfin.cpp.mesh import GhostMode
from dolfin.fem import assemble_matrix, assemble_scalar, assemble_vector
from dolfin.io import XDMFFile
from multiphenics import *
from multiphenics.fem import DirichletBCLegacy

"""
In this tutorial we compare the formulation and solution
of a Navier-Stokes by standard FEniCS code (using the
MixedElement class) and multiphenics code.
"""

# Constitutive parameters
nu = 0.01
@function.expression.numba_eval
def u_in_eval(values, x, cell):
    values[:, 0] = 1.0
    values[:, 1] = 0.0
@function.expression.numba_eval
def u_wall_eval(values, x, cell):
    values[:, 0] = 0.0
    values[:, 1] = 0.0

# Solver parameters
def set_solver_parameters(solver):
    solver.max_it = 20
                                          
# Mesh
mesh = XDMFFile(MPI.comm_world, "data/backward_facing_step.xdmf").read_mesh(MPI.comm_world, GhostMode.none)
subdomains = XDMFFile(MPI.comm_world, "data/backward_facing_step_physical_region.xdmf").read_mf_size_t(mesh)
boundaries = XDMFFile(MPI.comm_world, "data/backward_facing_step_facet_region.xdmf").read_mf_size_t(mesh)
boundaries_1 = where(boundaries.array() == 1)[0]
boundaries_2 = where(boundaries.array() == 2)[0]

# Function spaces
V_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)

# -------------------------------------------------- #

# STANDARD FEniCS FORMULATION BY FEniCS MixedElement #
def run_monolithic():
    # Function spaces
    W_element = MixedElement(V_element, Q_element)
    W = FunctionSpace(mesh, W_element)

    # Test and trial functions: monolithic
    vq = TestFunction(W)
    (v, q) = split(vq)
    dup = TrialFunction(W)
    up = Function(W)
    (u, p) = split(up)

    # Variational forms
    F = (
            nu*inner(grad(u), grad(v))*dx
          + inner(grad(u)*u, v)*dx
          - div(v)*p*dx
          + div(u)*q*dx
        )
    J = derivative(F, up, dup)

    # Boundary conditions
    u_in = interpolate(Expression(u_in_eval, shape=(2,)), W.sub(0).collapse())
    u_wall = interpolate(Expression(u_wall_eval, shape=(2,)), W.sub(0).collapse())
    inlet_bc = DirichletBC(W.sub(0), u_in, boundaries_1)
    wall_bc = DirichletBC(W.sub(0), u_wall, boundaries_2)
    bc = [inlet_bc, wall_bc]
    
    # Class for interfacing with the Newton solver
    class NavierStokesProblem(NonlinearProblem):
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
            DirichletBCLegacy.apply(self._bc, self._F_vec, self._up.vector)
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

    # Solve
    problem = NavierStokesProblem(F, up, bc, J)
    solver = NewtonSolver(mesh.mpi_comm())
    set_solver_parameters(solver)
    solver.solve(problem, up.vector)

    # Extract solutions
    return up
    
up_m = run_monolithic()

# -------------------------------------------------- #

#                 multiphenics FORMULATION           #
def run_block():
    # Function spaces
    W_element = BlockElement(V_element, Q_element)
    W = BlockFunctionSpace(mesh, W_element)

    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    dup = BlockTrialFunction(W)
    up = BlockFunction(W)
    (u, p) = block_split(up)

    # Variational forms
    F = [nu*inner(grad(u), grad(v))*dx + inner(grad(u)*u, v)*dx - div(v)*p*dx,
         div(u)*q*dx]
    J = block_derivative(F, up, dup)

    # Boundary conditions
    u_in = interpolate(Expression(u_in_eval, shape=(2,)), W.sub(0))
    u_wall = interpolate(Expression(u_wall_eval, shape=(2,)), W.sub(0))
    inlet_bc = DirichletBC(W.sub(0), u_in, boundaries_1)
    wall_bc = DirichletBC(W.sub(0), u_wall, boundaries_2)
    bc = BlockDirichletBC([[inlet_bc, wall_bc], []])

    # Solve
    problem = BlockNonlinearProblem(F, up, bc, J)
    solver = BlockNewtonSolver(mesh.mpi_comm())
    set_solver_parameters(solver)
    solver.solve(problem, up.block_vector)

    # Extract solutions
    return up
    
up_b = run_block()

# -------------------------------------------------- #

#                  ERROR COMPUTATION                 #
def run_error(up_m, up_b):
    (u_m, p_m) = up_m.split()
    (u_b, p_b) = up_b.block_split()
    u_m_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(u_m), grad(u_m))*dx)))
    err_u_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(u_b - u_m), grad(u_b - u_m))*dx)))
    p_m_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(p_m, p_m)*dx)))
    err_p_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(p_b - p_m, p_b - p_m)*dx)))
    print("Relative error for velocity component is equal to", err_u_norm/u_m_norm)
    print("Relative error for pressure component is equal to", err_p_norm/p_m_norm)
    assert isclose(err_u_norm/u_m_norm, 0., atol=1.e-10)
    assert isclose(err_p_norm/p_m_norm, 0., atol=1.e-10)

run_error(up_m, up_b)

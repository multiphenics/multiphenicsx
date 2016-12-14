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

"""
In this tutorial we compare the computation of the inf-sup constant
of a Stokes by standard FEniCS code (using the
MixedElement class) and block_ext code.
"""

## -------------------------------------------------- ##

##                  MESH GENERATION                   ##
# Create mesh
mesh = UnitSquareMesh(32, 32)

# Create boundaries
class Wall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[1] < 0 + DOLFIN_EPS or x[1] > 1 - DOLFIN_EPS)
    
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
wall = Wall()
wall.mark(boundaries, 1)

# Function spaces
V_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)

def normalize(u1, u2, p):
    u1.vector()[:] /= assemble(inner(grad(u1), grad(u1))*dx)
    u2.vector()[:] /= assemble(inner(grad(u2), grad(u2))*dx)
    p.vector()[:] /= assemble(p*p*dx)

## -------------------------------------------------- ##

## STANDARD FEniCS FORMULATION BY FEniCS MixedElement ##
def run_monolithic():
    # Function spaces
    W_element = MixedElement(V_element, Q_element)
    W = FunctionSpace(mesh, W_element)

    # Test and trial functions: monolithic
    vq  = TestFunction(W)
    (v, q) = split(vq)
    up = TrialFunction(W)
    (u, p) = split(up)

    # Variational forms
    lhs = (   inner(grad(u), grad(v))*dx
              - div(v)*p*dx
              - div(u)*q*dx
            )
    rhs =   - inner(p, q)*dx

    # Boundary conditions
    bc = DirichletBC(W.sub(0), Constant((0., 0.)), boundaries, 1)

    # Assemble lhs and rhs matrices, "removing" dofs associated to Dirichlet BCs
    def constrain(matrix, diag_value):
        constrained_dofs = [bc.function_space().dofmap().local_to_global_index(local_dof_index) for local_dof_index in bc.get_boundary_values().keys()]
        as_backend_type(matrix).mat().zeroRowsColumns(constrained_dofs, diag_value)
    LHS = assemble(lhs)
    RHS = assemble(rhs)
    diag_value = 10. # this will insert a spurious eigenvalue equal to 10, which hopefully will not be the smallest one
    constrain(LHS, diag_value)
    constrain(RHS, 1.)

    # Solve
    LHS = as_backend_type(LHS)
    RHS = as_backend_type(RHS)
    eigensolver = SLEPcEigenSolver(LHS, RHS)
    eigensolver.parameters["problem_type"] = "gen_non_hermitian"
    eigensolver.parameters["spectrum"] = "smallest real"
    eigensolver.parameters["spectral_transform"] = "shift-and-invert"
    eigensolver.parameters["spectral_shift"] = 1.e-5
    eigensolver.solve(1)
    r, c = eigensolver.get_eigenvalue(0)
    assert abs(c) < 1.e-10
    assert r > 0., "r = " + str(r) + " is not positive"
    print "Inf-sup constant (monolithic): ", sqrt(r)

    # Extract eigenfunctions
    (_, _, r_vec, _) = eigensolver.get_eigenpair(0)
    r_fun = Function(W, r_vec)
    (u_fun, p_fun) = r_fun.split(deepcopy=True)
    (u_fun_1, u_fun_2) = u_fun.split(deepcopy=True)
    normalize(u_fun_1, u_fun_2, p_fun)
    plot(u_fun_1, title="Velocity 1 monolithic", mode="color")
    plot(u_fun_2, title="Velocity 2 monolithic", mode="color")
    plot(p_fun, title="Pressure monolithic", mode="color")
    interactive()
    
    return (u_fun_1, u_fun_2, p_fun)
    
(u_fun_1_m, u_fun_2_m, p_fun_m) = run_monolithic()

## -------------------------------------------------- ##
##                  block_ext FORMULATION             ##
def run_block():
    # Function spaces
    W_element = BlockElement(V_element, Q_element)
    W = BlockFunctionSpace(mesh, W_element)

    # Test and trial functions
    vq  = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    u, p = block_split(up)

    # Variational forms
    lhs = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
           [- div(u)*q*dx             , 0            ]]
    rhs = [[0                         , 0            ],
           [0                         , - p*q*dx     ]]

    # Boundary conditions
    wallc = DirichletBC(W.sub(0), Constant((0., 0.)), boundaries, 1)
    bc = BlockDirichletBC([[wallc], []])

    # Assemble lhs and rhs matrices
    LHS = block_assemble(lhs)
    RHS = block_assemble(rhs)

    # Solve
    eigensolver = BlockSLEPcEigenSolver(LHS, RHS, bc)
    eigensolver.parameters["problem_type"] = "gen_non_hermitian"
    eigensolver.parameters["spectrum"] = "smallest real"
    eigensolver.parameters["spectral_transform"] = "shift-and-invert"
    eigensolver.parameters["spectral_shift"] = 1.e-5
    eigensolver.solve(1)
    r, c = eigensolver.get_eigenvalue(0)
    assert abs(c) < 1.e-10
    assert r > 0., "r = " + str(r) + " is not positive"
    print "Inf-sup constant (block): ", sqrt(r)

    # Extract eigenfunctions
    (_, _, r_vec, _) = eigensolver.get_eigenpair(0)
    r_fun = BlockFunction(W, r_vec)
    (u_fun, p_fun) = r_fun.block_split()
    (u_fun_1, u_fun_2) = u_fun.split(deepcopy=True)
    normalize(u_fun_1, u_fun_2, p_fun)
    plot(u_fun_1, title="Velocity 1 block", mode="color")
    plot(u_fun_2, title="Velocity 2 block", mode="color")
    plot(p_fun, title="Pressure block", mode="color")
    interactive()
    
    return (u_fun_1, u_fun_2, p_fun)
    
(u_fun_1_b, u_fun_2_b, p_fun_b) = run_block()

## -------------------------------------------------- ##

##                  ERROR COMPUTATION                 ##
def run_error(u_fun_1_m, u_fun_1_b, u_fun_2_m, u_fun_2_b, p_fun_m, p_fun_b):
    # Even after normalization, eigenfunctions may have different signs. Try both and assume that the correct
    # error computation is the one for which the error is minimum
    err_1_plus = u_fun_1_b + u_fun_1_m
    err_2_plus = u_fun_2_b + u_fun_2_m
    err_p_plus = p_fun_b + p_fun_m
    err_1_minus = u_fun_1_b - u_fun_1_m
    err_2_minus = u_fun_2_b - u_fun_2_m
    err_p_minus = p_fun_b - p_fun_m
    err_1_plus_norm = assemble(inner(grad(err_1_plus), grad(err_1_plus))*dx)
    err_2_plus_norm = assemble(inner(grad(err_2_plus), grad(err_2_plus))*dx)
    err_p_plus_norm = assemble(err_p_plus*err_p_plus*dx)
    err_1_minus_norm = assemble(inner(grad(err_1_minus), grad(err_1_minus))*dx)
    err_2_minus_norm = assemble(inner(grad(err_2_minus), grad(err_2_minus))*dx)
    err_p_minus_norm = assemble(err_p_minus*err_p_minus*dx)
    u_fun_1_norm = assemble(inner(grad(u_fun_1_m), grad(u_fun_1_m))*dx)
    u_fun_2_norm = assemble(inner(grad(u_fun_2_m), grad(u_fun_2_m))*dx)
    p_fun_norm = assemble(p_fun_m*p_fun_m*dx)
    def select_error(err_plus, err_plus_norm, err_minus, err_minus_norm, vec_norm, component_name):
        ratio_plus = err_plus_norm/vec_norm
        ratio_minus = err_minus_norm/vec_norm
        if ratio_minus < ratio_plus:
            print "Relative error for ", component_name, "component of eigenvector equal to", ratio_minus, "(the one with opposite sign was", ratio_plus, ")"
            return err_minus
        else:
            print "Relative error for", component_name, "component of eigenvector equal to", ratio_plus, "(the one with opposite sign was", ratio_minus, ")"
            return err_plus
    err_1 = select_error(err_1_plus, err_1_plus_norm, err_1_minus, err_1_minus_norm, u_fun_1_norm, "velocity 1")
    err_2 = select_error(err_2_plus, err_2_plus_norm, err_2_minus, err_2_minus_norm, u_fun_2_norm, "velocity 2")
    err_p = select_error(err_p_plus, err_p_plus_norm, err_p_minus, err_p_minus_norm, p_fun_norm, "pressure")
    plot(err_1, title="Velocity 1 error", mode="color")
    plot(err_2, title="Velocity 2 error", mode="color")
    plot(err_p, title="Pressure error", mode="color")
    interactive()
    
run_error(u_fun_1_m, u_fun_1_b, u_fun_2_m, u_fun_2_b, p_fun_m, p_fun_b)

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
# Function spaces
W_element_m = MixedElement(V_element, Q_element)
W_m = FunctionSpace(mesh, W_element_m)

# Test and trial functions: monolithic
vq_m  = TestFunction(W_m)
(v_m, q_m) = split(vq_m)
up_m = TrialFunction(W_m)
(u_m, p_m) = split(up_m)

# Variational forms
lhs_m = (   inner(grad(u_m), grad(v_m))*dx
          - div(v_m)*p_m*dx
          - div(u_m)*q_m*dx
        )
rhs_m =   - inner(p_m, q_m)*dx

# Boundary conditions
bc_m = DirichletBC(W_m.sub(0), Constant((0., 0.)), boundaries, 1)

# Assemble lhs and rhs matrices, "removing" dofs associated to Dirichlet BCs
def constrain_m(matrix_m, diag_value_m):
    dummy_m = Function(W_m)
    DUMMY_m = dummy_m.vector()
    bc_m.zero(matrix_m)
    bc_m.zero_columns(matrix_m, DUMMY_m, diag_value_m)
LHS_m = assemble(lhs_m)
RHS_m = assemble(rhs_m)
diag_value_m = 10. # this will insert a spurious eigenvalue equal to 10, which hopefully will not be the smallest one
constrain_m(LHS_m, diag_value_m)
constrain_m(RHS_m, 1.)

# Solve
LHS_m = as_backend_type(LHS_m)
RHS_m = as_backend_type(RHS_m)
eigensolver_m = SLEPcEigenSolver(LHS_m, RHS_m)
eigensolver_m.parameters["problem_type"] = "gen_non_hermitian"
eigensolver_m.parameters["spectrum"] = "smallest real"
eigensolver_m.parameters["spectral_transform"] = "shift-and-invert"
eigensolver_m.parameters["spectral_shift"] = 1.e-5
eigensolver_m.solve(1)
r_m, c_m = eigensolver_m.get_eigenvalue(0)
assert abs(c_m) < 1.e-10
assert r_m > 0., "r_m = " + str(r_m) + " is not positive"
print "Inf-sup constant (monolithic): ", sqrt(r_m)

# Extract eigenfunctions
(_, _, r_vec_m, _) = eigensolver_m.get_eigenpair(0)
r_fun_m = Function(W_m, r_vec_m)
(u_fun_m, p_fun_m) = r_fun_m.split(deepcopy=True)
(u_fun_1_m, u_fun_2_m) = u_fun_m.split(deepcopy=True)
normalize(u_fun_1_m, u_fun_2_m, p_fun_m)
plot(u_fun_1_m, title="Velocity 1 monolithic", mode="color")
plot(u_fun_2_m, title="Velocity 2 monolithic", mode="color")
plot(p_fun_m, title="Pressure monolithic", mode="color")

## -------------------------------------------------- ##
##                  block_ext FORMULATION             ##
# Function spaces
W_element_b = BlockElement(V_element, Q_element)
W_b = BlockFunctionSpace(mesh, W_element_b)

# Test and trial functions
vq_b  = BlockTestFunction(W_b)
(v_b, q_b) = block_split(vq_b)
up_b = BlockTrialFunction(W_b)
u_b, p_b = block_split(up_b)

# Variational forms
lhs_b = [[inner(grad(u_b), grad(v_b))*dx, - div(v_b)*p_b*dx],
         [- div(u_b)*q_b*dx, Constant(0.)*p_b*q_b*dx]]
rhs_b = [[Constant(0.)*inner(u_b, v_b)*dx, Constant(0.)*div(v_b)*p_b*dx],
         [- Constant(0.)*div(u_b)*q_b*dx, - p_b*q_b*dx]]

# Boundary conditions
wall_bc_b = DirichletBC(W_b.sub(0), Constant((0., 0.)), boundaries, 1)
bc_b = BlockDirichletBC([[wall_bc_b], []])

# Assemble lhs and rhs matrices, "removing" dofs associated to Dirichlet BCs
def constrain_b(matrix_b, diag_value_b):
    dummy_b = BlockFunction(W_b)
    DUMMY_b = dummy_b.block_vector()
    bc_b.zero(matrix_b)
    bc_b.zero_columns(matrix_b, DUMMY_b, diag_value_b)
LHS_b = block_assemble(lhs_b)
RHS_b = block_assemble(rhs_b)
diag_value_b = 10. # this will insert a spurious eigenvalue equal to 10, which hopefully will not be the smallest one
constrain_b(LHS_b, diag_value_b)
constrain_b(RHS_b, 1.)

# Solve
eigensolver_b = BlockSLEPcEigenSolver(LHS_b, RHS_b)
eigensolver_b.parameters["problem_type"] = "gen_non_hermitian"
eigensolver_b.parameters["spectrum"] = "smallest real"
eigensolver_b.parameters["spectral_transform"] = "shift-and-invert"
eigensolver_b.parameters["spectral_shift"] = 1.e-5
eigensolver_b.solve(1)
r_b, c_b = eigensolver_b.get_eigenvalue(0)
assert abs(c_b) < 1.e-10
assert r_b > 0., "r_b = " + str(r_b) + " is not positive"
print "Inf-sup constant (block): ", sqrt(r_b)

# Extract eigenfunctions
(_, _, r_vec_b, _) = eigensolver_b.get_eigenpair(0)
r_fun_b = BlockFunction(W_b, r_vec_b)
(u_fun_b, p_fun_b) = r_fun_b.block_split()
(u_fun_1_b, u_fun_2_b) = u_fun_b.split(deepcopy=True)
normalize(u_fun_1_b, u_fun_2_b, p_fun_b)
plot(u_fun_1_b, title="Velocity 1 block", mode="color")
plot(u_fun_2_b, title="Velocity 2 block", mode="color")
plot(p_fun_b, title="Pressure block", mode="color")

## -------------------------------------------------- ##

##                  ERROR COMPUTATION                 ##
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

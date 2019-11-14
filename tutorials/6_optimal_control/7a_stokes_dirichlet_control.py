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

from numpy import isclose, where, zeros
from ufl import *
from dolfin import *
from dolfin.cpp.mesh import GhostMode
from dolfin.fem import assemble_scalar
from dolfin.plotting import plot
import matplotlib.pyplot as plt
from multiphenics import *
from multiphenics.io import XDMFFile

r"""
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

# MESH #
# Mesh
mesh = XDMFFile(MPI.comm_world, "data/vorticity_reduction.xdmf").read_mesh(GhostMode.none)
subdomains = XDMFFile(MPI.comm_world, "data/vorticity_reduction_physical_region.xdmf").read_mf_size_t(mesh)
boundaries = XDMFFile(MPI.comm_world, "data/vorticity_reduction_facet_region.xdmf").read_mf_size_t(mesh)
boundaries_side = {idx: where(boundaries.values == idx)[0] for idx in (1, 2, 4, 5)}
# Dirichlet control boundary
control_boundary = XDMFFile(MPI.comm_world, "data/vorticity_reduction_restriction_control.rtc.xdmf").read_mesh_restriction(mesh)
# Normal and tangent
n = FacetNormal(mesh)
t = as_vector([n[1], -n[0]])

# FUNCTION SPACES #
Y_velocity = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Y_pressure = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
U = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
L = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
Q_velocity = Y_velocity
Q_pressure = Y_pressure
W_el = BlockElement(Y_velocity, Y_pressure, U, L, Q_velocity, Q_pressure)
W = BlockFunctionSpace(mesh, W_el, restrict=[None, None, control_boundary, control_boundary, None, None])

# PROBLEM DATA #
nu = 1.
alpha = 1.e-2
f = Constant(mesh, (0., 0.))
bc0 = Function(W.sub(0))
bc0_component = Function(W.sub(0).sub(0).collapse())
def non_zero_eval(x):
    values = zeros((2, x.shape[1]))
    values[0, :] = 2.5
    return values
bc1 = Function(W.sub(0))
bc1.interpolate(non_zero_eval)

# TRIAL/TEST FUNCTIONS #
trial = BlockTrialFunction(W)
(v, p, u, l, z, b) = block_split(trial)
test = BlockTestFunction(W)
(w, q, r, m, s, d) = block_split(test)

# MEASURES #
dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)

# OPTIMALITY CONDITIONS #
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
                            DirichletBC(W.sub(0), bc1, boundaries_side[1]),
                            DirichletBC(W.sub(0).sub(1), bc0_component, boundaries_side[2]),
                            DirichletBC(W.sub(0).sub(0), bc0_component, boundaries_side[4]),
                            DirichletBC(W.sub(0), bc0, boundaries_side[5]),
                        ],
                       [],
                       [],
                       [],
                       [
                            DirichletBC(W.sub(4), bc0, boundaries_side[1]),
                            DirichletBC(W.sub(4).sub(1), bc0_component, boundaries_side[2]),
                            DirichletBC(W.sub(4), bc0, boundaries_side[4]),
                            DirichletBC(W.sub(4), bc0, boundaries_side[5]),
                        ],
                       []])

# SOLUTION #
solution = BlockFunction(W)
(v, p, u, l, z, b) = block_split(solution)

# FUNCTIONAL #
J = 0.5*vorticity(v, v)*dx(4) + 0.5*penalty(u, u)*ds(4)

# UNCONTROLLED FUNCTIONAL VALUE #
W_state_trial = W.extract_block_sub_space((0, 1))
W_state_test = W.extract_block_sub_space((4, 5))
a_state = block_restrict(a, [W_state_test, W_state_trial])
f_state = block_restrict(f, W_state_test)
bc_state = BlockDirichletBC([[
                                  DirichletBC(W_state_trial.sub(0), bc1, boundaries_side[1]),
                                  DirichletBC(W_state_trial.sub(0).sub(1), bc0_component, boundaries_side[2]),
                                  DirichletBC(W_state_trial.sub(0), bc0, boundaries_side[4]),
                                  DirichletBC(W_state_trial.sub(0), bc0, boundaries_side[5]),
                              ],
                             []])
solution_state = block_restrict(solution, W_state_trial)
solver_parameters = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
block_solve(a_state, solution_state, f_state, bc_state, petsc_options=solver_parameters)
J_uncontrolled = MPI.sum(mesh.mpi_comm(), assemble_scalar(J))
print("Uncontrolled J =", J_uncontrolled)
assert isclose(J_uncontrolled, 2.9377904)
plt.figure()
plot(v, title="uncontrolled state velocity")
plt.figure()
plot(p, title="uncontrolled state pressure")
plt.show()

# OPTIMAL CONTROL #
block_solve(a, solution, f, bc, petsc_options=solver_parameters)
J_controlled = MPI.sum(mesh.mpi_comm(), assemble_scalar(J))
print("Optimal J =", J_controlled)
assert isclose(J_controlled, 1.71502760)
plt.figure()
plot(v, title="state velocity")
plt.figure()
plot(p, title="state pressure")
plt.figure()
plot(u, title="control")
plt.figure()
plot(l, title="lambda")
plt.figure()
plot(z, title="adjoint velocity")
plt.figure()
plot(b, title="adjoint pressure")
plt.show()

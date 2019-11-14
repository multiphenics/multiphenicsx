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

from numpy import isclose, isin, where, zeros
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

# MESH #
# Mesh
if MPI.size(MPI.comm_world) > 1:
    mesh_ghost_mode = GhostMode.shared_facet # shared_facet ghost mode is required by dS
else:
    mesh_ghost_mode = GhostMode.none
mesh = XDMFFile(MPI.comm_world, "data/bifurcation.xdmf").read_mesh(mesh_ghost_mode)
subdomains = XDMFFile(MPI.comm_world, "data/bifurcation_physical_region.xdmf").read_mf_size_t(mesh)
boundaries = XDMFFile(MPI.comm_world, "data/bifurcation_facet_region.xdmf").read_mf_size_t(mesh)
boundaries_1 = where(boundaries.values == 1)[0]
boundaries_2 = where(boundaries.values == 2)[0]
boundaries_12 = where(isin(boundaries.values, (1, 2)))[0]
# Neumann control boundary
control_boundary = XDMFFile(MPI.comm_world, "data/bifurcation_restriction_control.rtc.xdmf").read_mesh_restriction(mesh)
# Normal and tangent
n = FacetNormal(mesh)
t = as_vector([n[1], -n[0]])

# FUNCTION SPACES #
Y_velocity = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Y_pressure = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
U = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q_velocity = Y_velocity
Q_pressure = Y_pressure
W_el = BlockElement(Y_velocity, Y_pressure, U, Q_velocity, Q_pressure)
W = BlockFunctionSpace(mesh, W_el, restrict=[None, None, control_boundary, None, None])

# PROBLEM DATA #
nu = 0.04
alpha_1 = 0.001
alpha_2 = 0.1*alpha_1
x = SpatialCoordinate(mesh)
a = 1.0
b = 0.8
v_d = as_vector((a*(b*10.0*(x[1]**3 - x[1]**2 - x[1] + 1.0)) + ((1.0-b)*10.0*(-x[1]**3 - x[1]**2 + x[1] + 1.0)), 0.0))
f = Constant(mesh, (0., 0.))
def g_eval(x):
    values = zeros((2, x.shape[1]))
    values[0, :] = 10.0*a*(x[1, :] + 1.0)*(1.0 - x[1, :])
    return values
g = Function(W.sub(0))
g.interpolate(g_eval)
bc0 = Function(W.sub(0))

# TRIAL/TEST FUNCTIONS #
trial = BlockTrialFunction(W)
(v, p, u, z, b) = block_split(trial)
test = BlockTestFunction(W)
(w, q, r, s, d) = block_split(test)

# MEASURES #
dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)
dS = Measure("dS")(subdomain_data=boundaries)

# OPTIMALITY CONDITIONS #
def tracking(v, w):
    return inner(v, w)("-")
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
bc = BlockDirichletBC([[DirichletBC(W.sub(0), g, boundaries_1), DirichletBC(W.sub(0), bc0, boundaries_2)],
                       [],
                       [],
                       [DirichletBC(W.sub(3), bc0, boundaries_12)],
                       []])

# SOLUTION #
solution = BlockFunction(W)
(v, p, u, z, b) = block_split(solution)

# FUNCTIONAL #
J = 0.5*tracking(v - v_d, v - v_d)*dS(4) + 0.5*penalty(u, u)*ds(3)

# UNCONTROLLED FUNCTIONAL VALUE #
W_state_trial = W.extract_block_sub_space((0, 1))
W_state_test = W.extract_block_sub_space((3, 4))
a_state = block_restrict(a, [W_state_test, W_state_trial])
f_state = block_restrict(f, W_state_test)
bc_state = block_restrict(bc, W_state_trial)
solution_state = block_restrict(solution, W_state_trial)
solver_parameters = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
block_solve(a_state, solution_state, f_state, bc_state, petsc_options=solver_parameters)
J_uncontrolled = MPI.sum(mesh.mpi_comm(), assemble_scalar(J))
print("Uncontrolled J =", J_uncontrolled)
assert isclose(J_uncontrolled, 2.8512005)
plt.figure()
plot(v, title="uncontrolled state velocity")
plt.figure()
plot(p, title="uncontrolled state pressure")
plt.show()

# OPTIMAL CONTROL #
block_solve(a, solution, f, bc, petsc_options=solver_parameters)
J_controlled = MPI.sum(mesh.mpi_comm(), assemble_scalar(J))
print("Optimal J =", J_controlled)
assert isclose(J_controlled, 1.7640778)
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

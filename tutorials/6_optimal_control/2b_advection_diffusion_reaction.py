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

from numpy import isclose, ones, where
from petsc4py import PETSc
from ufl import *
from dolfinx import *
from dolfinx.cpp.mesh import GhostMode
from dolfinx.fem import assemble_scalar, locate_dofs_topological
from dolfinx.io import XDMFFile
from dolfinx.plotting import plot
import matplotlib.pyplot as plt
from multiphenics import *

r"""
In this tutorial we solve the optimal control problem

min J(y, u) = 1/2 \int_{\Omega} (y - y_d)^2 dx + \alpha/2 \int_{\Omega} u^2 dx
s.t.
    - \epsilon \Delta y + \beta \cdot \nabla y + \sigma y = f + u       in \Omega
                                                        y = g_D         on \partial \Omega
             
where
    \Omega                      unit square
    u \in L^2(\Omega)           control variable
    y \in H^1_0(\Omega)         state variable
    \alpha > 0                  penalization parameter
    y_d = piecewise constant    desired state
    \epsilon > 0                diffusion coefficient
    \beta in IR^2               advection field
    \sigma > 0                  reaction coefficient
    f                           forcing term
    g_D = piecewise constant    non homogeneous Dirichlet BC
    
using an adjoint formulation solved by a one shot approach
"""

# MESH #
mesh = XDMFFile(MPI.comm_world, "data/graetz_1.xdmf").read_mesh(GhostMode.none)
subdomains = XDMFFile(MPI.comm_world, "data/graetz_1_physical_region.xdmf").read_mf_size_t(mesh)
boundaries = XDMFFile(MPI.comm_world, "data/graetz_1_facet_region.xdmf").read_mf_size_t(mesh)

# FUNCTION SPACES #
Y = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
U = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Q = Y
W_el = BlockElement(Y, U, Q)
W = BlockFunctionSpace(mesh, W_el)

# PROBLEM DATA #
alpha = 0.01
y_d_1 = 0.6
y_d_2 = 1.8
epsilon = 1./15.
x = SpatialCoordinate(mesh)
beta = as_vector((x[1]*(1-x[1]), 0))
sigma = Constant(mesh, 0.)
f = Constant(mesh, 0.)
def bc_generator(val):
    bc = Function(W.sub(0))
    bc.interpolate(lambda x: val*ones(x.shape[1]))
    return bc

# TRIAL/TEST FUNCTIONS #
yup = BlockTrialFunction(W)
(y, u, p) = block_split(yup)
zvq = BlockTestFunction(W)
(z, v, q) = block_split(zvq)

# MEASURES #
dx = Measure("dx")(subdomain_data=subdomains)

# OPTIMALITY CONDITIONS #
state_operator = epsilon*inner(grad(y), grad(q))*dx + inner(beta, grad(y))*q*dx + sigma*y*q*dx
adjoint_operator = epsilon*inner(grad(p), grad(z))*dx - inner(beta, grad(p))*z*dx + sigma*p*z*dx
a = [[y*z*(dx(1) + dx(2)), 0           , adjoint_operator],
     [0                  , alpha*u*v*dx, - p*v*dx        ],
     [state_operator     , - u*q*dx    , 0               ]]
f =  [y_d_1*z*dx(1) + y_d_2*z*dx(2),
      0                            ,
      f*q*dx                        ]
def bdofs_W0(idx):
    return locate_dofs_topological((W.sub(0), W.sub(0)), mesh.topology.dim - 1, where(boundaries.values == idx)[0])
def bdofs_W2(idx):
    return locate_dofs_topological((W.sub(2), W.sub(0)), mesh.topology.dim - 1, where(boundaries.values == idx)[0])
bc = BlockDirichletBC([[DirichletBC(bc_generator(idx), bdofs_W0(idx), W.sub(0)) for idx in (1, 2)],
                       [],
                       [DirichletBC(bc_generator(0.), bdofs_W2(idx), W.sub(2)) for idx in (1, 2)]])

# SOLUTION #
yup = BlockFunction(W)
(y, u, p) = block_split(yup)

# FUNCTIONAL #
J = 0.5*inner(y - y_d_1, y - y_d_1)*dx(1) + 0.5*inner(y - y_d_2, y - y_d_2)*dx(2) + 0.5*alpha*inner(u, u)*dx

# UNCONTROLLED FUNCTIONAL VALUE #
a_state = replace(a[2][0], {q: z})
f_state = replace(f[2], {q: z})
bc_state = bc[0]
solver_parameters = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
solve(a_state == f_state, y, bc_state, petsc_options=solver_parameters)
y.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
J_uncontrolled = MPI.sum(mesh.mpi_comm(), assemble_scalar(J))
print("Uncontrolled J =", J_uncontrolled)
assert isclose(J_uncontrolled, 0.028050019)
plt.figure()
plot(y, title="uncontrolled state")
plt.show()

# OPTIMAL CONTROL #
block_solve(a, yup, f, bc, petsc_options=solver_parameters)
J_controlled = MPI.sum(mesh.mpi_comm(), assemble_scalar(J))
print("Optimal J =", J_controlled)
assert isclose(J_controlled, 0.001777817)
plt.figure()
plot(y, title="state")
plt.figure()
plot(u, title="control")
plt.figure()
plot(p, title="adjoint")
plt.show()

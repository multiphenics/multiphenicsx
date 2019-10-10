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

from numpy import isclose, ones, where, zeros
from petsc4py import PETSc
from ufl import *
from dolfin import *
from dolfin.cpp.mesh import GhostMode
from dolfin.fem import assemble_scalar
from dolfin.io import XDMFFile
from dolfin.plotting import plot
import matplotlib.pyplot as plt
from multiphenics import *

r"""
In this tutorial we solve the optimal control problem

min J(y, u) = 1/2 \int_{\Omega} (y - y_d)^2 dx + \alpha/2 \int_{\Omega} u^2 dx
s.t.
    - \Delta y = f + u   in \Omega
             y = 1       on \partial \Omega
             
where
    \Omega                      unit square
    u \in L^2(\Omega)           control variable
    y \in H^1_0(\Omega)         state variable
    \alpha > 0                  penalization parameter
    y_d = piecewise constant    desired state
    f                           forcing term
    
using an adjoint formulation solved by a one shot approach.
The test case is from section 5.1 of
F. Negri, G. Rozza, A. Manzoni and A. Quarteroni. Reduced Basis Method for Parametrized Elliptic Optimal Control Problems. SIAM Journal on Scientific Computing, 35(5): A2316-A2340, 2013.
"""

# MESH #
mesh = XDMFFile(MPI.comm_world, "data/rectangle.xdmf").read_mesh(GhostMode.none)
subdomains = XDMFFile(MPI.comm_world, "data/rectangle_physical_region.xdmf").read_mf_size_t(mesh)
boundaries = XDMFFile(MPI.comm_world, "data/rectangle_facet_region.xdmf").read_mf_size_t(mesh)
boundaries_1 = where(boundaries.values == 1)[0]

# FUNCTION SPACES #
Y = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
U = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Q = Y
W_el = BlockElement(Y, U, Q)
W = BlockFunctionSpace(mesh, W_el)

# PROBLEM DATA #
alpha = 0.01
y_d_1 = 1.0
y_d_2 = 0.6
f = Constant(mesh, 0.)
bc0 = Function(W.sub(0))
bc0.interpolate(lambda x: zeros(x.shape[0]))
bc1 = Function(W.sub(0))
bc1.interpolate(lambda x: ones(x.shape[0]))

# TRIAL/TEST FUNCTIONS #
yup = BlockTrialFunction(W)
(y, u, p) = block_split(yup)
zvq = BlockTestFunction(W)
(z, v, q) = block_split(zvq)

# MEASURES #
dx = Measure("dx")(subdomain_data=subdomains)

# OPTIMALITY CONDITIONS #
a = [[y*z*dx                    , 0           , inner(grad(p), grad(z))*dx],
     [0                         , alpha*u*v*dx, - p*v*dx                  ],
     [inner(grad(y), grad(q))*dx, - u*q*dx    , 0                         ]]
f =  [y_d_1*z*dx(1) + y_d_2*z*dx(2),
      0                            ,
      f*q*dx                        ]
bc = BlockDirichletBC([[DirichletBC(W.sub(0), bc1, boundaries_1)],
                       [],
                       [DirichletBC(W.sub(2), bc0, boundaries_1)]])

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
assert isclose(J_uncontrolled, 0.24)
plt.figure()
plot(y, title="uncontrolled state")
plt.show()

# OPTIMAL CONTROL #
block_solve(a, yup, f, bc, petsc_options=solver_parameters)
J_controlled = MPI.sum(mesh.mpi_comm(), assemble_scalar(J))
print("Optimal J =", J_controlled)
assert isclose(J_controlled, 0.158450589)
plt.figure()
plot(y, title="state")
plt.figure()
plot(u, title="control")
plt.figure()
plot(p, title="adjoint")
plt.show()

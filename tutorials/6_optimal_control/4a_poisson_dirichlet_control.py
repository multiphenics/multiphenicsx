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

from numpy import isclose, isin, where
from petsc4py import PETSc
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

min J(y, u) = 1/2 \int_{\Omega} (y - y_d)^2 dx + \alpha/2 \int_{\Gamma_2} u^2 ds
s.t.
      - \Delta y = f       in \Omega
    \partial_n y = 0       on \Gamma_1
               y = u       on \Gamma_2
    \partial_n y = 0       on \Gamma_3
               y = 0       on \Gamma_4
             
where
    \Omega                      unit square
    u \in L^2(\Gamma_2)         control variable
    y \in H^1_0(\Omega)         state variable
    \alpha > 0                  penalization parameter
    y_d = 1                     desired state
    f                           forcing term
    
using an adjoint formulation solved by a one shot approach
"""

# MESH #
# Mesh
mesh = XDMFFile(MPI.comm_world, "data/square.xdmf").read_mesh(GhostMode.none)
boundaries = XDMFFile(MPI.comm_world, "data/square_facet_region.xdmf").read_mf_size_t(mesh)
boundaries_4 = where(boundaries.values == 4)[0]
boundaries_24 = where(isin(boundaries.values, (2, 4)))[0]
# Neumann boundary
left = XDMFFile(MPI.comm_world, "data/square_restriction_boundary_2.rtc.xdmf").read_mesh_restriction(mesh)

# FUNCTION SPACES #
Y = FunctionSpace(mesh, ("Lagrange", 2))
U = FunctionSpace(mesh, ("Lagrange", 2))
L = FunctionSpace(mesh, ("Lagrange", 2))
Q = Y
W = BlockFunctionSpace([Y, U, L, Q], restrict=[None, left, left, None])

# PROBLEM DATA #
alpha = 1.e-5
y_d = 1.
x = SpatialCoordinate(mesh)
f = 10*sin(2*pi*x[0])*sin(2*pi*x[1])
def zero_eval(values, x):
    values[:] = 0.0
bc0 = interpolate(zero_eval, W.sub(0))

# TRIAL/TEST FUNCTIONS #
yulp = BlockTrialFunction(W)
(y, u, l, p) = block_split(yulp)
zvmq = BlockTestFunction(W)
(z, v, m, q) = block_split(zvmq)

# MEASURES #
ds = Measure("ds")(subdomain_data=boundaries)

# OPTIMALITY CONDITIONS #
a = [[y*z*dx                    , 0              ,   l*z*ds(2), inner(grad(p), grad(z))*dx],
     [0                         , alpha*u*v*ds(2), - l*v*ds(2), 0                         ],
     [y*m*ds(2)                 , - u*m*ds(2)    , 0          , 0                         ],
     [inner(grad(y), grad(q))*dx, 0              , 0          , 0                         ]]
f =  [y_d*z*dx,
      0       ,
      0       ,
      f*q*dx   ]
bc = BlockDirichletBC([[DirichletBC(W.sub(0), bc0, boundaries_4)],
                       [],
                       [],
                       [DirichletBC(W.sub(3), bc0, boundaries_24)]])

# SOLUTION #
yulp = BlockFunction(W)
(y, u, l, p) = block_split(yulp)

# FUNCTIONAL #
J = 0.5*inner(y - y_d, y - y_d)*dx + 0.5*alpha*inner(u, u)*ds(2)

# UNCONTROLLED FUNCTIONAL VALUE #
a_state = replace(a[3][0], {q: z})
f_state = replace(f[3], {q: z})
bc_state = [DirichletBC(W.sub(0), bc0, boundaries_24)]
solver_parameters = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_7": 3}
solve(a_state == f_state, y, bc_state, petsc_options=solver_parameters)
y.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
J_uncontrolled = MPI.sum(mesh.mpi_comm(), assemble_scalar(J))
print("Uncontrolled J =", J_uncontrolled)
assert isclose(J_uncontrolled, 0.5038977)
plt.figure()
plot(y, title="uncontrolled state")
plt.show()

# OPTIMAL CONTROL #
block_solve(a, yulp, f, bc, petsc_options=solver_parameters)
J_controlled = MPI.sum(mesh.mpi_comm(), assemble_scalar(J))
print("Optimal J =", J_controlled)
assert isclose(J_controlled, 0.1281224)
plt.figure()
plot(y, title="state")
plt.figure()
plot(u, title="control")
plt.figure()
plot(l, title="lambda")
plt.figure()
plot(p, title="adjoint")
plt.show()

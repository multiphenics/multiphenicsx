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
from petsc4py import PETSc
from ufl import *
from dolfin import *
from dolfin.cpp.mesh import GhostMode
from dolfin.fem import assemble_scalar
from multiphenics import *
from multiphenics.fem import block_assemble
from multiphenics.io import XDMFFile

r"""
In this example we solve a Laplace problem with non-homogeneous
Dirichlet boundary conditions, imposed through a penalty method, i.e.
    (A + P) u = f + q
where A and f are associated to the discretization of the Laplace problem
with homogeneous Neumann boundary conditions, while the penalty matrix P
is defined as
    P_{ij} = penalty * \delta_{ij}, if i on boundary,
    P_{ij} = 0, if i in the interior,
and the penalty vector q as
    q_i = penalty * g(coordinate(i)), if i on boundary,
    q_i = 0, if i in the interior
being penalty a large number, g(x, y) the prescribed non-homogeneous
Dirichlet boundary condition, coordinate(i) the coordinate of the i-th
degree of freedom.

The preferred way to impose non-homogeneous Dirichlet boundary conditions should still
either be through BlockDirichletBC objects (see tutorial 1) or lagrange multipliers
(see tutorial 3). This example means to show how to get the list of degrees of freedom
associated to a specific restriction, and how to perform local modifications to assembled
matrices.
"""

# MESHES #
# Mesh
mesh = XDMFFile(MPI.comm_world, "data/circle.xdmf").read_mesh(GhostMode.none)
subdomains = XDMFFile(MPI.comm_world, "data/circle_physical_region.xdmf").read_mf_size_t(mesh)
boundaries = XDMFFile(MPI.comm_world, "data/circle_facet_region.xdmf").read_mf_size_t(mesh)
# Dirichlet boundary
boundary_restriction = XDMFFile(MPI.comm_world, "data/circle_restriction_boundary.rtc.xdmf").read_mesh_restriction(mesh)

# FUNCTION SPACES #
# Function space
V = FunctionSpace(mesh, ("Lagrange", 2))
# Block function space
W = BlockFunctionSpace([V], restrict=[None])

# TRIAL/TEST FUNCTIONS #
(u, ) = BlockTrialFunction(W)
(v, ) = BlockTestFunction(W)

# MEASURES #
dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)

# HELPER FUNCTIONS #
def get_local_dofs(W, component):
    """
    Computes local dofs of W[component]. Returns two lists:
    * the first list stores local dof numbering with respect to W[component], e.g. to be used to fetch data
    from FEniCS solution vectors. Note that this list *neglects* restrictions. If interested in this output
    for restricted block function spaces, please see get_local_dofs_on_restriction.
    * the second list stores local dof numbering with respect to W, e.g. to be used to fetch data from
    multiphenics solution block_vector. Note that this list *considers* restrictions.
    """
    return (
        list(range(0, W[component].dofmap.index_map.block_size*(W[component].dofmap.index_map.local_range[1] - W[component].dofmap.index_map.local_range[0]))),
        W.block_dofmap.block_owned_dofs__local_numbering(component)
    )

def get_local_dofs_on_restriction(W, component, restriction):
    """
    Computes dofs of W[component] which are on the provided restriction, which can be smaller or equal to the restriction
    provided at construction time of W (or it can be any restriction if W[component] is unrestricted). Returns two lists:
    * the first list stores local dof numbering with respect to W[component], e.g. to be used to fetch data
    from FEniCS solution vectors.
    * the second list stores local dof numbering with respect to W, e.g. to be used to fetch data from
    multiphenics solution block_vector.
    """
    # Extract unrestricted space associated to the provided component
    V = W.sub(component)
    # Prepare an auxiliary block function space, restricted on the boundary
    W_restricted = BlockFunctionSpace([V], restrict=[restriction])
    component_restricted = 0 # there is only one block in the W_restricted space
    # Get list of all local dofs on the restriction, numbered according to W_restricted. This will be a contiguous list
    # [1, 2, ..., # local dofs on the restriction]
    (_, restricted_dofs) = get_local_dofs(W_restricted, component_restricted)
    # Get the mapping of local dofs numbering from W_restricted[0] to V
    restricted_to_original = W_restricted.block_dofmap.block_to_original(component_restricted)
    # Get list of all local dofs on the restriction, but numbered according to V. Note that this list will not be
    # contiguous anymore, because there are DOFs on V other than the ones in the restriction (i.e., the ones in the
    # interior)
    original_dofs = [restricted_to_original[restricted] for restricted in restricted_dofs]
    # Get the mapping of local dofs numbering from V to W[b].
    original_to_block = W.block_dofmap.original_to_block(component)
    # Get list of all local dofs on the restriction, but numbered according to W. Note again that this list will not
    # be contiguous, and, in case of space W with multiple blocks, it will not be the same as original_dofs.
    block_dofs = [original_to_block[original] for original in original_dofs]
    return original_dofs, block_dofs

def generate_penalty_system(W, component, restriction, penalty, g):
    """
    Generate matrix and vector to be added to the system to handle the penalty terms
    """
    (fenics_local_dofs, multiphenics_local_dofs) = get_local_dofs(W, component)
    (fenics_boundary_dofs, multiphenics_boundary_dofs) = get_local_dofs_on_restriction(W, component, restriction)
    fenics_interior_dofs = [dof for dof in fenics_local_dofs if dof not in fenics_boundary_dofs]
    # Assemble penalty matrix in a multiphenics matrix. Note that, as the matrix is assembled by multiphenics,
    # we will be using multiphenics_*_dofs variables, i.e. multiphenics numbering.
    p = [[u*v*dx]] # this u*v*dx form is not really used per se, just to allocated the correct sparsity pattern
    P = block_assemble(p)
    P.zeroEntries()
    for dof in multiphenics_boundary_dofs:
        P.setValuesLocal([dof], [dof], [penalty]) # set P_{ij} = penalty * \delta_{ij}, if i on boundary
    P.assemble()
    # Note that the for loop could have been replaced by
    #   P.mat().zeroRowsColumnsLocal(multiphenics_boundary_dofs, penalty)
    # We provide a manual version to show how to query petsc4py to manually change matrix entries (not necessarily on the
    # diagonal, but necessarily included in the sparsity pattern).
    #
    # Next, assemble the penalty vector. First, copy g and rescale by penalty, throwing away interior values.
    # Note that here we are using fenics_*_dofs variables, because g is a FEniCS vector.
    q_function = BlockFunction(W)
    g.vector.copy(result=q_function.sub(component).vector)
    q_function.sub(component).vector.scale(penalty)
    q_function.sub(component).vector.setValuesLocal(fenics_interior_dofs, [0.]*len(fenics_interior_dofs))
    q_function.sub(component).vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    q_function.apply("from subfunctions", component)
    # Return matrix and vector
    return P, q_function.block_vector

# ASSEMBLE #
def g_eval(values, x):
    values[:, 0] = numpy.sin(3*x[:, 0] + 1)*numpy.sin(3*x[:, 1] + 1)
g = interpolate(g_eval, V)
a = [[inner(grad(u), grad(v))*dx]]
f =  [v*dx                      ]
A = block_assemble(a)
F = block_assemble(f)
(P, Q) = generate_penalty_system(W, 0, boundary_restriction, 1.e10, g)

# SOLVE #
# Store options
options = PETSc.Options()
solver_parameters = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
for k, v in solver_parameters.items():
    options.setValue("multiphenics_solve_" + k, v)
# Solve
U = BlockFunction(W)
solver = PETSc.KSP().create(W.mesh.mpi_comm())
solver.setOptionsPrefix("multiphenics_solve_")
solver.setFromOptions()
solver.setOperators(A + P)
solver.solve(F + Q, U.block_vector)
# Keep subfunctions up to date
U.apply("to subfunctions")

# ERROR #
boundaries_1 = where(boundaries.values == 1)[0]
bc_ex = DirichletBC(V, g, boundaries_1)
u_ex = Function(V)
solve(a[0][0] == f[0], u_ex, bc_ex, petsc_options=solver_parameters)
u_ex.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
u_ex_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(u_ex), grad(u_ex))*dx)))
err_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(u_ex - U[0]), grad(u_ex - U[0]))*dx)))
print("Relative error is equal to", err_norm/u_ex_norm)
assert isclose(err_norm/u_ex_norm, 0., atol=1.e-10)

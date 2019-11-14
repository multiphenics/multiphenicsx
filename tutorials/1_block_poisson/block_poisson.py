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

from numpy import finfo, isclose, where
from petsc4py import PETSc
from ufl import *
from dolfin import *
from dolfin.cpp.la import get_local_vectors
from dolfin.fem import assemble_matrix_block, assemble_scalar, assemble_vector_block, create_vector_block
from multiphenics import *

"""
In this tutorial we first solve the problem

-u'' = f    in Omega = [0, 1]
 u   = 0    on Gamma = {0, 1}
 
using standard FEniCS code.

Then we use multiphenics to solve the system

-   w_1'' - 2 w_2'' = 3 f    in Omega
- 3 w_1'' - 4 w_2'' = 7 f    in Omega

subject to

 w_1 = 0    on Gamma = {0, 1}
 w_2 = 0    on Gamma = {0, 1}
 
By construction the solution of the system is
    (w_1, w_2) = (u, u)

We then compare the solution provided by multiphenics
to the one provided by standard FEniCS.
"""

# Mesh generation
mesh = UnitIntervalMesh(MPI.comm_world, 32)

def left(x):
    return abs(x[0] - 0.) < finfo(float).eps

def right(x):
    return abs(x[0] - 1.) < finfo(float).eps
        
boundaries = MeshFunction("size_t", mesh, mesh.topology.dim - 1, 0)
boundaries.mark(left, 1)
boundaries.mark(right, 1)
boundaries_1 = where(boundaries.values == 1)[0]

x0 = SpatialCoordinate(mesh)[0]

# Solver parameters
solver_parameters = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}

def run_standard():
    # Define a function space
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define problems forms
    a = inner(grad(u), grad(v))*dx + u*v*dx
    f = 100*sin(20*x0)*v*dx

    # Define boundary conditions
    zero = Function(V)
    with zero.vector.localForm() as zero_local:
        zero_local.set(0.0)
    bc = DirichletBC(V, zero, boundaries_1)
    
    # Solve the linear system
    u = Function(V)
    solve(a == f, u, bc, petsc_options=solver_parameters)
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    # Return the solution
    return u
    
u = run_standard()

def run_standard_block():
    # Define a block function space
    V1 = FunctionSpace(mesh, ("Lagrange", 2))
    V2 = FunctionSpace(mesh, ("Lagrange", 2))
    (u1, u2) = (TrialFunction(V1), TrialFunction(V2))
    (v1, v2) = (TestFunction(V1), TestFunction(V2))

    # Define problem block forms
    aa = [[1*inner(grad(u1), grad(v1))*dx + 1*u1*v1*dx, 2*inner(grad(u2), grad(v1))*dx + 2*u2*v1*dx],
          [3*inner(grad(u1), grad(v2))*dx + 3*u1*v2*dx, 4*inner(grad(u2), grad(v2))*dx + 4*u2*v2*dx]]
    ff = [300*sin(20*x0)*v1*dx,
          700*sin(20*x0)*v2*dx]
    
    # Define block boundary conditions
    zero = Function(V1)
    with zero.vector.localForm() as zero_local:
        zero_local.set(0.0)
    bc1 = DirichletBC(V1, zero, boundaries_1)
    bc2 = DirichletBC(V2, zero, boundaries_1)
    bcs = [bc1, bc2]
    
    # Assemble the block linear system
    AA = assemble_matrix_block(aa, bcs)
    AA.assemble()
    FF = assemble_vector_block(ff, aa, bcs)
    
    # Solve the block linear system
    uu = create_vector_block(ff)
    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setOperators(AA)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setFromOptions()
    ksp.solve(FF, uu)
    (u1, u2) = (Function(V1), Function(V2))
    with u1.vector.localForm() as u1_local, u2.vector.localForm() as u2_local:
        (u1_local[:], u2_local[:]) = get_local_vectors(
            uu,
            [u1.function_space.dofmap.index_map, u2.function_space.dofmap.index_map]
        )
    u1.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    u2.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    # Return the block solution
    return u1, u2
    
u1, u2 = run_standard_block()

def run_multiphenics():
    # Define a block function space
    V = FunctionSpace(mesh, ("Lagrange", 2))
    VV = BlockFunctionSpace([V, V])
    uu = BlockTrialFunction(VV)
    vv = BlockTestFunction(VV)
    (u1, u2) = block_split(uu)
    (v1, v2) = block_split(vv)

    # Define problem block forms
    aa = [[1*inner(grad(u1), grad(v1))*dx + 1*u1*v1*dx, 2*inner(grad(u2), grad(v1))*dx + 2*u2*v1*dx],
          [3*inner(grad(u1), grad(v2))*dx + 3*u1*v2*dx, 4*inner(grad(u2), grad(v2))*dx + 4*u2*v2*dx]]
    ff = [300*sin(20*x0)*v1*dx,
          700*sin(20*x0)*v2*dx]
    
    # Define block boundary conditions
    zero = Function(V)
    with zero.vector.localForm() as zero_local:
        zero_local.set(0.0)
    bc1 = DirichletBC(VV.sub(0), zero, boundaries_1)
    bc2 = DirichletBC(VV.sub(1), zero, boundaries_1)
    bcs = BlockDirichletBC([bc1,
                            bc2])
    
    # Solve the block linear system
    uu = BlockFunction(VV)
    block_solve(aa, uu, ff, bcs, petsc_options=solver_parameters)
    uu1, uu2 = uu
    
    # Return the block solution
    return uu1, uu2
    
uu1, uu2 = run_multiphenics()

def compute_errors(u1, u2, uu1, uu2):
    u_1_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(u1), grad(u1))*dx)))
    u_2_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(u2), grad(u2))*dx)))
    err_1_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(u1 - uu1), grad(u1 - uu1))*dx)))
    err_2_norm = sqrt(MPI.sum(mesh.mpi_comm(), assemble_scalar(inner(grad(u2 - uu2), grad(u2 - uu2))*dx)))
    print("  Relative error for first component is equal to", err_1_norm/u_1_norm)
    print("  Relative error for second component is equal to", err_2_norm/u_2_norm)
    assert isclose(err_1_norm/u_1_norm, 0., atol=1.e-10)
    assert isclose(err_2_norm/u_2_norm, 0., atol=1.e-10)

print("Computing errors between standard and standard block")
compute_errors(u, u, u1, u2)
print("Computing errors between standard and multiphenics")
compute_errors(u, u, uu1, uu2)
print("Computing errors between standard block and multiphenics")
compute_errors(u1, u2, uu1, uu2)

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

from numpy import ndarray as array
from petsc4py import PETSc
from multiphenics.fem import block_assemble, BlockDirichletBC, BlockDirichletBCLegacy, BlockForm, BlockForm1, BlockForm2
from multiphenics.function import BlockFunction

def block_solve(block_lhs, block_x, block_rhs, block_bcs=None, petsc_options=None):
    # Process inputs
    if isinstance(block_lhs, (array, list)):
        block_lhs = BlockForm(block_lhs)
    assert isinstance(block_lhs, BlockForm2)
    assert isinstance(block_x, BlockFunction)
    if isinstance(block_rhs, (array, list)):
        block_rhs = BlockForm(block_rhs)
    assert isinstance(block_rhs, BlockForm1)
    assert block_bcs is None or isinstance(block_bcs, BlockDirichletBC)
    # Assemble
    block_A = block_assemble(block_lhs)
    block_b = block_assemble(block_rhs)
    # Apply boundary conditions
    if block_bcs is not None:
        BlockDirichletBCLegacy.apply(block_bcs, block_A, 1.0)
        BlockDirichletBCLegacy.apply(block_bcs, block_b)
    # Store options
    options = PETSc.Options()
    if petsc_options is not None:
        for k, v in petsc_options.items():
            options.setValue("multiphenics_solve_" + k, v)
    # Solve
    solver = PETSc.KSP().create(block_x.block_function_space.mesh.mpi_comm())
    solver.setOptionsPrefix("multiphenics_solve_")
    solver.setFromOptions()
    solver.setOperators(block_A)
    solver.solve(block_b, block_x.block_vector)
    # Keep subfunctions up to date
    block_x.apply("to subfunctions")

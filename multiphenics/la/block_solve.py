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

from dolfin import PETScLUSolver
from multiphenics.la.block_petsc_matrix import BlockPETScMatrix
from multiphenics.la.block_petsc_vector import BlockPETScVector

def block_solve(block_A, block_x, block_b, linear_solver="default"):
    assert isinstance(block_A, BlockPETScMatrix)
    assert isinstance(block_x, BlockPETScVector)
    assert isinstance(block_b, BlockPETScVector)
    # Solve
    solver = PETScLUSolver(linear_solver)
    solver.solve(block_A, block_x, block_b)
    # Keep subfunctions up to date
    block_x.block_function().apply("to subfunctions")

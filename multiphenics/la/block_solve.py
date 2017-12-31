# Copyright (C) 2016-2018 by the multiphenics authors
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
from multiphenics.la.generic_block_matrix import GenericBlockMatrix
from multiphenics.la.generic_block_vector import GenericBlockVector

def block_solve(block_A, block_x, block_b):
    assert isinstance(block_A, GenericBlockMatrix)
    assert isinstance(block_x, GenericBlockVector)
    assert isinstance(block_b, GenericBlockVector)
    # Solve
    solver = PETScLUSolver("mumps")
    solver.solve(block_A, block_x, block_b)
    # Keep subfunctions up to date
    block_x.block_function().apply("to subfunctions")

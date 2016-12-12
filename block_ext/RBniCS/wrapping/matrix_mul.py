# Copyright (C) 2015-2016 by the RBniCS authors
# Copyright (C) 2016 by the block_ext authors
#
# This file is part of the RBniCS interface to block_ext.
#
# RBniCS and block_ext are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and block_ext are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and block_ext. If not, see <http://www.gnu.org/licenses/>.
#

from block_ext.RBniCS.function import Function
from RBniCS.backends.fenics.wrapping import matrix_mul_vector as fenics_matrix_mul_vector, vectorized_matrix_inner_vectorized_matrix as fenics_vectorized_matrix_inner_vectorized_matrix

def matrix_mul_vector(matrix, vector):
    if isinstance(vector, Function.Type()):
        vector = vector.block_vector()
    return matrix.matvec(vector)

def vectorized_matrix_inner_vectorized_matrix(matrix, other_matrix):
    assert len(matrix.blocks.shape) == 2
    assert len(other_matrix.blocks.shape) == 2
    assert matrix.blocks.shape[0] == other_matrix.blocks.shape[0]
    assert matrix.blocks.shape[1] == other_matrix.blocks.shape[1]
    output = 0.
    for I in matrix.blocks.shape[0]:
        for J in matrix.blocks.shape[1]:
            output += fenics_vectorized_matrix_inner_vectorized_matrix(matrix.blocks[I, J], other_matrix.blocks[I, J])
    return output
            

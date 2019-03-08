# Copyright (C) 2016-2019 by the multiphenics authors
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

from numbers import Number
from numpy import empty
from multiphenics.python import cpp

BlockPETScMatrix = cpp.la.BlockPETScMatrix

# Preserve _bcs_zero_off_block_diagonal attribute for BCs application
def preserve_bcs_zero_off_block_diagonal__first_arg_matrix_second_arg_number(operator):
    original_operator = getattr(BlockPETScMatrix, operator)
    def custom_operator(self, other):
        output = original_operator(self, other)
        if isinstance(other, Number):
            assert hasattr(self, "_bcs_zero_off_block_diagonal")
            output._bcs_zero_off_block_diagonal = self._bcs_zero_off_block_diagonal
        return output
    setattr(BlockPETScMatrix, operator, custom_operator)
    
for operator in ("__mul__", "__rmul__", "__imul__", "__truediv__", "__itruediv__"):
    preserve_bcs_zero_off_block_diagonal__first_arg_matrix_second_arg_number(operator)
    
def preserve_bcs_zero_off_block_diagonal__first_and_second_args_matrices(operator):
    original_operator = getattr(BlockPETScMatrix, operator)
    def custom_operator(self, other):
        output = original_operator(self, other)
        assert hasattr(self, "_bcs_zero_off_block_diagonal")
        assert hasattr(other, "_bcs_zero_off_block_diagonal")
        assert len(self._bcs_zero_off_block_diagonal) == len(other._bcs_zero_off_block_diagonal)
        assert all(len(self._bcs_zero_off_block_diagonal[I]) == len(other._bcs_zero_off_block_diagonal[I]) for I in range(len(self._bcs_zero_off_block_diagonal)))
        bcs_zero_off_block_diagonal = empty((len(self._bcs_zero_off_block_diagonal), len(self._bcs_zero_off_block_diagonal[0])), dtype=bool)
        for I in range(bcs_zero_off_block_diagonal.shape[0]):
            for J in range(bcs_zero_off_block_diagonal.shape[1]):
                bcs_zero_off_block_diagonal[I, J] = self._bcs_zero_off_block_diagonal[I][J] or other._bcs_zero_off_block_diagonal[I][J]
        output._bcs_zero_off_block_diagonal = bcs_zero_off_block_diagonal.tolist()
        return output
    setattr(BlockPETScMatrix, operator, custom_operator)
    
for operator in ("__add__", "__iadd__", "__sub__", "__isub__"):
    preserve_bcs_zero_off_block_diagonal__first_and_second_args_matrices(operator)

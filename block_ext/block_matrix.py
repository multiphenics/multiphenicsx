# Copyright (C) 2016-2017 by the block_ext authors
#
# This file is part of block_ext.
#
# block_ext is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# block_ext is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with block_ext. If not, see <http://www.gnu.org/licenses/>.
#

from block import block_mat, block_vec

BlockMatrix = block_mat

# Do not use block_compose in algebraic operators
def custom__add__(self, other):
    if isinstance(other, block_mat):
        output = self.copy()
        for (block_index_I, block_output_I) in enumerate(output):
            for (block_index_J, block_output_IJ) in enumerate(block_output_I):
                block_output_IJ += other[block_index_I, block_index_J]
        return output
    else:
        return NotImplemented
BlockMatrix.__add__ = custom__add__

def custom__radd__(self, other):
    if isinstance(other, block_mat):
        return other.__add__(self)
    else:
        return NotImplemented
BlockMatrix.__radd__ = custom__radd__

def custom__sub__(self, other):
    if isinstance(other, block_mat):
        output = self.copy()
        for (block_index_I, block_output_I) in enumerate(output):
            for (block_index_J, block_output_IJ) in enumerate(block_output_I):
                block_output_IJ -= other[block_index_I, block_index_J]
        return output
    else:
        return NotImplemented
BlockMatrix.__sub__ = custom__sub__

def custom__rsub__(self, other):
    if isinstance(other, block_mat):
        return other.__sub__(self)
    else:
        return NotImplemented
BlockMatrix.__rsub__ = custom__rsub__

def custom__mul__(self, other):
    if isinstance(other, block_mat):
        return NotImplemented
    elif isinstance(other, block_vec):
        return self.matvec(other)
    elif isinstance(other, (float, int)):
        output = self.copy()
        for (block_index_I, block_output_I) in enumerate(output):
            for (block_index_J, block_output_IJ) in enumerate(block_output_I):
                block_output_IJ *= other
        return output
    else:
        return NotImplemented
BlockMatrix.__mul__ = custom__mul__

def custom__rmul__(self, other):
    if isinstance(other, block_mat):
        return other.__mul__(self)
    elif isinstance(other, (float, int)):
        return self.__mul__(other)
    else:
        return NotImplemented
BlockMatrix.__rmul__ = custom__rmul__

def custom__neg__(self):
    return self.__mul__(self, -1.)
BlockMatrix.__neg__ = custom__neg__

# Preserve _block_discard_dofs attribute in algebraic operators
def preserve_block_discard_dofs_attribute(operator):
    original_operator = getattr(BlockMatrix, operator)
    def custom_operator(self, other):
        assert hasattr(self, "_block_discard_dofs")
        output = original_operator(self, other)
        output._block_discard_dofs = self._block_discard_dofs
        return output
    setattr(BlockMatrix, operator, custom_operator)
    
for operator in ("__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__neg__"):
    preserve_block_discard_dofs_attribute(operator)
    

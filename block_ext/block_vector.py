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

from block import block_vec

BlockVector = block_vec

# Preserve _block_discard_dofs attribute in algebraic operators
def preserve_block_discard_dofs_attribute(operator):
    original_operator = getattr(BlockVector, operator)
    def custom_operator(self, other):
        assert hasattr(self, "_block_discard_dofs")
        output = original_operator(self, other)
        assert self._block_discard_dofs == other._block_discard_dofs
        output._block_discard_dofs = self._block_discard_dofs
        return output
    setattr(BlockVector, operator, custom_operator)
    
for operator in ("__add__", "__radd__", "__iadd__", "__sub__", "__rsub__", "__isub__", "__mul__", "__rmul__", "__imul__"):
    preserve_block_discard_dofs_attribute(operator)

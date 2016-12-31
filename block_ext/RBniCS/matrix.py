# Copyright (C) 2015-2017 by the RBniCS authors
# Copyright (C) 2016-2017 by the block_ext authors
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

from block_ext.block_matrix import BlockMatrix
from block_ext.RBniCS.function import Function

def Matrix():
    raise NotImplementedError("This is dummy function (not required by the interface) just store the Type")
    
def _Matrix_Type():
    return BlockMatrix
Matrix.Type = _Matrix_Type

# Enable matrix*function product (i.e. matrix*function.block_vector())
original__mul__ = BlockMatrix.__mul__
def custom__mul__(self, other):
    if isinstance(other, Function.Type()):
        return original__mul__(self, other.block_vector())
    else:
        return original__mul__(self, other)
BlockMatrix.__mul__ = custom__mul__

# Preserve generator attribute in algebraic operators, as required by DEIM
def preserve_generator_attribute(operator):
    original_operator = getattr(BlockMatrix, operator)
    def custom_operator(self, other):
        if hasattr(self, "generator"):
            output = original_operator(self, other)
            output.generator = self.generator
            return output
        else:
            return original_operator(self, other)
    setattr(BlockMatrix, operator, custom_operator)
    
for operator in ("__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__"):
    preserve_generator_attribute(operator)

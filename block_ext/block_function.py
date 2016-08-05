# Copyright (C) 2016 by the block_ext authors
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

from dolfin import Function
from block_vector import BlockVector

class BlockFunction(list):
    def __init__(self, arg1):
        assert isinstance(arg1, list)
        if isinstance(arg1[0], Function):
            functions = arg1
            vectors = [f.vector() for f in arg1]
        else:
            functions = []
            vectors = []
            for V in arg1:
                current_function = Function(V)
                functions.append(current_function)
                vectors.append(current_function.vector())
        list.__init__(self, functions)
        self._block_vector = BlockVector(vectors)
        
    def block_vector(self):
        return self._block_vector
        
    def block_split(self):
        return self
        
    def copy(self, deepcopy=False):
        assert deepcopy # no usage envisioned for the other case
        functions = [f.copy(deepcopy) for f in self]
        return BlockFunction(functions)
            
        
        

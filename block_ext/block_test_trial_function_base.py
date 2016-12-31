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

import types
from ufl import Argument
from dolfin import FunctionSpace
from block_ext.block_function_space import BlockFunctionSpace

class BlockTestTrialFunction_Base(tuple):
    def __new__(cls, arg1, Generator):
        assert isinstance(arg1, (list, tuple))
        if isinstance(arg1[0], FunctionSpace):
            return tuple.__new__(cls, [Generator(V) for V in arg1])
        else:
            assert isinstance(arg1[0], Argument)
            return tuple.__new__(cls, arg1)
        
    def __init__(self, arg1, Generator):
        # Store block function space
        if isinstance(arg1, BlockFunctionSpace):
            self._block_function_space = arg1
        else:
            self._block_function_space = BlockFunctionSpace([v.function_space() for v in self])
            
        # Make sure to add a block_function_space also to each single trial/test function
        def block_function_space(self_):
            return self._block_function_space
        for v in self:
            v.block_function_space = types.MethodType(block_function_space, v)
            
    def block_function_space(self):
        return self._block_function_space
        

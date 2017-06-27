# Copyright (C) 2016-2017 by the multiphenics authors
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

import types
from ufl import Argument
from dolfin import FunctionSpace
from multiphenics.function.block_function_space import BlockFunctionSpace

def TestTrialFunction_Base(function_space, Generator, part=None, block_function_space=None, block_index=None):
    assert isinstance(function_space, FunctionSpace)
    assert isinstance(block_function_space, BlockFunctionSpace) or block_function_space is None
    assert isinstance(block_index, int) or block_index is None
    
    # Generate trial/test function
    v = Generator(function_space, part)

    # Store block function space
    assert (block_function_space is None) == (block_index is None)
    if block_function_space is not None:
        assert block_index is not None
        v._block_function_space = block_function_space
        v._block_index = block_index
    else:
        v._block_function_space = None
        v._block_index = None
    
    if block_function_space is not None:
        # Add a block_function_space method
        def block_function_space(self_):
            return v._block_function_space
        v.block_function_space = types.MethodType(block_function_space, v)
    
    if block_index is not None:
        # Add a block_index method
        def block_index(self_):
            return v._block_index
        v.block_index = types.MethodType(block_index, v)
            
    # Return
    return v
        

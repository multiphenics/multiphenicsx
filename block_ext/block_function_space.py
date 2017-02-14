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

from dolfin import FunctionSpace
from block_replace_zero import zeros

class BlockFunctionSpace(tuple):
    def __new__(cls, arg1, arg2=None, keep=None):
        if isinstance(arg1, (list, tuple)):
            assert arg2 is None
            function_spaces = arg1
        else:
            assert arg2 is not None
            mesh = arg1
            block_element = arg2
            function_spaces = []
            for e in block_element:
                function_spaces.append(FunctionSpace(mesh, e))
                
        return tuple.__new__(cls, function_spaces)
        
    def __init__(self, arg1, arg2=None, keep=None):
        if keep is not None:
            self.keep = BlockFunctionSpace(keep)
        else:
            self.keep = None
        
    def sub(self, i):
        return self[i]
        
def extract_block_function_space(block_form, size):
    block_function_space = dict()
        
    assert len(size) in (1, 2)
    if len(size) == 2:
        for I in range(size[0]):
            block_function_space_I = extract_block_function_space(block_form[I], (size[1], ))
            for number in (0, 1):
                if number in block_function_space_I:
                    if number in block_function_space:
                        assert block_function_space[number] == block_function_space_I[number]
                    else:
                        block_function_space[number] = block_function_space_I[number]
    else:
        for I in range(size[0]):
            if block_form[I] in zeros or block_form[I].empty():
                continue
            for (index, arg) in enumerate(block_form[I].arguments()):
                number = arg.number()
                try:
                    block_function_space_arg = arg.block_function_space()
                except AttributeError: # Arguments were defined without using Block{Trial,Test}Function
                    block_function_space_arg = None
                finally:
                    if number in block_function_space:
                        assert block_function_space[number] == block_function_space_arg
                    else:
                        block_function_space[number] = block_function_space_arg
                        
    return block_function_space
        

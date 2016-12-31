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

from dolfin import Function, FunctionSpace
from block_ext.block_vector import BlockVector
from block_ext.block_function_space import BlockFunctionSpace
from block_ext.block_discard_dofs import BlockDiscardDOFs

class BlockFunction(tuple):
    def __new__(cls, arg1, arg2=None):
        def new_from_block_function(cls, block_function):
            return tuple.__new__(cls, block_function)
            
        def new_from_block_function_space(cls, block_function_space):
            functions = list()
            for V in block_function_space:
                current_function = Function(V)
                functions.append(current_function)
            return tuple.__new__(cls, functions)
            
        def new_from_block_function_space_and_block_vector(cls, block_function_space, block_vector):
            functions = list()
            for (V, vec) in zip(block_function_space, block_vector):
                current_function = Function(V, vec)
                functions.append(current_function)
            return tuple.__new__(cls, functions)
            
        def new_from_functions(cls, functions):
            return tuple.__new__(cls, functions)
        
        assert isinstance(arg1, (list, tuple, BlockFunction, BlockFunctionSpace))
        assert isinstance(arg2, BlockVector) or arg2 is None
        if isinstance(arg1, BlockFunction):
            return new_from_block_function(cls, arg1)
        elif isinstance(arg1, BlockFunctionSpace):
            if arg2 is not None:
                return new_from_block_function_space_and_block_vector(cls, arg1, arg2)
            else:
                return new_from_block_function_space(cls, arg1)
        elif isinstance(arg1, (list, tuple)):
            assert isinstance(arg1[0], (Function, FunctionSpace))
            if isinstance(arg1[0], Function):
                return new_from_functions(cls, arg1)
            elif isinstance(arg1[0], FunctionSpace):
                arg1 = BlockFunctionSpace(arg1)
                if arg2 is not None:
                    return new_from_block_function_space_and_block_vector(cls, arg1, arg2)
                else:
                    return new_from_block_function_space(cls, arg1)
            else: # impossible to arrive here anyway, thanks to the assert
                raise AssertionError("Invalid arguments in BlockFunction constructor.")
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in BlockFunction constructor.")
        
    def __init__(self, arg1, arg2=None):
        def init_from_block_function(self, block_function):
            self._block_vector = block_function.block_vector()
            self._block_function_space = block_function.block_function_space()
            
        def init_from_block_function_space(self, block_function_space):
            self._block_vector = BlockVector([f.vector() for f in self])
            self._block_function_space = block_function_space
            
        def init_from_block_function_space_and_block_vector(self, block_function_space, block_vector):
            self._block_vector = block_vector
            self._block_function_space = block_function_space
            
        def init_from_functions(self, _):
            self._block_vector = BlockVector([f.vector() for f in self])
            self._block_function_space = BlockFunctionSpace([f.function_space() for f in self])
        
        assert isinstance(arg1, (list, tuple, BlockFunction, BlockFunctionSpace))
        assert isinstance(arg2, BlockVector) or arg2 is None
        if isinstance(arg1, BlockFunction):
            init_from_block_function(self, arg1)
        elif isinstance(arg1, BlockFunctionSpace):
            if arg2 is not None:
                init_from_block_function_space_and_block_vector(self, arg1, arg2)
            else:
                init_from_block_function_space(self, arg1)
        elif isinstance(arg1, (list, tuple)):
            assert isinstance(arg1[0], (Function, FunctionSpace))
            if isinstance(arg1[0], Function):
                init_from_functions(self, arg1)
            elif isinstance(arg1[0], FunctionSpace):
                arg1 = BlockFunctionSpace(arg1)
                if arg2 is not None:
                    init_from_block_function_space_and_block_vector(self, arg1, arg2)
                else:
                    init_from_block_function_space(self, arg1)
            else: # impossible to arrive here anyway, thanks to the assert
                raise AssertionError("Invalid arguments in BlockFunction constructor.")
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in BlockFunction constructor.")
        
        # Setup block discard dofs, if any
        if self._block_function_space.keep is not None:
            self._block_discard_dofs = BlockDiscardDOFs(self._block_function_space.keep, self._block_function_space)
        else:
            self._block_discard_dofs = None
        
    def block_vector(self):
        return self._block_vector

    def block_function_space(self):
        return self._block_function_space
                
    def block_split(self):
        return self
        
    def copy(self, deepcopy=False):
        assert deepcopy # no usage envisioned for the other case
        block_vector_copy = BlockVector([f.copy(deepcopy).vector() for f in self])
        return BlockFunction(self._block_function_space, block_vector_copy)
        
    def sub(self, i):
        return self[i]
        
    def __add__(self, other):
        if isinstance(other, BlockFunction):
            output = self.copy(deepcopy=True)
            for (block_fun_output, block_fun_other) in zip(output, other):
                block_fun_output.vector().add_local(block_fun_other.vector().array())
                block_fun_output.vector().apply("add")
            return output
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, BlockFunction):
            output = self.copy(deepcopy=True)
            for (block_fun_output, block_fun_other) in zip(output, other):
                block_fun_output.vector().add_local(- block_fun_other.vector().array())
                block_fun_output.vector().apply("add")
            return output
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, float):
            output = self.copy(deepcopy=True)
            for block_fun_output in output:
                block_fun_output.vector()._scale(other)
            return output
        else:
            return NotImplemented

    def __div__(self, other):
        if isinstance(other, float):
            output = self.copy(deepcopy=True)
            for block_fun_output in output:
                block_fun_output.vector()._scale(1./other)
            return output
        else:
            return NotImplemented

    def __truediv__(self, other):
        return self.__div__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        output = self.__sub__(other)
        output.__imul__(-1.0)
        return output

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rdiv__(self, other):
        return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, BlockFunction):
            for (block_fun_output, block_fun_other) in zip(self, other):
                block_fun_output.vector().add_local(block_fun_other.vector().array())
                block_fun_output.vector().apply("add")
            return self
        else:
            return NotImplemented

    def __isub__(self, other):
        if isinstance(other, BlockFunction):
            for (block_fun_output, block_fun_other) in zip(self, other):
                block_fun_output.vector().add_local(- block_fun_other.vector().array())
                block_fun_output.vector().apply("add")
            return self
        else:
            return NotImplemented

    def __imul__(self, other):
        if isinstance(other, float):
            for block_fun_output in self:
                block_fun_output.vector()._scale(other)
            return self
        else:
            return NotImplemented

    def __idiv__(self, other):
        if isinstance(other, float):
            for block_fun_output in self:
                block_fun_output.vector()._scale(1./other)
            return self
        else:
            return NotImplemented

    def __itruediv__(self, other):
        return self.__idiv__(other)


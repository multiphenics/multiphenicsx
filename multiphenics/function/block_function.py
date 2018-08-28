# Copyright (C) 2016-2018 by the multiphenics authors
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
import dolfin
from dolfin import Function
from multiphenics.python import cpp
from multiphenics.function.block_function_space import BlockFunctionSpace
from multiphenics.la.as_backend_type import as_backend_type
from multiphenics.la.generic_block_vector import GenericBlockVector

def unwrap_sub_functions(sub_functions):
    return [sub_function._cpp_object for sub_function in sub_functions]
    
BlockFunction_Base = cpp.function.BlockFunction

class BlockFunction(object):
    def __init__(self, *args, **kwargs):
        """Initialize BlockFunction."""
        assert len(args) in (1, 2, 3)
        
        assert isinstance(args[0], BlockFunctionSpace)
        block_V = args[0]
        if len(args) == 1:
            # If passing only the BlockFunctionSpace
            self._init_from_block_function_space(block_V)
        elif len(args) == 2:
            # If passing BlockFunctionSpace together with a block vector
            if isinstance(args[1], GenericBlockVector):
                self._init_from_block_function_space_and_block_vector(block_V, args[1])
            # If passing BlockFunctionSpace together with a BlockFunction_Base
            elif isinstance(args[1], BlockFunction_Base):
                self._init_from_block_function_space_and_cpp_block_function(block_V, args[1])
            # If passing BlockFunctionSpace together with a list of subfunctions
            elif isinstance(args[1], list) and isinstance(args[1][0], Function):
                self._init_from_block_function_space_and_sub_functions(block_V, args[1])
            else:
                raise TypeError("Invalid arguments")
        elif len(args) == 3:
            # If passing BlockFunctionSpace together with a block vector and list of subfunctions
            assert isinstance(args[1], GenericBlockVector)
            assert isinstance(args[2], list)
            assert isinstance(args[2][0], Function)
            self._init_from_block_function_space_and_block_vector_and_sub_functions(block_V, args[1], args[2])
        else:
            raise TypeError("Too many arguments")

    def _init_from_block_function_space(self, block_V):
        self._cpp_object = BlockFunction_Base(block_V.cpp_object())
        self._block_function_space = block_V
        self._init_sub_functions()

    def _init_from_block_function_space_and_block_vector(self, block_V, block_vec):
        self._cpp_object = BlockFunction_Base(block_V.cpp_object(), block_vec)
        self._block_function_space = block_V
        self._init_sub_functions()
        
    def _init_from_block_function_space_and_cpp_block_function(self, block_V, cpp_object):
        self._cpp_object = cpp_object
        self._block_function_space = block_V
        self._init_sub_functions()
        
    def _init_from_block_function_space_and_sub_functions(self, block_V, sub_functions):
        self._cpp_object = BlockFunction_Base(block_V.cpp_object(), unwrap_sub_functions(sub_functions))
        self._block_function_space = block_V
        self._num_sub_spaces = block_V.num_sub_spaces()
        assert len(sub_functions) == self._num_sub_spaces
        self._sub_functions = sub_functions
        
    def _init_from_block_function_space_and_block_vector_and_sub_functions(self, block_V, block_vec, sub_functions):
        self._cpp_object = BlockFunction_Base(block_V.cpp_object(), block_vec, unwrap_sub_functions(sub_functions))
        self._block_function_space = block_V
        self._num_sub_spaces = block_V.num_sub_spaces()
        assert len(sub_functions) == self._num_sub_spaces
        self._sub_functions = sub_functions
        
    def _init_sub_functions(self):
        def extend_sub_function(sub_function, i):
            # Make sure to preserve a reference to the block function
            def block_function(self_):
                return self
            sub_function.block_function = types.MethodType(block_function, sub_function)
            
            # ... and a reference to the block index
            def block_index(self_):
                return i
            sub_function.block_index = types.MethodType(block_index, sub_function)
            
            # ... and that these methods are preserved by sub_function.sub()
            original_sub = sub_function.sub
            def sub(self_, j, deepcopy=False):
                output = original_sub(j, deepcopy)
                extend_sub_function(output, i)
                return output
            sub_function.sub = types.MethodType(sub, sub_function)
            
        self._num_sub_spaces = self.block_function_space().num_sub_spaces()
        self._sub_functions = list()
        for i in range(self._num_sub_spaces):
            # Extend with the python layer of dolfin's Function
            sub_function = dolfin.Function(self._cpp_object.sub(i))
            
            # Extend with block function and block index methods
            extend_sub_function(sub_function, i)
            
            # Append
            self._sub_functions.append(sub_function)
            
    def __len__(self):
        "Return the number of sub functions"
        return self._num_sub_spaces
        
    def __getitem__(self, i):
        """
        Return a sub function, *neglecting* restrictions.

        The sub functions are numbered from i = 0..N-1, where N is the
        total number of sub spaces.

        *Arguments*
            i : int
                The number of the sub function

        """
        assert isinstance(i, (int, slice))
        if isinstance(i, int):
            return self.sub(i)
        elif isinstance(i, slice):
            output = list()
            for j in range(i.start or 0, i.stop or len(self), i.step or 1):
                output.append(self.sub(j))
            return output
        
    def block_function_space(self):
        """
        Return block function space

        *Returns*
            _BlockFunctionSpace_
                The block subspace.
        """
        
        return self._block_function_space
        
    def block_vector(self):
        """
        Return block vector
        """
        
        def extend_block_vector(block_vector):
            # Make sure to preserve a reference to the block function
            def block_function(self_):
                return self
            block_vector.block_function = types.MethodType(block_function, block_vector)
            
        block_vector = self._cpp_object.block_vector()
        block_vector = as_backend_type(block_vector)
        extend_block_vector(block_vector)
        
        return block_vector
        
    def ufl_element(self):
        return self._block_function_space.ufl_element()

    def sub(self, i, deepcopy=False):
        """
        Return a sub function, *neglecting* restrictions.

        The sub functions are numbered from i = 0..N-1, where N is the
        total number of sub spaces.

        *Arguments*
            i : int
                The number of the sub function

        """
        if not isinstance(i, int):
            raise TypeError("expects an 'int' as first argument")
        if i >= self._num_sub_spaces:
            raise RuntimeError("Can only extract subfunctions with i = 0..%d"
                               % (self._num_sub_spaces - 1))

        assert deepcopy is False # no usage envisioned for the other case
        
        return self._sub_functions[i]

    def block_split(self, deepcopy=False):
        """
        Extract any sub functions.

        A sub function can be extracted from a discrete function that
        is in a mixed, vector, or tensor FunctionSpace. The sub
        function resides in the subspace of the mixed space.

        *Arguments*
            deepcopy
                Copy sub function vector instead of sharing

        """

        return tuple(self.sub(i, deepcopy) for i in range(self._num_sub_spaces))
        
    def __iter__(self):
        return self._sub_functions.__iter__()
        
    def copy(self, deepcopy=False):
        """
        Return a copy of itself
        *Arguments*
            deepcopy (bool)
                If false (default) the dof vector is shared.
        *Returns*
             _BlockFunction_
                 The BlockFunction
        """
        assert deepcopy is True # no usage envisioned for the other case
        return BlockFunction(self.block_function_space(), self.block_vector().copy())
        
    def __str__(self):
        return str([str(subf) for subf in self._sub_functions])
        
    def apply(self, mode, only=None):
        if only is None:
            only = -1
        self._cpp_object.apply(mode, only)

    def __add__(self, other):
        if isinstance(other, BlockFunction):
            output = self.copy(deepcopy=True)
            for (block_fun_output, block_fun_other) in zip(output, other):
                block_fun_output.vector().add_local(block_fun_other.vector().get_local())
                block_fun_output.vector().apply("add")
            output.apply("from subfunctions")
            return output
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, BlockFunction):
            output = self.copy(deepcopy=True)
            for (block_fun_output, block_fun_other) in zip(output, other):
                block_fun_output.vector().add_local(- block_fun_other.vector().get_local())
                block_fun_output.vector().apply("add")
            output.apply("from subfunctions")
            return output
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, float):
            output = self.copy(deepcopy=True)
            for block_fun_output in output:
                block_fun_output.vector()[:] *= other
            output.apply("from subfunctions")
            return output
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, float):
            output = self.copy(deepcopy=True)
            for block_fun_output in output:
                block_fun_output.vector()[:] /= other
            output.apply("from subfunctions")
            return output
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        output = self.__sub__(other)
        output.__imul__(-1.0)
        return output

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, BlockFunction):
            for (block_fun_output, block_fun_other) in zip(self, other):
                block_fun_output.vector().add_local(block_fun_other.vector().get_local())
                block_fun_output.vector().apply("add")
            self.apply("from subfunctions")
            return self
        else:
            return NotImplemented

    def __isub__(self, other):
        if isinstance(other, BlockFunction):
            for (block_fun_output, block_fun_other) in zip(self, other):
                block_fun_output.vector().add_local(- block_fun_other.vector().get_local())
                block_fun_output.vector().apply("add")
            self.apply("from subfunctions")
            return self
        else:
            return NotImplemented

    def __imul__(self, other):
        if isinstance(other, float):
            for block_fun_output in self:
                block_fun_output.vector()[:] *= other
            self.apply("from subfunctions")
            return self
        else:
            return NotImplemented

    def __itruediv__(self, other):
        if isinstance(other, float):
            for block_fun_output in self:
                block_fun_output.vector()[:] /= other
            self.apply("from subfunctions")
            return self
        else:
            return NotImplemented

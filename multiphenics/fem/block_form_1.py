# Copyright (C) 2016-2020 by the multiphenics authors
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

from numpy import empty
from ufl import Form
from dolfin.cpp.fem import Form as cpp_Form
from dolfin.fem.assembling import _create_cpp_form
from multiphenics.fem.block_replace_zero import block_replace_zero, _is_zero
from multiphenics.python import cpp

BlockForm1_Base = cpp.fem.BlockForm1

class BlockForm1(BlockForm1_Base):
    def __init__(self, block_form, block_function_space, form_compiler_parameters=None):
        # Store UFL form
        self._block_form = block_form
        # Store block function space
        assert len(block_function_space) == 1
        self._block_function_space = block_function_space
        # Replace UFL form by Dolfin form before passing it to the constructor
        # (note that we assume that block_form has been already preprocessed,
        #  so we can assume that nested blocks have been unrolled and zero
        #  placeholders have been replaced by zero forms)
        N = len(block_form)
        replaced_block_form = empty((N, ), dtype=object)
        for I in range(N):
            replaced_block_form[I] = block_replace_zero(block_form, (I, ), block_function_space)
            assert isinstance(replaced_block_form[I], Form) or _is_zero(replaced_block_form[I])
            if isinstance(replaced_block_form[I], Form):
                replaced_block_form[I] = _create_cpp_form(
                    form=replaced_block_form[I],
                    form_compiler_parameters=form_compiler_parameters
                )
            elif _is_zero(replaced_block_form[I]):
                assert isinstance(replaced_block_form[I], cpp_Form)
            else:
                raise TypeError("Invalid form")
        BlockForm1_Base.__init__(self, replaced_block_form.tolist(), [block_function_space_._cpp_object for block_function_space_ in block_function_space])
        # Store size for len and shape method
        self.N = N
        
    def __len__(self):
        return self.N
        
    @property
    def shape(self):
        return (self.N, )
        
    def __getitem__(self, i):
        assert isinstance(i, int)
        return self._block_form[i]
        
    def block_function_spaces(self):
        return self._block_function_space
        
    def __str__(self):
        vector_of_str = empty((self.N, ), dtype=object)
        for I in range(self.N):
            vector_of_str[I] = str(self._block_form[I])
        return str(vector_of_str)
        
    def __add__(self, other):
        if isinstance(other, BlockForm1):
            assert self.N == other.N
            assert self._block_function_space[0] is other._block_function_space[0]
            output_block_form = empty((self.N, ), dtype=object)
            for I in range(self.N):
                assert isinstance(self[I], Form) or _is_zero(self[I])
                assert isinstance(other[I], Form) or _is_zero(other[I])
                if (
                    isinstance(self[I], Form)
                        and
                    isinstance(other[I], Form)
                ):
                    output_block_form[I] = self[I] + other[I]
                elif (
                    isinstance(self[I], Form)
                        and
                    _is_zero(other[I])
                ):
                    output_block_form[I] = self[I]
                elif (
                    isinstance(other[I], Form)
                        and
                    _is_zero(self[I])
                ):
                    output_block_form[I] = other[I]
                elif (
                    _is_zero(self[I])
                        and
                    _is_zero(other[I])
                ):
                    output_block_form[I] = 0
                else:
                    raise TypeError("Invalid form")
            return BlockForm1(output_block_form, self._block_function_space)
        else:
            return NotImplemented
            
    def __sub__(self, other):
        return self + (-1.*other)
        
    def __radd__(self, other):
        return self.__add__(other)
        
    def __rsub__(self, other):
        return -1.*self.__sub__(other)
        
    def __rmul__(self, other):
        if isinstance(other, float):
            output_block_form = empty((self.N, ), dtype=object)
            for I in range(self.N):
                assert isinstance(self[I], Form) or _is_zero(self[I])
                if isinstance(self[I], Form):
                    output_block_form[I] = other*self[I]
                elif _is_zero(self[I]):
                    output_block_form[I] = 0
                else:
                    raise TypeError("Invalid form")
            return BlockForm1(output_block_form, self._block_function_space)
        else:
            return NotImplemented
    
    def __neg__(self):
        return -1.*self

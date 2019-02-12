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
from ufl.algorithms import expand_derivatives
from ufl.algorithms.analysis import has_exact_type
from ufl.classes import CoefficientDerivative
from dolfin.fem.assemble import _create_cpp_form
from dolfin.cpp.fem import Form as cpp_Form
from multiphenics.fem.block_form_1 import BlockForm1
from multiphenics.fem.block_replace_zero import block_replace_zero, _is_zero
from multiphenics.function import BlockFunction
from multiphenics.python import cpp

BlockForm2_Base = cpp.fem.BlockForm2

class BlockForm2(BlockForm2_Base):
    def __init__(self, block_form, block_function_space):
        # Store UFL form
        self._block_form = block_form
        # Store block function space
        assert len(block_function_space) == 2
        self._block_function_space = block_function_space
        # Replace UFL form by Dolfin form before passing it to the constructor
        # (note that we assume that block_form has been already preprocessed,
        #  so we can assume that nested blocks have been unrolled and zero
        #  placeholders have been replaced by zero forms)
        N = len(block_form)
        M = len(block_form[0])
        assert all([len(block_form_I) == M for block_form_I in block_form])
        replaced_block_form = empty((N, M), dtype=object)
        for I in range(N):
            for J in range(M):
                if isinstance(block_form[I, J], Form) and has_exact_type(block_form[I, J], CoefficientDerivative):
                    block_form[I, J] = expand_derivatives(block_form[I, J])
                replaced_block_form[I, J] = block_replace_zero(block_form, (I, J), block_function_space)
                assert isinstance(replaced_block_form[I, J], Form) or _is_zero(replaced_block_form[I, J])
                if isinstance(replaced_block_form[I, J], Form):
                    replaced_block_form[I, J] = _create_cpp_form(
                        form=replaced_block_form[I, J]
                    )
                elif _is_zero(replaced_block_form[I, J]):
                    assert isinstance(replaced_block_form[I, J], cpp_Form)
                else:
                    raise TypeError("Invalid form")
        BlockForm2_Base.__init__(self, replaced_block_form.tolist(), [block_function_space_._cpp_object for block_function_space_ in block_function_space])
        # Store sizes for shape method
        self.N = N
        self.M = M
    
    @property
    def shape(self):
        return (self.N, self.M)
        
    def __getitem__(self, ij):
        assert isinstance(ij, tuple)
        assert len(ij) == 2
        return self._block_form[ij]
        
    def block_function_spaces(self):
        return self._block_function_space
        
    def __str__(self):
        matrix_of_str = empty((self.N, self.M), dtype=object)
        for I in range(self.N):
            for J in range(self.M):
                matrix_of_str[I, J] = str(self._block_form[I, J])
        return str(matrix_of_str)
        
    def __add__(self, other):
        if isinstance(other, BlockForm2):
            assert self.N == other.N
            assert self.M == other.M
            assert self._block_function_space[0] is other._block_function_space[0]
            assert self._block_function_space[1] is other._block_function_space[1]
            output_block_form = empty((self.N, self.M), dtype=object)
            for I in range(self.N):
                for J in range(self.M):
                    assert isinstance(self[I, J], Form) or _is_zero(self[I, J])
                    assert isinstance(other[I, J], Form) or _is_zero(other[I, J])
                    if (
                        isinstance(self[I, J], Form)
                            and
                        isinstance(other[I, J], Form)
                    ):
                        output_block_form[I, J] = self[I, J] + other[I, J]
                    elif (
                        isinstance(self[I, J], Form)
                            and
                        _is_zero(other[I, J])
                    ):
                        output_block_form[I, J] = self[I, J]
                    elif (
                        isinstance(other[I, J], Form)
                            and
                        _is_zero(self[I, J])
                    ):
                        output_block_form[I, J] = other[I, J]
                    elif (
                        _is_zero(self[I, J])
                            and
                        _is_zero(other[I, J])
                    ):
                        output_block_form[I, J] = 0
                    else:
                        raise TypeError("Invalid form")
            return BlockForm2(output_block_form, self._block_function_space)
        else:
            return NotImplemented
            
    def __mul__(self, other):
        if isinstance(other, BlockFunction):
            assert self.M == other._num_sub_spaces
            assert self._block_function_space[1] is other._block_function_space
            output_block_form = empty((self.N, ), dtype=object)
            for I in range(self.N):
                non_zero_J = list()
                for J in range(self.M):
                    if isinstance(self[I, J], Form):
                        non_zero_J.append(J)
                if len(non_zero_J) > 0:
                    output_block_form[I] = self[I, non_zero_J[0]]*other[non_zero_J[0]]
                    for J in non_zero_J[1:]:
                        output_block_form[I] += self[I, J]*other[J]
                else:
                    output_block_form = 0
            return BlockForm1(output_block_form, [self._block_function_space[0]])
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
            output_block_form = empty((self.N, self.M), dtype=object)
            for I in range(self.N):
                for J in range(self.M):
                    assert isinstance(self[I, J], Form) or _is_zero(self[I, J])
                    if isinstance(self[I, J], Form):
                        output_block_form[I, J] = other*self[I, J]
                    elif _is_zero(self[I, J]):
                        output_block_form[I, J] = 0
                    else:
                        raise TypeError("Invalid form")
            return BlockForm2(output_block_form, self._block_function_space)
        else:
            return NotImplemented
    
    def __neg__(self):
        return -1.*self

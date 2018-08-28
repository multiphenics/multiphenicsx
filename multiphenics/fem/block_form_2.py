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

from numpy import empty
from dolfin import has_pybind11
from dolfin.fem.assembling import _create_dolfin_form
from multiphenics.python import cpp

if has_pybind11():
    BlockForm2_Base = cpp.fem.BlockForm2
else:
    BlockForm2_Base = cpp.BlockForm2

class BlockForm2(BlockForm2_Base):
    def __init__(self, block_form, block_function_space, form_compiler_parameters=None):
        # Store UFL form
        self._block_form = block_form
        # Store block function space
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
                replaced_block_form[I, J] = _create_dolfin_form(
                    form=block_form[I, J],
                    form_compiler_parameters=form_compiler_parameters
                )
        BlockForm2_Base.__init__(self, replaced_block_form.tolist(), [block_function_space_.cpp_object() for block_function_space_ in block_function_space])
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

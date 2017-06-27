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

from numpy import empty
from dolfin.fem.assembling import _create_dolfin_form
from multiphenics.swig import cpp

class BlockForm2(cpp.BlockForm2):
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
        replaced_block_form = empty((N, M), dtype=object)
        for I in range(N):
            for J in range(M):
                replaced_block_form[I, J] = _create_dolfin_form(
                    form=block_form[I, J],
                    form_compiler_parameters=form_compiler_parameters
                )
        cpp.BlockForm2.__init__(self, replaced_block_form.tolist(), block_function_space)
        
    def __getitem__(self, ij):
        assert isinstance(ij, tuple)
        assert len(ij) == 2
        return self._block_form[ij]
        
    def block_function_spaces(self):
        return self._block_function_space
        

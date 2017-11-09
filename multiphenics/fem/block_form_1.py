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
from multiphenics.python import cpp

class BlockForm1(cpp.BlockForm1):
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
        replaced_block_form = empty((N, ), dtype=object)
        for I in range(N):
            replaced_block_form[I] = _create_dolfin_form(
                form=block_form[I],
                form_compiler_parameters=form_compiler_parameters
            )
        cpp.BlockForm1.__init__(self, replaced_block_form.tolist(), block_function_space)
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

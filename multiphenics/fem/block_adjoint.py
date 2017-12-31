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

from numpy import ndarray as array, empty
from dolfin import adjoint
from multiphenics.fem.block_form import _block_form_preprocessing
from multiphenics.fem.block_form_2 import BlockForm2
from multiphenics.function import BlockTestFunction, BlockTrialFunction

def block_adjoint(block_form):
    assert isinstance(block_form, (array, list, BlockForm2))
    if isinstance(block_form, (array, list)):
        input_type = array
        (block_form, block_function_space, block_form_rank) = _block_form_preprocessing(block_form)
        assert block_form_rank is 2
        N = len(block_form)
        M = len(block_form[0])
        block_adjoint_function_space = [block_function_space[1], block_function_space[0]]
    else:
        input_type = BlockForm2
        N = block_form.block_size(0)
        M = block_form.block_size(1)
        block_adjoint_function_space = [block_form.block_function_spaces(1), block_form.block_function_spaces(0)]
    block_test_function_adjoint = BlockTestFunction(block_adjoint_function_space[0])
    block_trial_function_adjoint = BlockTrialFunction(block_adjoint_function_space[1])
    block_adjoint_form = empty((M, N), dtype=object)
    for I in range(N):
        for J in range(M):
            block_adjoint_form[J, I] = adjoint(block_form[I, J], (block_test_function_adjoint[J], block_trial_function_adjoint[I]))
    if input_type is array:
        return block_adjoint_form
    elif input_type is BlockForm2:
        return BlockForm2(block_adjoint_form, block_function_space=block_adjoint_function_space)

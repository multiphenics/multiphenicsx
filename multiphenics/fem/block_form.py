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

from numpy import ndarray as array
from ufl import Form
from multiphenics.fem.block_flatten_nested import block_flatten_nested
from multiphenics.fem.block_form_1 import BlockForm1
from multiphenics.fem.block_form_2 import BlockForm2
from multiphenics.fem.block_replace_zero import _get_block_form_rank, _is_zero
from multiphenics.function import BlockFunctionSpace

def BlockForm(block_form, block_function_space=None, block_form_rank=None):
    assert isinstance(block_form, (array, list, BlockForm1, BlockForm2))
    if isinstance(block_form, (array, list)):
        (replaced_block_form, block_function_space, block_form_rank) = \
            _block_form_preprocessing(block_form, block_function_space, block_form_rank)
        if block_form_rank == 2:
            return BlockForm2(replaced_block_form, block_function_space)
        elif block_form_rank == 1:
            return BlockForm1(replaced_block_form, block_function_space)
    else:
        return block_form
        
def _block_form_preprocessing(block_form, block_function_space=None, block_form_rank=None):
    assert isinstance(block_form, (array, list))
    if block_form_rank is None:
        block_form_rank = _get_block_form_rank(block_form)
        assert block_form_rank is not None, \
            "A block form rank should be provided when assemblying a zero block vector/matrix."
    assert block_form_rank in (1, 2)
    if block_form_rank == 2:
        # Extract BlockFunctionSpace from the current form, if required
        if not block_function_space:
            assert not all([_is_zero(block_form_I_J) for block_form_I in block_form for block_form_I_J in block_form_I]), \
                "A BlockFunctionSpace should be provided when assemblying a zero block matrix."
            block_function_space = _extract_block_function_space_2(block_form)
            assert len(block_function_space) == 2
            assert block_function_space[0] is not None
            assert block_function_space[1] is not None
            block_function_space = [block_function_space[0], block_function_space[1]] # convert from dict to list
        else:
            assert isinstance(block_function_space, list)
            assert len(block_function_space) == 2
            assert isinstance(block_function_space[0], BlockFunctionSpace)
            assert isinstance(block_function_space[1], BlockFunctionSpace)
        
        # Flatten nested blocks, if any
        block_form = block_flatten_nested(block_form, block_function_space)
        
        # Return preprocessed data
        return (block_form, block_function_space, block_form_rank)
    elif block_form_rank == 1:
        # Extract BlockFunctionSpace from the current form, if required
        if not block_function_space:
            assert not all([_is_zero(block_form_I) for block_form_I in block_form]), \
                "A BlockFunctionSpace should be provided when assemblying a zero block vector."
            block_function_space = _extract_block_function_space_1(block_form)
            assert len(block_function_space) == 1
            assert block_function_space[0] is not None
            block_function_space = [block_function_space[0]] # convert from dict to list
        else:
            assert isinstance(block_function_space, BlockFunctionSpace)
            block_function_space = [block_function_space]
        
        # Flatten nested blocks, if any
        block_form = block_flatten_nested(block_form, block_function_space)
        
        # Return preprocessed data
        return (block_form, block_function_space, block_form_rank)

def _extract_block_function_space_2(block_form):
    block_function_space = dict()
    
    for block_form_I in block_form:
        block_function_space_I = _extract_block_function_space_1(block_form_I)
        for (number, block_function_space_number) in block_function_space_I.items():
            if number in block_function_space:
                assert block_function_space[number] == block_function_space_number
            else:
                block_function_space[number] = block_function_space_number
                        
    return block_function_space
    
def _extract_block_function_space_1(block_form):
    block_function_space = dict()
    
    for block_form_I in block_form:
        if _is_zero(block_form_I):
            continue
        elif isinstance(block_form_I, (array, list)):
            if isinstance(block_form_I[0], list) or isinstance(block_form_I[0], array):
                block_function_space_recursive = _extract_block_function_space_2(block_form_I)
            else:
                block_function_space_recursive = _extract_block_function_space_1(block_form_I)
            for (number, block_function_space_number) in block_function_space_recursive.items():
                if number in block_function_space:
                    assert block_function_space[number] == block_function_space_number
                else:
                    block_function_space[number] = block_function_space_number
        else:
            assert isinstance(block_form_I, Form)
            for (index, arg) in enumerate(block_form_I.arguments()):
                number = arg.number()
                block_function_space_arg = arg.block_function_space()
                if number in block_function_space:
                    assert block_function_space[number] == block_function_space_arg
                else:
                    block_function_space[number] = block_function_space_arg
                        
    return block_function_space

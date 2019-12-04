# Copyright (C) 2016-2019 by the multiphenics authors
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

from numpy import ndarray as array, zeros
from ufl import Argument, Form
from ufl.algorithms.traversal import iter_expressions
from ufl.corealg.traversal import traverse_unique_terminals
from multiphenics.fem.block_replace_zero import _get_block_form_rank, _is_zero

def block_flatten_nested(block_form, block_function_space):
    assert isinstance(block_form, (array, list))
    assert isinstance(block_function_space, list)
    assert len(block_function_space) in (1, 2)
    if len(block_function_space) == 2:
        N = block_function_space[0].num_sub_spaces()
        M = block_function_space[1].num_sub_spaces()
        flattened_block_form = zeros((N, M), dtype=object)
        for block_form_I_nested in block_form:
            for block_form_IJ_nested in block_form_I_nested:
                _flatten_nested_2(block_form_IJ_nested, flattened_block_form, block_function_space)
        return flattened_block_form
    elif len(block_function_space) == 1:
        N = block_function_space[0].num_sub_spaces()
        flattened_block_form = zeros((N, ), dtype=object)
        for block_form_I_nested in block_form:
            _flatten_nested_1(block_form_I_nested, flattened_block_form, block_function_space)
        return flattened_block_form
        
def _flatten_nested_2(form_or_block_form, flattened_block_form, block_function_space):
    is_zero = _is_zero(form_or_block_form)
    assert is_zero or isinstance(form_or_block_form, (array, Form, list))
    if is_zero:
        pass
    elif isinstance(form_or_block_form, Form):
        args = _extract_arguments(form_or_block_form)
        test_block_index = None
        test_block_function_space = None
        trial_block_index = None
        trial_block_function_space = None
        for arg in args:
            assert arg.number() in (0, 1)
            if arg.number() == 0:
                if test_block_index is not None:
                    assert test_block_index == arg.block_index(), "Test functions corresponding to different blocks appear in the same form."
                    assert test_block_function_space == arg.block_function_space(), "Test functions defined in different block function spaces appear in the same form."
                else:
                    test_block_index = arg.block_index()
                    test_block_function_space = arg.block_function_space()
            elif arg.number() == 1:
                if trial_block_index is not None:
                    assert trial_block_index == arg.block_index(), "Trial functions corresponding to different blocks appear in the same form."
                    assert trial_block_function_space == arg.block_function_space(), "Trial functions defined in different block function spaces appear in the same form."
                else:
                    trial_block_index = arg.block_index()
                    trial_block_function_space = arg.block_function_space()
        assert test_block_index is not None
        assert test_block_function_space is not None
        assert trial_block_index is not None
        assert trial_block_function_space is not None
        if hasattr(test_block_function_space, "is_block_subspace"):
            assert test_block_index in test_block_function_space.sub_components_to_components, "Block function space and test block index are not consistent on the sub space."
            test_block_index = test_block_function_space.sub_components_to_components[test_block_index]
            test_block_function_space = test_block_function_space.parent_block_function_space
        if hasattr(block_function_space[0], "is_block_subspace"):
            assert test_block_index in block_function_space[0].components_to_sub_components, "Block function space and test block index are not consistent on the sub space."
            test_block_index = block_function_space[0].components_to_sub_components[test_block_index]
            assert test_block_function_space == block_function_space[0].parent_block_function_space
        else:
            assert test_block_function_space == block_function_space[0]
        if hasattr(trial_block_function_space, "is_block_subspace"):
            assert trial_block_index in trial_block_function_space.sub_components_to_components, "Block function space and trial block index are not consistent on the sub space."
            trial_block_index = trial_block_function_space.sub_components_to_components[trial_block_index]
            trial_block_function_space = trial_block_function_space.parent_block_function_space
        if hasattr(block_function_space[1], "is_block_subspace"):
            assert trial_block_index in block_function_space[1].components_to_sub_components, "Block function space and trial block index are not consistent on the sub space."
            trial_block_index = block_function_space[1].components_to_sub_components[trial_block_index]
            assert trial_block_function_space == block_function_space[1].parent_block_function_space
        else:
            assert trial_block_function_space == block_function_space[1]
        flattened_block_form[test_block_index, trial_block_index] += form_or_block_form
    elif isinstance(form_or_block_form, (array, list)):
        assert _get_block_form_rank(form_or_block_form) == 2
        for block_form_I_nested in form_or_block_form:
            for block_form_IJ_nested in block_form_I_nested:
                _flatten_nested_2(block_form_IJ_nested, flattened_block_form, block_function_space)
    else:
        raise AssertionError("Invalid case in _flatten_nested_2")
        
def _flatten_nested_1(form_or_block_form, flattened_block_form, block_function_space):
    is_zero = _is_zero(form_or_block_form)
    assert is_zero or isinstance(form_or_block_form, (array, Form, list))
    if is_zero:
        pass
    elif isinstance(form_or_block_form, Form):
        args = _extract_arguments(form_or_block_form)
        test_block_index = None
        test_block_function_space = None
        for arg in args:
            assert arg.number() == 0
            if test_block_index is not None:
                assert test_block_index == arg.block_index(), "Test functions corresponding to different blocks appear in the same form."
                assert test_block_function_space == arg.block_function_space(), "Test functions defined in different block function spaces appear in the same form."
            else:
                test_block_index = arg.block_index()
                test_block_function_space = arg.block_function_space()
        assert test_block_index is not None
        assert test_block_function_space is not None
        if hasattr(test_block_function_space, "is_block_subspace"):
            assert test_block_index in test_block_function_space.sub_components_to_components, "Block function space and test block index are not consistent on the sub space."
            test_block_index = test_block_function_space.sub_components_to_components[test_block_index]
            test_block_function_space = test_block_function_space.parent_block_function_space
        if hasattr(block_function_space[0], "is_block_subspace"):
            assert test_block_index in block_function_space[0].components_to_sub_components, "Block function space and test block index are not consistent on the sub space."
            test_block_index = block_function_space[0].components_to_sub_components[test_block_index]
            assert test_block_function_space == block_function_space[0].parent_block_function_space
        else:
            assert test_block_function_space == block_function_space[0]
        flattened_block_form[test_block_index] += form_or_block_form
    elif isinstance(form_or_block_form, (array, list)):
        assert _get_block_form_rank(form_or_block_form) == 1
        for block_form_I_nested in form_or_block_form:
            _flatten_nested_1(block_form_I_nested, flattened_block_form, block_function_space)
    else:
        raise AssertionError("Invalid case in _flatten_nested_1")
    
def _extract_arguments(form):
    # This is a copy of extract_type in ufl.algorithms.analysis
    # without wrapping the result in a set
    return [o for e in iter_expressions(form) for o in traverse_unique_terminals(e) if isinstance(o, Argument)]
    
def _assert_flattened_form_2_is_square(block_form):
    N = len(block_form)
    M = len(block_form[0])
    assert N == M
    for n in range(N):
        assert len(block_form[n]) == M

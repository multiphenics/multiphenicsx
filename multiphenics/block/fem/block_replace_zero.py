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

from numpy import ndarray as array, asarray as to_numpy
from ufl import Form
from ufl.algorithms import expand_derivatives
from ufl.algorithms.analysis import has_exact_type
from ufl.classes import CoefficientDerivative
from dolfin import Constant, div, dx, inner, tr
from multiphenics.block.function.test_function import TestFunction
from multiphenics.block.function.trial_function import TrialFunction

zeros = (0, 0.)

def block_replace_zero(block_form, index, block_function_space):
    assert len(index) in (1, 2)
    if len(index) == 2:
        I = index[0]
        J = index[1]
        assert (
            isinstance(block_form[I][J], Form) # this function is always called after flattening, so it cannot be an array or list
                or 
            (isinstance(block_form[I][J], (float, int)) and block_form[I][J] in zeros)
        )
        if block_form[I][J] in zeros:
            block_form_IJ = block_form[I][J]
        elif has_exact_type(block_form[I][J], CoefficientDerivative):
            block_form_IJ = expand_derivatives(block_form[I][J])
        else:
            block_form_IJ = block_form[I][J]
        if block_form_IJ in zeros or block_form_IJ.empty():
            block_form_IJ = _get_zero_form(block_function_space, (I, J))
        else:
            assert not block_form_IJ.empty()
        return block_form_IJ
    else:
        I = index[0]
        assert (
            isinstance(block_form[I], Form) # this function is always called after flattening, so it cannot be an array or list
                or 
            (isinstance(block_form[I], (float, int)) and block_form[I] in zeros)
        )
        block_form_I = block_form[I]
        if block_form_I in zeros:
            block_form_I = _get_zero_form(block_function_space, (I, ))
        else:
            assert not block_form_I.empty()
        return block_form_I
        
def _is_zero(form_or_block_form):
    assert (
        isinstance(form_or_block_form, (array, Form, list)) 
            or 
        (isinstance(form_or_block_form, (float, int)) and form_or_block_form in zeros)
    )
    if isinstance(form_or_block_form, Form):
        return form_or_block_form.empty()
    elif isinstance(form_or_block_form, (array, list)):
        block_form_rank = _get_block_form_rank(form_or_block_form)
        assert block_form_rank in (None, 1, 2)
        if block_form_rank is 2:
            for block_form_I in form_or_block_form:
                for block_form_IJ in block_form_I:
                    if not _is_zero(block_form_IJ):
                        return False
            return True
        elif block_form_rank is 1:
            for block_form_I in form_or_block_form:
                if not _is_zero(block_form_I):
                    return False
            return True
        elif block_form_rank is None:
            return True
    elif isinstance(form_or_block_form, (float, int)) and form_or_block_form in zeros:
        return True
    else:
        raise AssertionError("Invalid case in _is_zero")
    
def _get_block_form_rank(form_or_block_form):
    assert (
        isinstance(form_or_block_form, (array, Form, list)) 
            or 
        (isinstance(form_or_block_form, (float, int)) and form_or_block_form in zeros)
    )
    if isinstance(form_or_block_form, Form):
        if form_or_block_form.empty():
            return None
        else:
            return len(form_or_block_form.arguments())
    elif isinstance(form_or_block_form, (array, list)):
        if isinstance(form_or_block_form, list):
            form_or_block_form = to_numpy(form_or_block_form, dtype=object)
        form_or_block_form = form_or_block_form.flatten()
        block_form_rank = None
        for sub_form_or_block_form in form_or_block_form:
            current_block_form_rank = _get_block_form_rank(sub_form_or_block_form)
            if current_block_form_rank is not None:
                if block_form_rank is not None:
                    assert block_form_rank == current_block_form_rank
                else:
                    block_form_rank = current_block_form_rank
        return block_form_rank
    elif isinstance(form_or_block_form, (float, int)) and form_or_block_form in zeros:
        return None
    else:
        raise AssertionError("Invalid case in _get_block_form_rank")
    
def _get_zero_form(block_function_space, index):
    zero = Constant(0.)
    assert len(index) in (1, 2)
    if len(index) == 2:
        test = TestFunction(block_function_space[0][index[0]], block_function_space=block_function_space[0], block_index=index[0])
        trial = TrialFunction(block_function_space[1][index[1]], block_function_space=block_function_space[1], block_index=index[1])
        len_test_shape = len(test.ufl_shape)
        len_trial_shape = len(trial.ufl_shape)
        assert len_test_shape in (0, 1, 2)
        assert len_trial_shape in (0, 1, 2)
        if len_test_shape == 0 and len_trial_shape == 0:
            return zero*test*trial*dx
        elif len_test_shape == 0 and len_trial_shape == 1:
            return zero*test*div(trial)*dx
        elif len_test_shape == 0 and len_trial_shape == 2:
            return zero*test*tr(trial)*dx
        elif len_test_shape == 1 and len_trial_shape == 0:
            return zero*div(test)*trial*dx
        elif len_test_shape == 1 and len_trial_shape == 1:
            return zero*inner(test, trial)*dx
        elif len_test_shape == 1 and len_trial_shape == 2:
            return zero*div(test)*tr(trial)*dx
        elif len_test_shape == 2 and len_trial_shape == 0:
            return zero*tr(test)*trial*dx
        elif len_test_shape == 2 and len_trial_shape == 1:
            return zero*tr(test)*div(trial)*dx
        elif len_test_shape == 2 and len_trial_shape == 2:
            return zero*inner(test, trial)*dx
        else:
            raise AssertionError("Invalid case in _get_zero_form.")
    else:
        test = TestFunction(block_function_space[0][index[0]], block_function_space=block_function_space[0], block_index=index[0])
        len_test_shape = len(test.ufl_shape)
        assert len_test_shape in (0, 1, 2)
        if len_test_shape == 0:
            return zero*test*dx
        elif len_test_shape == 1:
            return zero*div(test)*dx
        elif len_test_shape == 2:
            return zero*tr(test)*dx
        else:
            raise AssertionError("Invalid case in _get_zero_form.")
    

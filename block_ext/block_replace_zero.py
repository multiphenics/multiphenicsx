# Copyright (C) 2016 by the block_ext authors
#
# This file is part of block_ext.
#
# block_ext is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# block_ext is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with block_ext. If not, see <http://www.gnu.org/licenses/>.
#

from ufl import Form
from ufl.algorithms import expand_derivatives
from ufl.algorithms.analysis import has_exact_type
from ufl.classes import CoefficientDerivative
from dolfin import Constant, div, dx, inner, TestFunction, tr, TrialFunction
from block_ext.block_outer import BlockOuterForm1, BlockOuterForm2

zeros = (0, 0.)

def block_replace_zero(block_form, index, block_function_space):
    assert len(index) in (1, 2)
    if len(index) == 2:
        I = index[0]
        J = index[1]
        if isinstance(block_form[I][J], BlockOuterForm2):
            return block_form[I][J]
        elif block_form[I][J] in zeros:
            block_form_IJ = block_form[I][J]
        elif has_exact_type(block_form[I][J], CoefficientDerivative):
            block_form_IJ = expand_derivatives(block_form[I][J])
        else:
            block_form_IJ = block_form[I][J]
        assert isinstance(block_form_IJ, Form) or block_form_IJ in zeros
        if block_form_IJ in zeros or block_form_IJ.empty():
            return _get_zero_form(block_function_space, (I, J))
        else:
            return block_form_IJ
    else:
        I = index[0]
        if isinstance(block_form[I], BlockOuterForm1):
            return block_form[I]
        block_form_I = block_form[I]
        assert isinstance(block_form_I, Form) or block_form_I in zeros
        if block_form_I in zeros:
            return _get_zero_form(block_function_space, (I, ))
        else:
            assert not block_form_I.empty()
            return block_form_I
            
def _get_zero_form(block_function_space, index):
    zero = Constant(0.)
    assert len(index) in (1, 2)
    if len(index) == 2:
        test = TestFunction(block_function_space[index[0]])
        trial = TrialFunction(block_function_space[index[1]])
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
        test = TestFunction(block_function_space[index[0]])
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
    

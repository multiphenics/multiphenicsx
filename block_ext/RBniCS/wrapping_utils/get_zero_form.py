# Copyright (C) 2015-2016 by the RBniCS authors
# Copyright (C) 2016 by the block_ext authors
#
# This file is part of the RBniCS interface to block_ext.
#
# RBniCS and block_ext are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and block_ext are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and block_ext. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import Constant, dx
from RBniCS.backends.fenics.wrapping_utils import get_form_argument

def get_zero_rank_1_form(form):
    test = _get_form_argument_first_component(form, 0)
    return Constant(0.)*test*dx
    
def get_zero_rank_2_form(form):
    test = _get_form_argument_first_component(form, 0)
    trial = _get_form_argument_first_component(form, 1)
    return Constant(0.)*trial*test*dx
    
def _get_form_argument_first_component(form, number):
    arg = get_form_argument(form, number)
    arg_shape = len(arg.ufl_shape)
    assert arg_shape in (0, 1, 2)
    if arg_shape == 0:
        return arg
    elif arg_shape == 1:
        return arg[0]
    elif arg_shape == 2:
        return arg[0, 0]
    else: # impossible to arrive here anyway thanks to the assert
        raise AssertionError("Invalid argument to SeparatedParametrizedForm.__init__")


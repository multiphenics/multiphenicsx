# Copyright (C) 2016-2017 by the block_ext authors
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

from numpy import empty
from dolfin import derivative
from block_ext.block_replace_zero import block_replace_zero
from block_ext.block_function_space import extract_block_function_space
from block_ext.block_outer import BlockOuterForm1, BlockOuterForm2

def block_derivative(F, u, du):
    # Extract BlockFunctionSpace from the current form
    block_V = extract_block_function_space(F, (len(F),))
    
    # Compute the derivative
    assert len(F) == len(u) == len(du)
    J = empty((len(F), len(u)), dtype=object)
    for i in range(len(F)):
        for j in range(len(u)):
            F_i = block_replace_zero(F, (i,), block_V)
            if not isinstance(F_i, BlockOuterForm1):
                J[i, j] = derivative(F_i, u[j], du[j])
            else:
                J[i, j] = None
                block_outer_residual = F_i
                while block_outer_residual is not None:
                    outer_derivative = BlockOuterForm2((
                        F_i.forms[0],
                        derivative(F_i.forms[1], u[j], du[j])
                    ))
                    outer_derivative.scale = F_i.scale
                    if J[i, j] is None:
                        J[i, j] = outer_derivative
                    else:
                        J[i, j] += outer_derivative
                    if block_outer_residual.addend_form is not None:
                        J[i, j] += derivative(block_outer_residual.addend_form, u[j], du[j])
                    block_outer_residual = block_outer_residual.addend_block_outer_form
    return J

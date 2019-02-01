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

from numpy import ndarray as array, empty
from dolfin import derivative
from multiphenics.fem.block_form import _block_form_preprocessing
from multiphenics.fem.block_form_1 import BlockForm1
from multiphenics.fem.block_form_2 import BlockForm2
from multiphenics.fem.block_replace_zero import _is_zero

def block_derivative(F, u, du):
    assert isinstance(F, (array, list, BlockForm1))
    if isinstance(F, (array, list)):
        input_type = array
        (F, block_V, block_form_rank) = _block_form_preprocessing(F)
        assert block_form_rank == 1
    else:
        input_type = BlockForm1
        block_V = F.block_function_spaces()
    assert len(block_V) == 1
    block_V = block_V[0]
    
    # Compute the derivative
    assert len(F) == len(u) == len(du)
    J = empty((len(F), len(u)), dtype=object)
    for i in range(len(F)):
        assert not _is_zero(F[i]) # J[i, :] would be zero, resulting in a singular matrix
        for j in range(len(u)):
            J[i, j] = derivative(F[i], u[j], du[j])
    if input_type is array:
        return J
    elif input_type is BlockForm1:
        return BlockForm2(J, block_function_space=[du.block_function_space(), block_V])

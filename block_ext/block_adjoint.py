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

from numpy import ndarray as array, empty
from dolfin import adjoint

def block_adjoint(block_form):
    assert isinstance(block_form, list) or isinstance(block_form, array)
    N = len(block_form)
    assert isinstance(block_form[0], list) or isinstance(block_form[0], array)
    M = len(block_form[0])
    block_adjoint_form = empty((N, M), dtype=object)
    for I in range(N):
        for J in range(M):
            if isinstance(block_form, list):
                block_adjoint_form[I, J] = adjoint(block_form[I][J])
            elif isinstance(block_form, array):
                block_adjoint_form[I, J] = adjoint(block_form[I, J])
            else:
                raise AssertionError("Invalid case in block_adjoint.")
    return block_adjoint_form

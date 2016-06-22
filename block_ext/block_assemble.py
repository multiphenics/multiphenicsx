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

# We provide here a simplified versione of cbc.block block_assemble, that however
# allows for the optional tensor argument

from block_matrix import BlockMatrix
from block_vector import BlockVector
import numpy as np

def block_assemble(block_form, block_tensor=None):
    from dolfin import assemble
    N = len(block_form)
    if isinstance(block_form[0], list) or isinstance(block_form[0], np.ndarray):
        M = len(block_form[0])
        if block_tensor is None:
            block_tensor_was_None = True
            block_tensor = BlockMatrix(N, M)
        else:
            block_tensor_was_None = False
        for I in range(N):
            for J in range(M):
                if block_tensor_was_None:
                    block_tensor.blocks[I, J] = assemble(block_form[I][J])
                else:
                    assemble(block_form[I][J], tensor=block_tensor.blocks[I, J])
    else:
        if block_tensor is None:
            block_tensor_was_None = True
            block_tensor = BlockVector(N)
        else:
            block_tensor_was_None = False
        for I in range(N):
            if block_tensor_was_None:
                block_tensor.blocks[I] = assemble(block_form[I])
            else:
                assemble(block_form[I], tensor=block_tensor.blocks[I])
    
    return block_tensor
    
    

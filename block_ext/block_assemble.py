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

from numpy import ndarray as array
from dolfin import assemble
from block_ext.block_matrix import BlockMatrix
from block_ext.block_vector import BlockVector
from block_ext.block_discard_dofs import BlockDiscardDOFs

def block_assemble(block_form, block_tensor=None, **kwargs):
    assert isinstance(block_form, list) or isinstance(block_form, array)
    N = len(block_form)
    if isinstance(block_form[0], list) or isinstance(block_form[0], array):
        M = len(block_form[0])
        if block_tensor is None:
            block_tensor_was_None = True
            block_tensor = BlockMatrix(N, M)
        else:
            block_tensor_was_None = False
        # Assemble
        for I in range(N):
            for J in range(M):
                if block_tensor_was_None:
                    block_tensor.blocks[I, J] = assemble(block_form[I][J], **kwargs)
                else:
                    assemble(block_form[I][J], tensor=block_tensor.blocks[I, J], **kwargs)
        # Extract BlockFunctionSpace from the current form
        block_function_space = _extract_block_function_space(block_form, (N, M))
    else:
        if block_tensor is None:
            block_tensor_was_None = True
            block_tensor = BlockVector(N)
        else:
            block_tensor_was_None = False
        # Assemble
        for I in range(N):
            if block_tensor_was_None:
                block_tensor.blocks[I] = assemble(block_form[I], **kwargs)
            else:
                assemble(block_form[I], tensor=block_tensor.blocks[I], **kwargs)
        # Extract BlockFunctionSpace from the current form
        block_function_space = _extract_block_function_space(block_form, (N,))
    
    # Attach BlockDiscardDOFs to the assembled tensor
    if (
        block_function_space is not None
            and
        block_function_space.keep is not None
    ):
        block_tensor._block_discard_dofs = BlockDiscardDOFs(block_function_space.keep, block_function_space)
    else:
        block_tensor._block_discard_dofs = None
    
    return block_tensor
    
def _extract_block_function_space(form, size):
    assert len(size) in (1, 2)
    if len(size) == 2:
        block_function_space = None
        for I in range(size[0]):
            block_function_space_I = _extract_block_function_space(form[I], (size[1], ))
            if I == 0:
                block_function_space = block_function_space_I
            else:
                assert block_function_space == block_function_space_I
        return block_function_space
    else:
        block_function_space = None
        for I in range(size[0]):
            for (index, arg) in enumerate(form[I].arguments()):
                try:
                    block_function_space_arg = arg.block_function_space()
                except AttributeError: # Arguments were defined without using Block{Trial,Test}Function
                    block_function_space_arg = None
                finally:
                    if I == 0 and index == 0:
                        block_function_space = block_function_space_arg
                    else:
                        assert block_function_space == block_function_space_arg
        return block_function_space
            

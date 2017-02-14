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

# We provide here a simplified versione of cbc.block block_assemble, that however
# allows for the optional tensor argument

from numpy import ndarray as array, empty
from dolfin import assemble
from block_ext.block_matrix import BlockMatrix
from block_ext.block_vector import BlockVector
from block_ext.block_discard_dofs import BlockDiscardDOFs
from block_ext.block_replace_zero import block_replace_zero
from block_ext.block_function_space import extract_block_function_space
from block_ext.block_outer import BlockOuterForm1, BlockOuterForm2, BlockOuterMatrix, BlockOuterVector

def block_assemble(block_form, block_tensor=None, **kwargs):
    assert isinstance(block_form, list) or isinstance(block_form, array)
    N = len(block_form)
    if isinstance(block_form[0], list) or isinstance(block_form[0], array):
        M = len(block_form[0])
        
        # Preprocess kwargs because keep_diagonal is needed only on diagonal blocks
        block_kwargs = empty((N, M), dtype=object)
        for I in range(N):
            for J in range(M):
                block_kwargs[I, J] = dict(kwargs)
                if "keep_diagonal" in kwargs and I != J:
                    del block_kwargs[I, J]["keep_diagonal"]
        
        # Prepare storage
        if block_tensor is None:
            block_tensor_was_None = True
            block_tensor = BlockMatrix(N, M)
        else:
            block_tensor_was_None = False
            
        # Extract BlockFunctionSpace from the current form
        block_function_space = extract_block_function_space(block_form, (N, M))
        assert len(block_function_space) == 2
        
        # Assemble
        for I in range(N):
            for J in range(M):
                block_form_IJ = block_replace_zero(block_form, (I, J), block_function_space)
                if not isinstance(block_form_IJ, BlockOuterForm2):
                    if block_tensor_was_None:
                        block_tensor.blocks[I, J] = assemble(block_form_IJ, **block_kwargs[I, J])
                    else:
                        assemble(block_form_IJ, tensor=block_tensor.blocks[I, J], **block_kwargs[I, J])
                else:
                    if block_tensor_was_None:
                        block_tensor.blocks[I, J] = BlockOuterMatrix(block_form_IJ, **block_kwargs[I, J])
                    else:
                        block_tensor.blocks[I, J].assemble(block_form_IJ, **block_kwargs[I, J])
                        
        # Attach BlockDiscardDOFs to the assembled tensor
        assert (
            block_function_space[0] is not None
                and
            block_function_space[0].keep is not None
        ) == (
            block_function_space[1] is not None
                and
            block_function_space[1].keep is not None
        )
        if (
            block_function_space[0] is not None
                and
            block_function_space[0].keep is not None
        ):
            block_tensor._block_discard_dofs = (
                BlockDiscardDOFs(block_function_space[0].keep, block_function_space[0]), 
                BlockDiscardDOFs(block_function_space[1].keep, block_function_space[1])
            )
        else:
            block_tensor._block_discard_dofs = (None, None)
    else:
        # Prepare storage
        if block_tensor is None:
            block_tensor_was_None = True
            block_tensor = BlockVector(N)
        else:
            block_tensor_was_None = False
            
        # Extract BlockFunctionSpace from the current form
        block_function_space = extract_block_function_space(block_form, (N,))
        assert len(block_function_space) == 1
        
        # Assemble
        for I in range(N):
            block_form_I = block_replace_zero(block_form, (I, ), block_function_space)
            if not isinstance(block_form_I, BlockOuterForm1):
                if block_tensor_was_None:
                    block_tensor.blocks[I] = assemble(block_form_I, **kwargs)
                else:
                    assemble(block_form_I, tensor=block_tensor.blocks[I], **kwargs)
            else:
                if block_tensor_was_None:
                    block_tensor.blocks[I] = BlockOuterVector(block_form_I, **kwargs)
                else:
                    block_tensor.blocks[I].assemble(block_form_I, **kwargs)
    
        # Attach BlockDiscardDOFs to the assembled tensor
        if (
            block_function_space[0] is not None
                and
            block_function_space[0].keep is not None
        ):
            block_tensor._block_discard_dofs = BlockDiscardDOFs(block_function_space[0].keep, block_function_space[0])
        else:
            block_tensor._block_discard_dofs = None
    
    return block_tensor
    

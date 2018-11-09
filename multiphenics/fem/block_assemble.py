# Copyright (C) 2016-2020 by the multiphenics authors
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

from numpy import ndarray as array
from multiphenics.fem.block_assembler import BlockAssembler
from multiphenics.fem.block_form import BlockForm
from multiphenics.fem.block_form_1 import BlockForm1
from multiphenics.fem.block_form_2 import BlockForm2

def block_assemble(block_form,
                   block_tensor=None,
                   form_compiler_parameters=None,
                   add_values=False,
                   finalize_tensor=True,
                   keep_diagonal=False,
                   backend=None):

    # Create a block form, the provided one is a list of Forms
    if isinstance(block_form, (array, list)):
        block_form = BlockForm(block_form, form_compiler_parameters=form_compiler_parameters)
    else:
        assert isinstance(block_form, (BlockForm1, BlockForm2))

    # Create tensor
    comm = block_form.mesh().mpi_comm()
    block_tensor = _create_block_tensor(comm, block_form, block_form.rank(), block_tensor)
        
    # Call C++ assemble function
    block_assembler = BlockAssembler()
    block_assembler.add_values = add_values
    block_assembler.finalize_tensor = finalize_tensor
    block_assembler.keep_diagonal = keep_diagonal
    block_assembler.assemble(block_tensor, block_form)

    # Return value
    return block_tensor
    
def _create_block_tensor(comm, block_form, rank, block_tensor):
    block_tensor = None # TODO not required anymore? Just call the standard dolfin one?
    
    return block_tensor

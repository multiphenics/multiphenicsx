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
from multiphenics.fem.block_form import BlockForm
from multiphenics.fem.block_form_1 import BlockForm1
from multiphenics.fem.block_form_2 import BlockForm2
from multiphenics.python import cpp

def block_assemble(block_form,
                   block_tensor=None):

    # Create a block form, the provided one is a list of Forms
    if isinstance(block_form, (array, list)):
        block_form = BlockForm(block_form)
    else:
        assert isinstance(block_form, (BlockForm1, BlockForm2))

    # Call C++ assemble function
    if block_tensor is None:
        return cpp.fem.block_assemble(block_form)
    else:
        cpp.fem.block_assemble(block_tensor, block_form)
        return block_tensor

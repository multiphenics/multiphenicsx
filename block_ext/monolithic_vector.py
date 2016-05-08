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

from dolfin import PETScVector, as_backend_type
from petsc4py import PETSc
from monolithic_index_helper import MonolithicIndexHelper

class MonolithicVector(PETScVector):
    # The values of block_vector are not really used, just the
    # block sparsity pattern. vec object has been already allocated
    # by the MonolithicMatrix::create_monolithic_vectors method
    def __init__(self, block_vector, vec):
        # Outer dimensions
        N = block_vector.blocks.shape[0]
        
        # Inner dimensions
        n = []
        for I in range(N):
            block = as_backend_type(block_vector[I]).vec()
            n.append(block.getSize())
        
        # Store dimensions
        self.N, self.n = N, n
        
        # Init PETScVector
        PETScVector.__init__(self, vec)
        
    def block_add(self, block_vector):
        N, n = self.N, self.n
        assert N == block_vector.blocks.shape[0]
        
        for I in range(N):
            block = as_backend_type(block_vector[I]).vec()
            assert block.getSize() == n[I]
            row_start, row_end = block.getOwnershipRange()
            for i in range(row_start, row_end):
                val = block.array[i - row_start]
                self.vec().setValues(i + sum(n[:I]), val, addv=PETSc.InsertMode.ADD)
        self.vec().assemble()
        
    def copy_values_to(self, block_vector):
        N, n = self.N, self.n
        assert N == block_vector.blocks.shape[0]
        
        row_start, row_end = self.vec().getOwnershipRange()
        index_helper = MonolithicIndexHelper(row_start, N, n)
        I = index_helper.block_index_init()
        i = index_helper.local_index_init(I)
        for k in range(row_start, row_end):
            val = self.vec().array[k - row_start]
            block = as_backend_type(block_vector[I]).vec()
            block.setValues(i, val, addv=PETSc.InsertMode.INSERT)
            I, i = index_helper.indices_increment(I, i)
        for I in range(N):
            as_backend_type(block_vector[I]).vec().assemble()
            as_backend_type(block_vector[I]).vec().ghostUpdate()
        

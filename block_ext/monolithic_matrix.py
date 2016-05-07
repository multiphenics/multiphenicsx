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

from dolfin import PETScMatrix, as_backend_type
from monolithic_vector import MonolithicVector
from petsc4py import PETSc

class MonolithicMatrix(PETScMatrix):
    # The values of block_matrix are not really used, just the
    # block sparsity pattern
    def __init__(self, block_matrix):
        PETScMatrix.__init__(self)
        
        # Outer dimensions
        N, M = block_matrix.blocks.shape
        
        # Inner dimensions
        n, m = [], []
        for I in range(N):
            block = as_backend_type(block_matrix[I, 0]).mat()
            n.append(block.getSize()[0])
        for J in range(M):
            block = as_backend_type(block_matrix[0, J]).mat()
            m.append(block.getSize()[1])
            
        # Assert
        for I in range(N):
            for J in range(M):
                block = as_backend_type(block_matrix[I, J]).mat()
                assert block.getSize()[0] == n[I] and block.getSize()[1] == m[J]
        
        # Initialize PETScMatrix        
        self.mat().setType("aij")
        self.mat().setSizes([sum(n), sum(m)])
        self.mat().setUp()
        
        # Store dimensions
        self.N, self.M, self.n, self.m = N, M, n, m
        
    def block_add(self, block_matrix):
        N, M, n, m = self.N, self.M, self.n, self.m
        assert N, M == block_matrix.blocks.shape
        
        for I in range(N):
            for J in range(M):
                block = as_backend_type(block_matrix[I, J]).mat()
                assert block.getSize()[0] == n[I] and block.getSize()[1] == m[J]
                row_start, row_end = block.getOwnershipRange()
                for i in range(row_start, row_end):
                    cols, vals = block.getRow(i)
                    cols[:] += sum(m[:J])
                    self.mat().setValues(i + sum(n[:I]), cols, vals, addv=PETSc.InsertMode.ADD)
        self.mat().assemble()
    
    def create_monolithic_vectors(self, block_x, block_b):
        petsc_x, petsc_b = self.mat().createVecs()
        return MonolithicVector(block_x, petsc_x), MonolithicVector(block_b, petsc_b)
        

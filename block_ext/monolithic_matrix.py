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
from monolithic_index_helper import MonolithicIndexHelper
from petsc4py import PETSc
from mpi4py import MPI

class MonolithicMatrix(PETScMatrix):
    # The values of block_matrix are not really used, just the
    # block sparsity pattern
    def __init__(self, block_matrix, mat=None, preallocate=True):
        # Outer dimensions
        M, N = block_matrix.blocks.shape
        
        # Inner dimensions
        m, n = [], []
        for I in range(M):
            block = as_backend_type(block_matrix[I, 0]).mat()
            m.append(block.getSize()[0])
        for J in range(N):
            block = as_backend_type(block_matrix[0, J]).mat()
            n.append(block.getSize()[1])
            
        # Assert
        for I in range(M):
            for J in range(N):
                block = as_backend_type(block_matrix[I, J]).mat()
                assert block.getSize()[0] == m[I] and block.getSize()[1] == n[J]

        # Store dimensions
        self.M, self.N, self.m, self.n = M, N, m, n
                
        # Initialize PETScMatrix
        if mat is None:
            PETScMatrix.__init__(self)
        else:
            PETScMatrix.__init__(self, mat)
                    
        if self.mat().size[0] < 0: # uninitialized matrix
            import numpy
            # Get local size as the sum of the local sizes of the block matrices
            ownership_range_sum = 0
            for I in range(M):
                block = as_backend_type(block_matrix[I, 0]).mat() # is the same for all J = 0, ..., N - 1
                row_start, row_end = block.getOwnershipRange()
                ownership_range_sum += row_end - row_start
            # Preallocation:
            if preallocate:
                # Auxiliary variables for parallel communication
                comm = self.mat().comm.tompi4py()
                comm_size = comm.size
                comm_rank = comm.rank
                # List local sizes for all ranks with parallel communication
                all_ownership_range_sum = []
                for r in range(comm_size):
                    all_ownership_range_sum.append( comm.bcast(ownership_range_sum, root=r) )
                # Define ownership ranges for self
                ownership_range_start = [sum(all_ownership_range_sum[: r   ]) for r in range(comm_size)]
                ownership_range_end   = [sum(all_ownership_range_sum[:(r+1)]) for r in range(comm_size)]
                # Get the number of diagonal and off-diagonal elements
                d_nnz, o_nnz = [], []
                for r in range(comm_size):
                    d_nnz.append( numpy.zeros(ownership_range_end[r] - ownership_range_start[r], dtype='i') )
                    o_nnz.append( numpy.zeros(ownership_range_end[r] - ownership_range_start[r], dtype='i') )
                for I in range(M):
                    for J in range(N):
                        block = as_backend_type(block_matrix[I, J]).mat()
                        row_start, row_end = block.getOwnershipRange()
                        # index helper on parallel partitioning blocks rather than PDEs blocks
                        index_helper = MonolithicIndexHelper(sum(m[:I]) + row_start, comm_size, all_ownership_range_sum)
                        A = index_helper.block_index_init()
                        a = index_helper.local_index_init(A)
                        for i in range(row_start, row_end):
                            cols, _ = block.getRow(i)
                            cols[:] += sum(n[:J])
                            for c in cols:
                                if c >= ownership_range_start[A] and c < ownership_range_end[A]:
                                    d_nnz[A][a] += 1
                                else:
                                    o_nnz[A][a] += 1
                            A, a = index_helper.indices_increment(A, a)
                # Parallel communication of d_nnz, o_nnz
                local_d_nnz = numpy.zeros(ownership_range_end[comm_rank] - ownership_range_start[comm_rank], dtype='i')
                local_o_nnz = numpy.zeros(ownership_range_end[comm_rank] - ownership_range_start[comm_rank], dtype='i')
                for r in range(comm_size):
                    comm.Reduce(d_nnz[r], local_d_nnz, root=r, op=MPI.SUM)
                    comm.Reduce(o_nnz[r], local_o_nnz, root=r, op=MPI.SUM)
            # Initialize
            self.mat().setType("aij")
            self.mat().setSizes([(ownership_range_sum, sum(n)), (ownership_range_sum, sum(m))])
            if preallocate:
                self.mat().setPreallocationNNZ([local_d_nnz, local_o_nnz])
                self.mat().setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, True)
                self.mat().setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
                self.mat().setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
            self.mat().setUp()
        
    def block_add(self, block_matrix):
        M, N, m, n = self.M, self.N, self.m, self.n
        assert M, N == block_matrix.blocks.shape
        
        for I in range(M):
            for J in range(N):
                block = as_backend_type(block_matrix[I, J]).mat()
                assert block.getSize()[0] == m[I] and block.getSize()[1] == n[J]
                row_start, row_end = block.getOwnershipRange()
                for i in range(row_start, row_end):
                    cols, vals = block.getRow(i)
                    cols[:] += sum(n[:J])
                    self.mat().setValues(i + sum(m[:I]), cols, vals, addv=PETSc.InsertMode.ADD)
        self.mat().assemble()
            
    def create_monolithic_vectors(self, block_x, block_b):
        petsc_x, petsc_b = self.mat().createVecs()
        return MonolithicVector(block_x, petsc_x), MonolithicVector(block_b, petsc_b)
        

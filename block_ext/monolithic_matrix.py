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
from block_ext.monolithic_vector import MonolithicVector
from petsc4py import PETSc
from mpi4py import MPI

class MonolithicMatrix(PETScMatrix):
    # The values of block_matrix are not really used, just the
    # block sparsity pattern
    def __init__(self, block_matrix, mat=None, preallocate=True, block_discard_dofs=None):
        # Outer dimensions
        M, N = block_matrix.blocks.shape
        
        # Inner dimensions
        m, n = [], []
        for I in range(M):
            block = as_backend_type(block_matrix[I, 0]).mat()
            current_m = block.getSize()[0]
            # Make sure that all row dimensions are consistent
            for J in range(1, N):
                block = as_backend_type(block_matrix[I, J]).mat()
                assert block.getSize()[0] == current_m
            if block_discard_dofs is not None and block_discard_dofs.need_to_discard_dofs[I]:
                current_m -= len(block_discard_dofs.dofs_to_be_discarded[I])
            m.append(current_m)
        for J in range(N):
            block = as_backend_type(block_matrix[0, J]).mat()
            current_n = block.getSize()[1]
            # Make sure that all row dimensions are consistent
            for I in range(1, M):
                block = as_backend_type(block_matrix[I, J]).mat()
                assert block.getSize()[1] == current_n
            if block_discard_dofs is not None and block_discard_dofs.need_to_discard_dofs[J]:
                current_n -= len(block_discard_dofs.dofs_to_be_discarded[J])
            n.append(current_n)
            
        # Store dimensions
        self.M, self.N, self.m, self.n = M, N, m, n
                
        # Initialize PETScMatrix
        if mat is None:
            PETScMatrix.__init__(self)
        else:
            PETScMatrix.__init__(self, mat)
                    
        if self.mat().size[0] < 0: # uninitialized matrix
            import numpy as np
            # Auxiliary variables for parallel communication
            comm = self.mat().comm.tompi4py()
            comm_size = comm.size
            comm_rank = comm.rank
            # Get local size as the sum of the local sizes of the block matrices
            ownership_range_sum = 0
            for I in range(M):
                block = as_backend_type(block_matrix[I, 0]).mat() # is the same for all J = 0, ..., N - 1
                row_start, row_end = block.getOwnershipRange()
                ownership_range_sum += row_end - row_start
            # Subtract rows to be discarded
            if block_discard_dofs is not None:
                total_rows_to_be_discarded = np.sum([len(block_discard_dofs.dofs_to_be_discarded[I]) for I in range(M) if block_discard_dofs.need_to_discard_dofs[I]])
                rows_to_be_discarded_per_process = total_rows_to_be_discarded/comm_size # integer division
                if comm_rank == comm_size - 1:
                	rows_to_be_discarded_per_process = total_rows_to_be_discarded - (comm_size - 1)*rows_to_be_discarded_per_process
                ownership_range_sum -= rows_to_be_discarded_per_process
            # Preallocation:
            if preallocate:
                # List local sizes for all ranks with parallel communication
                all_ownership_range_sum = []
                for r in range(comm_size):
                    all_ownership_range_sum.append( comm.bcast(ownership_range_sum, root=r) )
                # Define ownership ranges for self
                ownership_range_start = np.array([sum(all_ownership_range_sum[: r   ]) for r in range(comm_size)], dtype='i')
                ownership_range_end   = np.array([sum(all_ownership_range_sum[:(r+1)]) for r in range(comm_size)], dtype='i')
                # Get the number of diagonal and off-diagonal elements
                d_nnz, o_nnz = [], []
                for r in range(comm_size):
                    d_nnz.append( np.zeros(ownership_range_end[r] - ownership_range_start[r], dtype='i') )
                    o_nnz.append( np.zeros(ownership_range_end[r] - ownership_range_start[r], dtype='i') )
                for I in range(M):
                    if block_discard_dofs is not None and block_discard_dofs.need_to_discard_dofs[I]:
                        row_reposition_dofs = block_discard_dofs.subspace_dofs_extended[I]
                    else:
                        row_reposition_dofs = None
                    for J in range(N):
                        if block_discard_dofs is not None and block_discard_dofs.need_to_discard_dofs[J]:
                            col_reposition_dofs = block_discard_dofs.subspace_dofs_extended[J]
                        else:
                            col_reposition_dofs = None
                        block = as_backend_type(block_matrix[I, J]).mat()
                        row_start, row_end = block.getOwnershipRange()
                        for i in range(row_start, row_end):
                            if row_reposition_dofs is not None:
                                if i not in row_reposition_dofs:
                                    continue
                                row = row_reposition_dofs[i]
                            else:
                                row = i
                            row += sum(m[:I])
                            A = np.argmax(ownership_range_end > row)
                            a = row - ownership_range_start[A]
                            cols, _ = block.getRow(i)
                            if col_reposition_dofs is not None:
                                cols = set(cols).difference(block_discard_dofs.dofs_to_be_discarded[J])
                                cols = [col_reposition_dofs[c] for c in cols]
                                cols = np.array(cols)
                            cols[:] += sum(n[:J])
                            for c in cols:
                                if c >= ownership_range_start[A] and c < ownership_range_end[A]:
                                    d_nnz[A][a] += 1
                                else:
                                    o_nnz[A][a] += 1
                # Parallel communication of d_nnz, o_nnz
                local_d_nnz = np.zeros(ownership_range_end[comm_rank] - ownership_range_start[comm_rank], dtype='i')
                local_o_nnz = np.zeros(ownership_range_end[comm_rank] - ownership_range_start[comm_rank], dtype='i')
                for r in range(comm_size):
                    comm.Reduce(d_nnz[r], local_d_nnz, root=r, op=MPI.SUM)
                    comm.Reduce(o_nnz[r], local_o_nnz, root=r, op=MPI.SUM)
            # Initialize
            self.mat().setType("aij")
            assert sum(n) == sum(m) # otherwise you cannot use the same ownership range for columns
            self.mat().setSizes([(ownership_range_sum, sum(n)), (ownership_range_sum, sum(m))])
            if preallocate:
                self.mat().setPreallocationNNZ([local_d_nnz, local_o_nnz])
                self.mat().setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, True)
                self.mat().setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
                self.mat().setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
            self.mat().setUp()
            
        # Store dofs to be discarded while adding
        self.block_discard_dofs = block_discard_dofs
        
    def block_add(self, block_matrix):
        import numpy as np
        M, N, m, n = self.M, self.N, self.m, self.n
        assert M, N == block_matrix.blocks.shape
        block_discard_dofs = self.block_discard_dofs
        
        for I in range(M):
            if block_discard_dofs is not None and block_discard_dofs.need_to_discard_dofs[I]:
                row_reposition_dofs = block_discard_dofs.subspace_dofs_extended[I]
            else:
                row_reposition_dofs = None
            for J in range(N):
                if block_discard_dofs is not None and block_discard_dofs.need_to_discard_dofs[J]:
                    col_reposition_dofs = block_discard_dofs.subspace_dofs_extended[J]
                else:
                    col_reposition_dofs = None
                block = as_backend_type(block_matrix[I, J]).mat()
                row_start, row_end = block.getOwnershipRange()
                for i in range(row_start, row_end):
                    if row_reposition_dofs is not None:
                        if i not in row_reposition_dofs:
                            continue
                        row = row_reposition_dofs[i]
                    else:
                        row = i
                    row += sum(m[:I])
                    cols, vals = block.getRow(i)
                    if col_reposition_dofs is not None:
                        cols_to_vals = dict(zip(cols, vals))
                        cols_after_discard = set(cols).difference(block_discard_dofs.dofs_to_be_discarded[J])
                        cols = [col_reposition_dofs[c] for c in cols_after_discard]
                        vals = [cols_to_vals[c] for c in cols_after_discard]
                        assert len(cols) == len(vals)
                        if len(cols) == 0:
                            continue
                        cols = np.array(cols)
                        vals = np.array(vals)
                    cols[:] += sum(n[:J])
                    self.mat().setValues(row, cols, vals, addv=PETSc.InsertMode.ADD)
        self.mat().assemble()
            
    def create_monolithic_vectors(self, block_x, block_b):
        petsc_x, petsc_b = self.mat().createVecs()
        return (
            MonolithicVector(block_x, petsc_x, block_discard_dofs=None), # this vector is already sized on the space with discarded DOFs
            MonolithicVector(block_b, petsc_b, block_discard_dofs=self.block_discard_dofs)
        )
        
    def create_monolithic_vector_left(self, block_b):
        petsc_b = self.mat().createVecLeft()
        return MonolithicVector(block_b, petsc_b, block_discard_dofs=self.block_discard_dofs)
        

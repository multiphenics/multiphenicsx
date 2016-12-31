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

import numpy as np
from dolfin import PETScMatrix, as_backend_type
from block_ext.monolithic_vector import MonolithicVector
from block_ext.block_outer import BlockOuterMatrix
from petsc4py import PETSc
from mpi4py import MPI

class MonolithicMatrix(PETScMatrix):
    # The values of block_matrix are not really used, just the
    # block sparsity pattern
    def __init__(self, block_matrix, mat=None, preallocate=True, block_discard_dofs=None, block_constrain_dofs=None):
        # Outer dimensions
        M, N = block_matrix.blocks.shape
        
        # Inner dimensions
        m, n = [], []
        for I in range(M):
            if not isinstance(block_matrix[I, 0], BlockOuterMatrix):
                block = as_backend_type(block_matrix[I, 0]).mat()
                current_m = block.getSize()[0]
            else:
                block_vec_0 = as_backend_type(block_matrix[I, 0].vecs[0]).vec()
                current_m = block_vec_0.getSize()
            # Make sure that all row dimensions are consistent
            for J in range(N):
                if not isinstance(block_matrix[I, J], BlockOuterMatrix):
                    block = as_backend_type(block_matrix[I, J]).mat()
                    assert block.getSize()[0] == current_m
                else:
                    block_outer_matrix = block_matrix[I, J]
                    while block_outer_matrix is not None:
                        block_vec_0 = as_backend_type(block_outer_matrix.vecs[0]).vec()
                        assert block_vec_0.getSize() == current_m
                        if block_outer_matrix.addend_matrix is not None:
                            assert as_backend_type(block_outer_matrix.addend_matrix).mat().getSize()[0] == current_m
                        block_outer_matrix = block_outer_matrix.addend_block_outer_matrix
            if block_discard_dofs is not None and block_discard_dofs.need_to_discard_dofs[I]:
                current_m -= len(block_discard_dofs.dofs_to_be_discarded[I])
            m.append(current_m)
        for J in range(N):
            if not isinstance(block_matrix[0, J], BlockOuterMatrix):
                block = as_backend_type(block_matrix[0, J]).mat()
                current_n = block.getSize()[1]
            else:
                block_vec_1 = as_backend_type(block_matrix[0, J].vecs[1]).vec()
                current_n = block_vec_1.getSize()
            # Make sure that all row dimensions are consistent
            for I in range(M):
                if not isinstance(block_matrix[I, J], BlockOuterMatrix):
                    block = as_backend_type(block_matrix[I, J]).mat()
                    assert block.getSize()[1] == current_n
                else:
                    block_outer_matrix = block_matrix[I, J]
                    while block_outer_matrix is not None:
                        block_vec_1 = as_backend_type(block_outer_matrix.vecs[1]).vec()
                        assert block_vec_1.getSize() == current_n
                        if block_outer_matrix.addend_matrix is not None:
                            assert as_backend_type(block_outer_matrix.addend_matrix).mat().getSize()[1] == current_n
                        block_outer_matrix = block_outer_matrix.addend_block_outer_matrix
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
            # Auxiliary variables for parallel communication
            comm = self.mat().comm.tompi4py()
            comm_size = comm.size
            comm_rank = comm.rank
            # Get local size as the sum of the local sizes of the block matrices
            ownership_range_sum = 0
            for I in range(M):
                if not isinstance(block_matrix[I, 0], BlockOuterMatrix):
                    block = as_backend_type(block_matrix[I, 0]).mat() # ownership range is the same for all J = 0, ..., N - 1
                else:
                    block = as_backend_type(block_matrix[I, 0].vecs[0]).vec() # ownership range is the same for all J = 0, ..., N - 1
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
                # Get the number of diagonal and off-diagonal non zero elements
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
                        if not isinstance(block_matrix[I, J], BlockOuterMatrix):
                            self._update_non_zeros_from_form(block_matrix[I, J], I, J, m, n, row_reposition_dofs, col_reposition_dofs, block_discard_dofs, ownership_range_start, ownership_range_end, d_nnz, o_nnz)
                        else:
                            block_outer_matrix = block_matrix[I, J]
                            while block_outer_matrix is not None:
                                self._update_non_zeros_from_outer(block_outer_matrix, I, J, m, n, row_reposition_dofs, col_reposition_dofs, block_discard_dofs, ownership_range_start, ownership_range_end, d_nnz, o_nnz)
                                if block_outer_matrix.addend_matrix is not None:
                                    self._update_non_zeros_from_form(block_outer_matrix.addend_matrix, I, J, m, n, row_reposition_dofs, col_reposition_dofs, block_discard_dofs, ownership_range_start, ownership_range_end, d_nnz, o_nnz)
                                block_outer_matrix = block_outer_matrix.addend_block_outer_matrix
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
        # Store dofs to be constrained while adding
        assert block_constrain_dofs is None or isinstance(block_constrain_dofs, tuple)
        if block_constrain_dofs is None or len(block_constrain_dofs[0]) == 0:
            self.block_constrain_dofs = None
            self.block_constrain_dofs__value = None
        else:
            self.block_constrain_dofs = block_constrain_dofs[0]
            self.block_constrain_dofs__value = block_constrain_dofs[1]
        
    def block_add(self, block_matrix):
        M, N, m, n = self.M, self.N, self.m, self.n
        assert M, N == block_matrix.blocks.shape
        block_discard_dofs = self.block_discard_dofs
        block_constrain_dofs = self.block_constrain_dofs
        block_constrain_dofs__value = self.block_constrain_dofs__value
        
        for I in range(M):
            if block_constrain_dofs is not None and len(block_constrain_dofs[I]) > 0:
                row_constrain_dofs = block_constrain_dofs[I]
            else:
                row_constrain_dofs = None
            if block_discard_dofs is not None and block_discard_dofs.need_to_discard_dofs[I]:
                row_reposition_dofs = block_discard_dofs.subspace_dofs_extended[I]
            else:
                row_reposition_dofs = None
            for J in range(N):
                if block_constrain_dofs is not None and len(block_constrain_dofs[J]) > 0:
                    col_constrain_dofs = block_constrain_dofs[J]
                else:
                    col_constrain_dofs = None
                if block_discard_dofs is not None and block_discard_dofs.need_to_discard_dofs[J]:
                    col_reposition_dofs = block_discard_dofs.subspace_dofs_extended[J]
                else:
                    col_reposition_dofs = None
                if not isinstance(block_matrix[I, J], BlockOuterMatrix):
                    self._block_add_from_form(block_matrix[I, J], I, J, m, n, row_reposition_dofs, col_reposition_dofs, row_constrain_dofs, col_constrain_dofs, block_constrain_dofs__value, block_discard_dofs)
                else:
                    block_outer_matrix = block_matrix[I, J]
                    while block_outer_matrix is not None:
                        self._block_add_from_outer(block_outer_matrix, I, J, m, n, row_reposition_dofs, col_reposition_dofs, row_constrain_dofs, col_constrain_dofs, block_constrain_dofs__value, block_discard_dofs)
                        if block_outer_matrix.addend_matrix is not None:
                            self._block_add_from_form(block_outer_matrix.addend_matrix, I, J, m, n, row_reposition_dofs, col_reposition_dofs, row_constrain_dofs, col_constrain_dofs, block_constrain_dofs__value, block_discard_dofs)
                        block_outer_matrix = block_outer_matrix.addend_block_outer_matrix
        self.mat().assemble()
            
    def create_monolithic_vectors(self, block_x, block_b):
        petsc_x, petsc_b = self.mat().createVecs()
        return (
            MonolithicVector(block_x, petsc_x, block_discard_dofs=self.block_discard_dofs),
            MonolithicVector(block_b, petsc_b, block_discard_dofs=self.block_discard_dofs)
        )
        
    def create_monolithic_vector_left(self, block_b):
        petsc_b = self.mat().createVecLeft()
        return MonolithicVector(block_b, petsc_b, block_discard_dofs=self.block_discard_dofs)
        
    def _update_non_zeros_from_form(self, block, I, J, m, n, row_reposition_dofs, col_reposition_dofs, block_discard_dofs, ownership_range_start, ownership_range_end, d_nnz, o_nnz):
        block = as_backend_type(block).mat()
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
                                    
    def _update_non_zeros_from_outer(self, block, I, J, m, n, row_reposition_dofs, col_reposition_dofs, block_discard_dofs, ownership_range_start, ownership_range_end, d_nnz, o_nnz):
            block_vec_0 = block.scattered_vecs[0]
            block_vec_1 = block.scattered_vecs[1]
            for i in block.scattered_vecs_non_zero_indices[0]:
                if row_reposition_dofs is not None:
                    if i not in row_reposition_dofs:
                        continue
                    row = row_reposition_dofs[i]
                else:
                    row = i
                row += sum(m[:I])
                A = np.argmax(ownership_range_end > row)
                a = row - ownership_range_start[A]
                for j in block.scattered_vecs_non_zero_indices[1]:
                    if col_reposition_dofs is not None:
                        if j not in col_reposition_dofs:
                            continue
                        col = col_reposition_dofs[j]
                    else:
                        col = j
                    col += sum(n[:J])
                    if col >= ownership_range_start[A] and col < ownership_range_end[A]:
                        d_nnz[A][a] += 1
                    else:
                        o_nnz[A][a] += 1
        
    def _block_add_from_form(self, block, I, J, m, n, row_reposition_dofs, col_reposition_dofs, row_constrain_dofs, col_constrain_dofs, block_constrain_dofs__value, block_discard_dofs):
        block = as_backend_type(block).mat()
        row_start, row_end = block.getOwnershipRange()
        for i in range(row_start, row_end):
            if row_reposition_dofs is not None:
                if i not in row_reposition_dofs:
                    continue
                row = row_reposition_dofs[i]
            else:
                row = i
            row += sum(m[:I])
            if row_constrain_dofs is not None and i in row_constrain_dofs:
                if I == J:
                    cols = np.array([i], dtype='i')
                    vals = np.array([block_constrain_dofs__value])
                    cols_to_preserve = set([i])
                else:
                    continue # zero the row
            else:
                cols, vals = block.getRow(i)
                cols_to_preserve = set()
            if col_reposition_dofs is not None:
                cols_to_vals = dict(zip(cols, vals))
                cols_after_discard = set(cols).difference(block_discard_dofs.dofs_to_be_discarded[J])
                if len(cols_after_discard) == 0:
                    continue
                if col_constrain_dofs is not None:
                    cols_after_discard = set(cols_after_discard).difference(col_constrain_dofs.difference(cols_to_preserve))
                    if len(cols_after_discard) == 0:
                        continue
                cols = [col_reposition_dofs[c] for c in cols_after_discard]
                vals = [cols_to_vals[c] for c in cols_after_discard]
                assert len(cols) == len(vals)
                cols = np.array(cols)
                vals = np.array(vals)
            cols[:] += sum(n[:J])
            self.mat().setValues(row, cols, vals, addv=PETSc.InsertMode.ADD)
            
    def _block_add_from_outer(self, block, I, J, m, n, row_reposition_dofs, col_reposition_dofs, row_constrain_dofs, col_constrain_dofs, block_constrain_dofs__value, block_discard_dofs):
        block_vec_0 = block.scattered_vecs[0]
        block_vec_1 = block.scattered_vecs[1]
        row_start_0, row_end_0 = block_vec_0.getOwnershipRange()
        row_start_1, row_end_1 = block_vec_1.getOwnershipRange()
        for i in block.scattered_vecs_non_zero_indices[0]:
            if row_reposition_dofs is not None:
                if i not in row_reposition_dofs:
                    continue
                row = row_reposition_dofs[i]
            else:
                row = i
            row += sum(m[:I])
            if row_constrain_dofs is not None and i in row_constrain_dofs:
                if I == J:
                    cols = np.array([i], dtype='i')
                    vals = np.array([block_constrain_dofs__value])
                    cols_to_preserve = set([i])
                else:
                    continue # zero the row
            else:
                cols = list()
                vals = list()
                for j in block.scattered_vecs_non_zero_indices[1]:
                    cols.append(j)
                    vals.append(block.scale*block_vec_0.array[i - row_start_0]*block_vec_1.array[j - row_start_1])
                cols = np.array(cols, dtype='i')
                vals = np.array(vals)
                cols_to_preserve = set()
            if col_reposition_dofs is not None:
                cols_to_vals = dict(zip(cols, vals))
                cols_after_discard = set(cols).difference(block_discard_dofs.dofs_to_be_discarded[J])
                if len(cols_after_discard) == 0:
                    continue
                if col_constrain_dofs is not None:
                    cols_after_discard = set(cols_after_discard).difference(col_constrain_dofs.difference(cols_to_preserve))
                    if len(cols_after_discard) == 0:
                        continue
                cols = [col_reposition_dofs[c] for c in cols_after_discard]
                vals = [cols_to_vals[c] for c in cols_after_discard]
                assert len(cols) == len(vals)
                cols = np.array(cols)
                vals = np.array(vals)
            cols[:] += sum(n[:J])
            self.mat().setValues(row, cols, vals, addv=PETSc.InsertMode.ADD)
        

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

from dolfin import DirichletBC, GenericMatrix, GenericVector
from block_ext.block_matrix import BlockMatrix
from block_ext.block_vector import BlockVector
from block_ext.block_outer import BlockOuterMatrix, BlockOuterVector

# This class is a simplified version of CBC block block_bc and block_rhs_bc, that does not account symmetric variants and float blocks
class BlockDirichletBC(object):
    def __init__(self, bcs):
        # Store bcs
        self.bcs = list()
        for bc in bcs:
            if bc is None:
                self.bcs.append([])
            elif not hasattr(bc, "__iter__"):
                self.bcs.append([bc])
            else:
                self.bcs.append(bc)
    
    def apply(self, arg1, arg2=None, arg3=None):
        if isinstance(arg1, BlockMatrix):
            assert arg2 is None and arg3 is None
            self._apply_A(arg1)
        elif isinstance(arg1, BlockVector):
            assert arg3 is None
            self._apply_b(arg1, arg2)
        else:
            raise RuntimeError("BlockDirichletBC::apply method not yet implemented")
        
    def _apply_A(self, A):
        for I, bcs_I in enumerate(self.bcs):
            for bc_I in bcs_I:
                for J, _ in enumerate(self.bcs):
                    assert isinstance(A[I, J], (GenericMatrix, BlockOuterMatrix))
                    if isinstance(A[I, J], GenericMatrix):
                        if I == J:
                            bc_I.apply(A[I, I])
                        else:
                            bc_I.zero(A[I, J])
                    elif isinstance(A[I, J], BlockOuterMatrix):
                        # We envision usage of block outer on boundary integrals,
                        # because otherwise the stiffness matrix would become full.
                        # In all envisioned user cases up to now the block outer 
                        # matrix is summed to a standard GenericMatrix, because if 
                        # that were not the case the system would be singular:
                        # it is thus possible to diagonalize the GenericMatrix
                        # and zero the outer vectors
                        block_outer_matrix = A[I, J]
                        number_addend_matrix = 0
                        while block_outer_matrix is not None:
                            homog_bc_I = DirichletBC(bc_I)
                            homog_bc_I.homogenize()
                            homog_bc_I.apply(block_outer_matrix.vecs[0])
                            homog_bc_I.apply(block_outer_matrix.vecs[1])
                            if block_outer_matrix.addend_matrix is not None:
                                number_addend_matrix += 1
                                if I == J:
                                    bc_I.apply(block_outer_matrix.addend_matrix)
                                else:
                                    bc_I.zero(block_outer_matrix.addend_matrix)
                            block_outer_matrix = block_outer_matrix.addend_block_outer_matrix
                        assert number_addend_matrix > 0
                        assert number_addend_matrix < 2 # the case of multiple addend matrices has not been considered yet
                    else:
                        raise AssertionError("BlockDirichletBC::_apply_A invalid matrix")
        
    def _apply_b(self, b, x=None):
        def bc_apply(bc_I, b_I, I):
            if x is None:
                bc_I.apply(b_I)
            else:
                bc_I.apply(b_I, x[I])
        for I, bcs_I in enumerate(self.bcs):
            for bc_I in bcs_I:
                assert isinstance(b[I], (GenericVector, BlockOuterVector))
                if isinstance(b[I], GenericVector):
                    bc_apply(bc_I, b[I], I)
                elif isinstance(b[I], BlockOuterVector):
                    bc_apply(bc_I, b[I].vec, I)
                else:
                    raise AssertionError("BlockDirichletBC::_apply_b invalid vector")
                        
    def __getitem__(self, key):
        return self.bcs[key]
        
    def __iter__(self):
        return self.bcs.__iter__()
        
    def __len__(self):
        return len(self.bcs)
                

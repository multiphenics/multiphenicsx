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

from block_ext.block_matrix import BlockMatrix
from block_ext.block_vector import BlockVector
from block.block_bc import block_bc, block_rhs_bc

class BlockDirichletBC(object):
    def __init__(self, bcs):
        self.block_bc = block_bc(bcs, symmetric=False)
        self.block_rhs_bc = block_rhs_bc(self.block_bc, A=None)
    
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
        self.block_bc.apply(A)
        
    def _apply_b(self, b, x=None):
        if x is None:
            self.block_rhs_bc.apply(b)
        else:
            # Simplified version from cbc.block method block_rhs_bc::apply
            # Not accounting for symmetric variant (which is hardcoded to False) and
            # neither for scalar values in blocks
            for i, bcs in enumerate(self.block_rhs_bc):
                for bc in bcs:
                    bc.apply(b[i], x[i])
                        
    def __getitem__(self, key):
        return self.block_bc[key]
        
    def __iter__(self):
        return self.block_bc.__iter__()
        
    def __len__(self):
        return len(self.block_bc)
                

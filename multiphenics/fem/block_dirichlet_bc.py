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

import collections
from multiphenics.cpp import cpp

BlockDirichletBC_Base = cpp.fem.BlockDirichletBC

class BlockDirichletBC(BlockDirichletBC_Base):
    def __init__(self, bcs, block_function_space=None):
        # Flatten out the input, in order to handle nesting
        bcs = self._flatten_bcs(bcs)
        # Detect the common block function space
        if block_function_space is None:
            for bc in bcs:
                if block_function_space is None:
                    block_function_space = bc.function_space.block_function_space()
                else:
                    assert block_function_space is bc.function_space.block_function_space()
        assert block_function_space is not None, "It is not possible to build an empty BlockDirichletBC without providing a block_function_space"
        self._block_function_space = block_function_space
        # Split again bcs according to block_index
        self.bcs = list()
        for _ in range(self._block_function_space.num_sub_spaces()):
            self.bcs.append(list())
        for bc in bcs:
            bc_block_index = bc.function_space.block_index()
            bc_block_function_space = bc.function_space.block_function_space()
            if hasattr(bc_block_function_space, "is_block_subspace"):
                assert bc_block_index in bc_block_function_space.sub_components_to_components, "Block function space and BC block index are not consistent on the sub space."
                bc_block_index = bc_block_function_space.sub_components_to_components[bc_block_index]
                bc_block_function_space = bc_block_function_space.parent_block_function_space
            if hasattr(block_function_space, "is_block_subspace"):
                assert bc_block_index in block_function_space.components_to_sub_components, "Block function space and BC block index are not consistent on the sub space."
                bc_block_index = block_function_space.components_to_sub_components[bc_block_index]
                assert bc_block_function_space == block_function_space.parent_block_function_space
            else:
                assert bc_block_function_space == block_function_space
            self.bcs[bc_block_index].append(bc)
        # Call Parent
        BlockDirichletBC_Base.__init__(self, self.bcs, self._block_function_space._cpp_object)
        
    def __getitem__(self, key):
        return self.bcs[key]
        
    def __iter__(self):
        return self.bcs.__iter__()
        
    def __len__(self):
        return len(self.bcs)
        
    @property
    def block_function_space(self):
        return self._block_function_space
    
    @staticmethod
    def _flatten_bcs(bcs):
        # https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
        def flatten(l):
            for el in l:
                if isinstance(el, collections.Iterable):
                    for sub in flatten(el):
                        yield sub
                else:
                    yield el
         
        # Flatten and remove any remaining None
        return [bc for bc in flatten(bcs) if bc is not None]

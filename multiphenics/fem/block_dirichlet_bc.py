# Copyright (C) 2016-2019 by the multiphenics authors
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
from dolfin.cpp.la import GenericMatrix, GenericVector
from multiphenics.python import cpp

BlockDirichletBC_Base = cpp.fem.BlockDirichletBC

class BlockDirichletBC(BlockDirichletBC_Base):
    def __init__(self, bcs, block_function_space=None):
        # Flatten out the input, in order to handle nesting
        bcs = self._flatten_bcs(bcs)
        # Detect the common block function space
        if block_function_space is None:
            for bc in bcs:
                if block_function_space is None:
                    block_function_space = bc.function_space().block_function_space()
                else:
                    assert block_function_space is bc.function_space().block_function_space()
        assert block_function_space is not None, "It is not possible to build an empty BlockDirichletBC without providing a block_function_space"
        self._block_function_space = block_function_space
        # Split again bcs according to block_index
        self.bcs = list()
        for _ in range(self._block_function_space.num_sub_spaces()):
            self.bcs.append(list())
        for bc in bcs:
            block_index = bc.function_space().block_index()
            if hasattr(block_function_space, "is_block_subspace"):
                assert block_index in block_function_space.components_to_sub_components, "Block function space and BC block index are not consistent on the sub space."
                block_index = block_function_space.components_to_sub_components[block_index]
            self.bcs[block_index].append(bc)
        # We disable the check on dof map range which is carried out by DirichletBC::check_arguments,
        # because BCs are defined on unrestricted function spaces, while sub tensors to which BCs
        # will be applied may be restricted
        for bc in self.bcs:
            for bc_I in bc:
                bc_I.parameters["check_dofmap_range"] = False
        # Call Parent
        BlockDirichletBC_Base.__init__(self, self.bcs, self._block_function_space.cpp_object())
        
    def __getitem__(self, key):
        return self.bcs[key]
        
    def __iter__(self):
        return self.bcs.__iter__()
        
    def __len__(self):
        return len(self.bcs)
        
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
        
    def apply(self, *args):
        assert len(args) in (1, 2, 3)
        if len(args) == 1:
            arg0 = args[0]
            assert isinstance(arg0, (GenericMatrix, GenericVector))
            if isinstance(arg0, GenericMatrix):
                assert hasattr(arg0, "_bcs_zero_off_block_diagonal")
                BlockDirichletBC_Base.apply(self, arg0, arg0._bcs_zero_off_block_diagonal)
            elif isinstance(arg0, GenericVector):
                BlockDirichletBC_Base.apply(self, arg0)
            else:
                raise ValueError("Invalid arguments")
        elif len(args) == 2:
            arg0 = args[0]
            arg1 = args[1]
            assert isinstance(arg0, (GenericMatrix, GenericVector))
            assert isinstance(arg1, GenericVector)
            if isinstance(arg0, GenericMatrix):
                BlockDirichletBC_Base.apply(self, arg0, arg1, arg0._bcs_zero_off_block_diagonal)
            elif isinstance(arg0, GenericVector):
                BlockDirichletBC_Base.apply(self, arg0, arg1)
            else:
                raise ValueError("Invalid arguments")
        elif len(args) == 3:
            arg0 = args[0]
            arg1 = args[1]
            arg2 = args[1]
            assert isinstance(arg0, GenericMatrix)
            assert isinstance(arg1, GenericVector)
            assert isinstance(arg2, GenericVector)
            BlockDirichletBC_Base.apply(self, arg0, arg1, arg2, arg0._bcs_zero_off_block_diagonal)
        else:
            raise ValueError("Invalid arguments")
        return
        
    def zero(self, *args):
        assert len(args) == 1
        arg0 = args[0]
        assert isinstance(arg0, GenericMatrix)
        BlockDirichletBC_Base.zero(self, arg0, arg0._bcs_zero_off_block_diagonal)

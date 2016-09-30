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

from ufl import Form
from block_ext import block_assemble, BlockDirichletBC
from RBniCS.backends.abstract import AffineExpansionStorage as AbstractAffineExpansionStorage
from RBniCS.utils.decorators import BackendFor, Extends, list_of, override, tuple_of

@Extends(AbstractAffineExpansionStorage)
@BackendFor("block_ext", inputs=((tuple_of(list_of(BlockDirichletBC)), tuple_of(list_of(Form)), tuple_of(list_of(list_of(Form)))), ))
class AffineExpansionStorage(AbstractAffineExpansionStorage):
    @override
    def __init__(self, args):
        self._content = None
        self._type = None
        # Type checking
        is_Form = self._is_Form(args[0])
        is_DirichletBC = self._is_DirichletBC(args[0])
        assert is_Form or is_DirichletBC
        for i in range(1, len(args)):
            if is_Form:
                assert self._is_Form(args[i])
            elif is_DirichletBC:
                assert self._is_DirichletBC(args[i])
            else:
                return TypeError("Invalid input arguments to AffineExpansionStorage")
        # Actual init
        if is_Form:
            self._content = [block_assemble(arg) for arg in args]
            self._type = "BlockForm"
        elif is_DirichletBC:
            self._content = args
            self._type = "BlockDirichletBC"
        else:
            return TypeError("Invalid input arguments to AffineExpansionStorage")
        
    @staticmethod
    def _is_Form(arg):
        if isinstance(arg, list):
            if isinstance(arg[0], Form): # block vector
                return True
            elif isinstance(arg[0], list):
                if isinstance(arg[0][0], Form): # block matrix
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
        
    @staticmethod
    def _is_DirichletBC(arg):
        return isinstance(arg, BlockDirichletBC)
        
    def type(self):
        return self._type
        
    @override
    def __getitem__(self, key):
        return self._content[key]
        
    @override
    def __iter__(self):
        return self._content.__iter__()
        
    @override
    def __len__(self):
        assert self._content is not None
        return len(self._content)
        

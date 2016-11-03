# Copyright (C) 2015-2016 by the RBniCS authors
# Copyright (C) 2016 by the block_ext authors
#
# This file is part of the RBniCS interface to block_ext.
#
# RBniCS and block_ext are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and block_ext are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and block_ext. If not, see <http://www.gnu.org/licenses/>.
#

from ufl import Form
from block_ext import block_adjoint, block_assemble, BlockDirichletBC
from RBniCS.backends.abstract import AffineExpansionStorage as AbstractAffineExpansionStorage
from RBniCS.utils.decorators import BackendFor, Extends, list_of, override, tuple_of
from block_ext.RBniCS.matrix import Matrix
from block_ext.RBniCS.vector import Vector

@Extends(AbstractAffineExpansionStorage)
@BackendFor("block_ext", inputs=((tuple_of(BlockDirichletBC), tuple_of(list_of(Form)), tuple_of(list_of(list_of(Form))), tuple_of(Matrix.Type()), tuple_of(Vector.Type())), (bool, None)))
class AffineExpansionStorage(AbstractAffineExpansionStorage):
    @override
    def __init__(self, args, symmetrize=None):
        self._content = None
        self._type = None
        # Type checking
        is_Form = self._is_Form(args[0])
        is_DirichletBC = self._is_DirichletBC(args[0])
        is_Tensor = self._is_Tensor(args[0])
        assert is_Form or is_DirichletBC or is_Tensor
        for i in range(1, len(args)):
            if is_Form:
                assert self._is_Form(args[i])
            elif is_DirichletBC:
                assert self._is_DirichletBC(args[i])
            elif is_Tensor:
                assert self._is_Tensor(args[i])
            else:
                return TypeError("Invalid input arguments to AffineExpansionStorage")
        # Actual init
        if is_Form:
            if symmetrize is None or symmetrize is False:
                self._content = [block_assemble(arg) for arg in args]
            elif symmetrize is True:
                self._content = [block_assemble(0.5*(arg + block_adjoint(arg))) for arg in args]
            else:
                return TypeError("Invalid input arguments to AffineExpansionStorage")
            self._type = "BlockForm"
        elif is_DirichletBC:
            assert symmetrize is None
            self._content = args
            self._type = "BlockDirichletBC"
        elif is_Tensor:
            assert symmetrize is None
            self._content = args
            self._type = "Form"
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
        
    @staticmethod
    def _is_Tensor(arg):
        return isinstance(arg, (Matrix.Type(), Vector.Type()))
        
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
        

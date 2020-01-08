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

import dolfin
from multiphenics.python import cpp

def DecorateGetEigenPair(BlockSLEPcEigenSolver):
    from multiphenics.function import BlockFunction # avoid recursive imports
    
    class DecoratedBlockSLEPcEigenSolver(BlockSLEPcEigenSolver):
        def get_eigenpair(self, r_fun, c_fun, i):
            assert isinstance(r_fun, BlockFunction)
            assert isinstance(c_fun, BlockFunction)
            (lr, lc, _, _) = BlockSLEPcEigenSolver.get_eigenpair(self, r_fun._cpp_object, c_fun._cpp_object, i)
            return (lr, lc, r_fun, c_fun)
            
    return DecoratedBlockSLEPcEigenSolver
    
def BlockSLEPcEigenSolver(A, B=None, bcs=None):
    from multiphenics.fem import BlockDirichletBC # avoid recursive imports
    
    if bcs is None:
        EigenSolver = DecorateGetEigenPair(dolfin.SLEPcEigenSolver) # applicable also to block matrices, because block la inherits from standard la
        return EigenSolver(A, B)
    else:
        assert isinstance(bcs, BlockDirichletBC)
        EigenSolver = DecorateGetEigenPair(cpp.la.CondensedBlockSLEPcEigenSolver)
        return EigenSolver(A, B, bcs)

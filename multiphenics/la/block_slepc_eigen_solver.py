# Copyright (C) 2016-2017 by the multiphenics authors
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
from dolfin import has_pybind11
from multiphenics.python import cpp

def DecorateGetEigenPair(BlockSLEPcEigenSolver):
    from multiphenics.function import BlockFunction # avoid recursive imports
    
    class DecoratedBlockSLEPcEigenSolver(BlockSLEPcEigenSolver):
        def get_eigenpair(self, i, r_vec=None, c_vec=None):
            if isinstance(r_vec, BlockFunction):
                r_vec_in = None # cannot use r_vec due to different ghosting
            else:
                r_vec_in = r_vec
            if isinstance(c_vec, BlockFunction):
                c_vec_in = None # cannot use r_vec due to different ghosting
            else:
                c_vec_in = c_vec
            (lr, lc, r_vec_out, c_vec_out) = BlockSLEPcEigenSolver.get_eigenpair(self, i, r_vec_in, c_vec_in)
            if isinstance(r_vec, BlockFunction):
                r_vec.block_vector().set_local(r_vec_out.get_local())
                r_vec.block_vector().apply("insert")
                r_vec.apply("to subfunctions")
                r_vec_out = r_vec
            if isinstance(c_vec, BlockFunction):
                c_vec.block_vector().set_local(c_vec_out.get_local())
                c_vec.block_vector().apply("insert")
                c_vec.apply("to subfunctions")
                c_vec_out = c_vec
            return (lr, lc, r_vec_out, c_vec_out)
            
    return DecoratedBlockSLEPcEigenSolver

def BlockSLEPcEigenSolver(A, B=None, bcs=None):
    from multiphenics.fem import BlockDirichletBC # avoid recursive imports
    
    if bcs is None:
        EigenSolver = DecorateGetEigenPair(dolfin.SLEPcEigenSolver) # applicable also to block matrices, because block la inherits from standard la
        return EigenSolver(A, B)
    else:
        assert isinstance(bcs, BlockDirichletBC)
        if has_pybind11():
            EigenSolver = DecorateGetEigenPair(cpp.la.CondensedBlockSLEPcEigenSolver)
        else:
            EigenSolver = DecorateGetEigenPair(cpp.CondensedBlockSLEPcEigenSolver)
        return EigenSolver(A, B, bcs)

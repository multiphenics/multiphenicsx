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

from block_ext import BlockFunction, BlockFunctionSpace, BlockSLEPcEigenSolver
from block_ext.RBniCS.backends.block_ext.matrix import Matrix
from RBniCS.backends.abstract import EigenSolver as AbstractEigenSolver
from RBniCS.utils.decorators import BackendFor, Extends, override

@Extends(AbstractEigenSolver)
@BackendFor("block_ext", inputs=(Matrix.Type(), (Matrix.Type(), None), (BlockFunctionSpace, None)))
class EigenSolver(AbstractEigenSolver):
    @override
    def __init__(self, A, B=None, V=None): # TODO deve mettere il block discard dofs?
        self.eigen_solver = BlockSLEPcEigenSolver(A, B)
        assert V is not None
        self.V = V
        
    @override
    def set_parameters(self, parameters):
        self.eigen_solver.parameters.update(parameters)
        
    @override
    def solve(self, n_eigs=None):
        assert n_eigs is not None
        self.eigen_solver.solve(n_eigs)
    
    @override
    def get_eigenvalue(self, i):
        return self.eigen_solver.get_eigenvalue(i)
    
    @override
    def get_eigenvector(self, i):
        (_, _, real_vector, imag_vector) = self.eigen_solver.get_eigenpair(i)
        return (BlockFunction(self.V, real_vector), BlockFunction(self.V, imag_vector))
        

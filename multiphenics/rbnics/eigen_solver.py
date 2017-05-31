# Copyright (C) 2015-2017 by the RBniCS authors
# Copyright (C) 2016-2017 by the multiphenics authors
#
# This file is part of the RBniCS interface to multiphenics.
#
# RBniCS and multiphenics are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and multiphenics are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

from multiphenics import BlockFunction, BlockFunctionSpace, BlockSLEPcEigenSolver
from multiphenics.rbnics.affine_expansion_storage import AffineExpansionStorage
from multiphenics.rbnics.matrix import Matrix
from multiphenics.rbnics.product import product
from multiphenics.rbnics.sum import sum
from rbnics.backends.abstract import EigenSolver as AbstractEigenSolver
from rbnics.utils.decorators import BackendFor, Extends, override

@Extends(AbstractEigenSolver)
@BackendFor("multiphenics", inputs=(BlockFunctionSpace, Matrix.Type(), (Matrix.Type(), None), (AffineExpansionStorage, None)))
class EigenSolver(AbstractEigenSolver):
    @override
    def __init__(self, V, A, B=None, bcs=None):
        self.V = V
        if bcs is not None:
            bcs_sum = sum(product(len(bcs)*(1., ), bcs))
            self.eigen_solver = BlockSLEPcEigenSolver(A, B, bcs_sum)
        else:
            self.eigen_solver = BlockSLEPcEigenSolver(A, B, None)
        
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
        

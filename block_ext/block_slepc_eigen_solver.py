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

from dolfin import SLEPcEigenSolver
from block_ext.monolithic_matrix import MonolithicMatrix
from block_ext.monolithic_vector import MonolithicVector

class BlockSLEPcEigenSolver(SLEPcEigenSolver):
    def __init__(self, block_A=None, block_B=None, block_discard_dofs=None):
        self.block_A = block_A
        (A, B) = self._convert_block_matrices_to_monolithic_matrices(block_A, block_B, block_discard_dofs)
        self._init_monolithic_vectors_for_eigenvectors(A)
        SLEPcEigenSolver.__init__(self, A, B)
        
    def set_operators(self, block_A=None, block_B=None, block_discard_dofs=None):
        assert block_A is not None
        self.block_A = block_A
        (A, B) = self._convert_block_matrices_to_monolithic_matrices(block_A, block_B, block_discard_dofs)
        self._init_monolithic_vectors_for_eigenvectors(A)
        SLEPcEigenSolver.set_operators(self, A, B)
    
    def get_eigenpair(self, i=0, block_r_vec=None, block_c_vec=None):
        assert (
            (block_r_vec is None and block_c_vec is None)
                or
            (block_r_vec is not None and block_c_vec is not None)
        )
        if block_r_vec is None and block_c_vec is None:
            (block_r_vec, block_c_vec) = self._create_block_vectors_for_eigenvectors()
        r, c, r_vec, c_vec = SLEPcEigenSolver.get_eigenpair(self, i, self.monolithic_r_vec, self.monolithic_c_vec)
        self._convert_monolithic_vectors_to_block_vectors(r_vec, c_vec, block_r_vec, block_c_vec)
        return r, c, block_r_vec, block_c_vec
        
    def _convert_block_matrices_to_monolithic_matrices(self, block_A=None, block_B=None, block_discard_dofs=None):
        if block_A is not None:
            A = MonolithicMatrix(block_A, block_discard_dofs=block_discard_dofs)
            A.zero(); A.block_add(block_A)
        else:
            A = None
        if block_B is not None:
            B = MonolithicMatrix(block_B, block_discard_dofs=block_discard_dofs)
            B.zero(); B.block_add(block_B)
        else:
            B = None
        return (A, B)
        
    def _init_monolithic_vectors_for_eigenvectors(self, A):
        (block_r_vec, block_c_vec) = self._create_block_vectors_for_eigenvectors()
        (petsc_r_vec, petsc_c_vec) = (A.mat().createVecRight(), A.mat().createVecRight())
        self.monolithic_r_vec = MonolithicVector(block_r_vec, petsc_r_vec)
        self.monolithic_c_vec = MonolithicVector(block_c_vec, petsc_c_vec)
        
    def _create_block_vectors_for_eigenvectors(self):
        return (self.block_A.create_vec(), self.block_A.create_vec())
        
    def _convert_monolithic_vectors_to_block_vectors(self, r_vec, c_vec, block_r_vec, block_c_vec):
        r_vec.copy_values_to(block_r_vec)
        c_vec.copy_values_to(block_c_vec)
        
        

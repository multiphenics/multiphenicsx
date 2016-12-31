# Copyright (C) 2016-2017 by the block_ext authors
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

from numpy import isclose, unique
from dolfin import SLEPcEigenSolver
from block_ext.monolithic_matrix import MonolithicMatrix
from block_ext.monolithic_vector import MonolithicVector

class BlockSLEPcEigenSolver(object):
    def __init__(self, block_A, block_B=None, block_bcs=None):
        self.block_A = block_A
        self.block_B = block_B
        self.block_bcs = block_bcs
        self._constrained_dofs = self._get_constrained_dofs()
        self._init_eigen_solver(0.1) # we will check in solve if it is an appropriate spurious eigenvalue
        
    def set_operators(self, block_A, block_B=None, block_bcs=None):
        self.block_A = block_A
        self.block_B = block_B
        self.block_bcs = block_bcs
        self._constrained_dofs = self._get_constrained_dofs()
        self._init_eigen_solver(1.) # we will check in solve if it is an appropriate spurious eigenvalue
        
    def _init_eigen_solver(self, spurious_eigenvalue):
        if hasattr(self, "parameters"): # backup previous parameters, if any
            parameters_backup = dict(self.parameters)
        else:
            parameters_backup = None
        self._spurious_eigenvalue = spurious_eigenvalue
        (A, B) = self._convert_block_matrices_to_monolithic_matrices()
        self._init_monolithic_vectors_for_eigenvectors(A)
        self._eigen_solver = SLEPcEigenSolver(A, B)
        self.parameters = self._eigen_solver.parameters
        if parameters_backup is not None:
            self.parameters.update(parameters_backup)

    def _get_constrained_dofs(self):
        constrained_dofs = list() # of list of dofs
        if self.block_bcs is not None:
            for I, bcs in enumerate(self.block_bcs):
                constrained_dofs_I = list()
                for bc in bcs:
                    for _ in range(len(self.block_bcs)):
                        constrained_dofs_I.extend([bc.function_space().dofmap().local_to_global_index(local_dof_index) for local_dof_index in bc.get_boundary_values().keys()])
                constrained_dofs_I = unique(constrained_dofs_I)
                constrained_dofs.append(constrained_dofs_I)
        return constrained_dofs
        
    def _convert_block_matrices_to_monolithic_matrices(self):
        A = MonolithicMatrix(self.block_A, block_discard_dofs=self.block_A._block_discard_dofs, block_constrain_dofs=(self._constrained_dofs, self._spurious_eigenvalue))
        A.zero(); A.block_add(self.block_A)
        if self.block_B is not None:
            assert self.block_A._block_discard_dofs == self.block_B._block_discard_dofs
            B = MonolithicMatrix(self.block_B, block_discard_dofs=self.block_B._block_discard_dofs, block_constrain_dofs=(self._constrained_dofs, 1.))
            B.zero(); B.block_add(self.block_B)
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
        
    def solve(self, n_eigs=None):
        def do_solve():
            assert n_eigs is not None
            self._eigen_solver.solve(n_eigs)
        
        # Check if the spurious eigenvalue related to BCs is part of the computed eigenvalues.
        # If it is, reinit the eigen problem with a different spurious eigenvalue
        def have_spurious_eigenvalue():
            if self.block_bcs is not None:
                assert self.parameters["spectrum"] in ("largest real", "smallest real")
                if self.parameters["spectrum"] == "largest real":
                    smallest_computed_eigenvalue, smallest_computed_eigenvalue_imag = self.get_eigenvalue(n_eigs - 1)
                    assert isclose(smallest_computed_eigenvalue_imag, 0), "The required eigenvalue is not real"
                    if self._spurious_eigenvalue - smallest_computed_eigenvalue >= 0. or isclose(self._spurious_eigenvalue - smallest_computed_eigenvalue, 0.):
                        self._init_eigen_solver(0.1*smallest_computed_eigenvalue)
                        return True
                    else:
                        return False
                elif self.parameters["spectrum"] == "smallest real":
                    largest_computed_eigenvalue, largest_computed_eigenvalue_imag = self.get_eigenvalue(n_eigs - 1)
                    assert isclose(largest_computed_eigenvalue_imag, 0), "The required eigenvalue is not real"
                    if self._spurious_eigenvalue - largest_computed_eigenvalue <= 0. or isclose(self._spurious_eigenvalue - largest_computed_eigenvalue, 0.):
                        self._init_eigen_solver(10.*largest_computed_eigenvalue)
                        return True
                    else:
                        return False
            else:
                return False
                
        do_solve()
        if have_spurious_eigenvalue():
            do_solve() # the spurious eigenvalue has been changed by the previous call to have_spurious_eigenvalue
            assert not have_spurious_eigenvalue()
    
    def get_eigenvalue(self, i=0):
        return self._eigen_solver.get_eigenvalue(i)
    
    def get_eigenpair(self, i=0, block_r_vec=None, block_c_vec=None):
        assert (
            (block_r_vec is None and block_c_vec is None)
                or
            (block_r_vec is not None and block_c_vec is not None)
        )
        if block_r_vec is None and block_c_vec is None:
            (block_r_vec, block_c_vec) = self._create_block_vectors_for_eigenvectors()
        r, c, r_vec, c_vec = self._eigen_solver.get_eigenpair(i, self.monolithic_r_vec, self.monolithic_c_vec)
        self._convert_monolithic_vectors_to_block_vectors(r_vec, c_vec, block_r_vec, block_c_vec)
        return r, c, block_r_vec, block_c_vec
        
    def _convert_monolithic_vectors_to_block_vectors(self, r_vec, c_vec, block_r_vec, block_c_vec):
        r_vec.copy_values_to(block_r_vec)
        c_vec.copy_values_to(block_c_vec)
        
        

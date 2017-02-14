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

import types
from numpy import ndarray as array
from dolfin import NonlinearProblem, as_backend_type
from block_ext.block_assemble import block_assemble
from block_ext.block_matrix import BlockMatrix
from block_ext.block_vector import BlockVector
from block_ext.monolithic_matrix import MonolithicMatrix
from block_ext.monolithic_vector import MonolithicVector

class BlockNonlinearProblem(NonlinearProblem):
    def __init__(self, residual_form_or_eval, block_solution, bcs, jacobian_form_or_eval):
        NonlinearProblem.__init__(self)
        # Store the input arguments
        self.residual_form_or_eval = residual_form_or_eval
        self.jacobian_form_or_eval = jacobian_form_or_eval
        self.block_solution = block_solution
        self.bcs = bcs
        # Assemble residual and jacobian in order to
        # have block storage with appropriate sizes
        # and sparsity patterns
        self.block_residual = self._block_residual_vector_assemble(None, block_solution)
        self.block_jacobian = self._block_jacobian_matrix_assemble(None, block_solution)
        # Add monolithic wrappers to PETSc objects, initialized
        # the first time F or J are called
        self.monolithic_residual = None
        self.monolithic_jacobian = None
        # Extract block discard dofs from block solution
        self.block_discard_dofs = block_solution._block_discard_dofs
        # Declare a monolithic_solution vector. The easiest way is to define a temporary monolithic matrix
        monolithic_jacobian = MonolithicMatrix(self.block_jacobian, preallocate=False, block_discard_dofs=(self.block_discard_dofs, self.block_discard_dofs))
        monolithic_solution, monolithic_residual = monolithic_jacobian.create_monolithic_vectors(self.block_solution.block_vector(), self.block_residual)
        self.monolithic_solution = monolithic_solution
        # Add the current block_solution as initial guess
        self.monolithic_solution.zero(); self.monolithic_solution.block_add(self.block_solution.block_vector())
        
    def F(self, fenics_residual, fenics_solution):
        # Initialize monolithic wrappers (only once)
        if self.monolithic_residual is None:
            self.monolithic_residual = MonolithicVector(self.block_residual, as_backend_type(fenics_residual).vec(), block_discard_dofs=self.block_discard_dofs)
        # Shorthands
        block_residual = self.block_residual
        block_solution = self.block_solution
        monolithic_residual = self.monolithic_residual # wrapper of the second input argument
        monolithic_solution = self.monolithic_solution # wrapper of the third input argument
        bcs = self.bcs
        # Convert monolithic_solution into block_solution
        monolithic_solution.copy_values_to(block_solution.block_vector())
        # Assemble
        self._block_residual_vector_assemble(block_residual, block_solution)
        bcs.apply(block_residual, block_solution.block_vector())
        # Copy values from block_residual/solution into monolithic_residual/solution
        monolithic_residual.zero(); monolithic_residual.block_add(block_residual)
        monolithic_solution.zero(); monolithic_solution.block_add(block_solution.block_vector())
        
    def _block_residual_vector_assemble(self, block_residual, block_solution):
        assert isinstance(self.residual_form_or_eval, (list, array, types.FunctionType))
        if isinstance(self.residual_form_or_eval, (list, array)):
            residual_form_or_vector = self.residual_form_or_eval
        elif isinstance(self.residual_form_or_eval, types.FunctionType):
            residual_form_or_vector = self.residual_form_or_eval(block_solution)
        else:
            raise AssertionError("Invalid case in BlockNonlinearProblem._block_residual_vector_assemble.")
        assert isinstance(residual_form_or_vector, (list, array, BlockVector))
        if isinstance(residual_form_or_vector, (list, array)):
            if block_residual is not None:
                block_assemble(residual_form_or_vector, block_tensor=block_residual)
                return block_residual
            else:
                return block_assemble(residual_form_or_vector)
        elif isinstance(residual_form_or_vector, BlockVector):
            N = len(residual_form_or_vector)
            if block_residual is None:
                block_residual = BlockVector(N)
            for I in range(N):
                block_residual.blocks[I] = residual_form_or_vector.blocks[I]
            return block_residual
        else:
            raise AssertionError("Invalid case in BlockNonlinearProblem._block_residual_vector_assemble.")
        
    def J(self, fenics_jacobian, _):
        # Initialize monolithic wrappers (only once)
        if self.monolithic_jacobian is None:
            self.monolithic_jacobian = MonolithicMatrix(self.block_jacobian, as_backend_type(fenics_jacobian).mat(), block_discard_dofs=(self.block_discard_dofs, self.block_discard_dofs))
        # Shorthands
        block_jacobian = self.block_jacobian
        block_solution = self.block_solution
        bcs = self.bcs
        monolithic_jacobian = self.monolithic_jacobian # wrapper of the second input argument
        # Assemble
        self._block_jacobian_matrix_assemble(block_jacobian, block_solution)
        bcs.apply(block_jacobian)
        # Copy values from block_jacobian into monolithic_jacobian
        monolithic_jacobian.zero(); monolithic_jacobian.block_add(block_jacobian)
        
    def _block_jacobian_matrix_assemble(self, block_jacobian, block_solution):
        assert isinstance(self.jacobian_form_or_eval, (list, array, types.FunctionType))
        if isinstance(self.jacobian_form_or_eval, (list, array)):
            jacobian_form_or_matrix = self.jacobian_form_or_eval
        elif isinstance(self.jacobian_form_or_eval, types.FunctionType):
            jacobian_form_or_matrix = self.jacobian_form_or_eval(block_solution)
        else:
            raise AssertionError("Invalid case in BlockNonlinearProblem._block_jacobian_matrix_assemble.")
        assert isinstance(jacobian_form_or_matrix, (list, array, BlockMatrix))
        if isinstance(jacobian_form_or_matrix, (list, array)):
            if block_jacobian is not None:
                block_assemble(jacobian_form_or_matrix, block_tensor=block_jacobian)
                return block_jacobian
            else:
                return block_assemble(jacobian_form_or_matrix)
        elif isinstance(jacobian_form_or_matrix, BlockMatrix):
            N = len(jacobian_form_or_matrix)
            M = len(jacobian_form_or_matrix[0])
            if block_jacobian is None:
                block_jacobian = BlockMatrix(N, M)
            # Assemble
            for I in range(N):
                for J in range(M):
                    block_jacobian.blocks[I, J] = jacobian_form_or_matrix.blocks[I, J]
            return block_jacobian
        else:
            raise AssertionError("Invalid case in BlockNonlinearProblem._block_jacobian_matrix_assemble.")

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

import types
from numpy import ndarray as array
from dolfin import NonlinearProblem
from multiphenics.fem import block_assemble, BlockForm1, BlockForm2
from multiphenics.la import as_backend_type, BlockDefaultFactory, GenericBlockMatrix, GenericBlockVector

class BlockNonlinearProblem(NonlinearProblem):
    def __init__(self, residual_form_or_eval, block_solution, bcs, jacobian_form_or_eval):
        NonlinearProblem.__init__(self)
        # Store the input arguments
        self.residual_form_or_eval = residual_form_or_eval
        self.jacobian_form_or_eval = jacobian_form_or_eval
        self.block_solution = block_solution
        self.bcs = bcs
        # Create block backend for wrapping
        self.block_backend = BlockDefaultFactory()
        self.block_dof_map = self.block_solution.block_function_space().block_dofmap()
        
    def F(self, fenics_residual, _):
        # Update block solution subfunctions based on the third argument, which has already been
        # updated in self.block_solution.block_vector()
        self.block_solution.apply("to subfunctions")
        # Wrap FEniCS residual into a block residual
        block_residual = self.block_backend.wrap_vector(fenics_residual)
        # Assemble the block residual
        self._block_residual_vector_assemble(block_residual, self.block_solution)
        # Apply boundary conditions
        if self.bcs is not None:
            self.bcs.apply(block_residual, self.block_solution.block_vector())
        
    def _block_residual_vector_assemble(self, block_residual, block_solution):
        assert isinstance(self.residual_form_or_eval, (list, array, BlockForm1, types.FunctionType))
        if isinstance(self.residual_form_or_eval, (list, array, BlockForm1)):
            residual_form_or_vector = self.residual_form_or_eval
        elif isinstance(self.residual_form_or_eval, types.FunctionType):
            residual_form_or_vector = self.residual_form_or_eval(block_solution)
        else:
            raise AssertionError("Invalid case in BlockNonlinearProblem._block_residual_vector_assemble.")
        assert isinstance(residual_form_or_vector, (list, array, BlockForm1, GenericBlockVector))
        if isinstance(residual_form_or_vector, (list, array, BlockForm1)):
            block_assemble(residual_form_or_vector, block_tensor=block_residual)
        elif isinstance(residual_form_or_vector, GenericBlockVector):
            as_backend_type(residual_form_or_vector).vec().swap(as_backend_type(block_residual).vec())
        else:
            raise AssertionError("Invalid case in BlockNonlinearProblem._block_residual_vector_assemble.")
        
    def J(self, fenics_jacobian, _):
        # No need to update block solution subfunctions, this has already been done in the residual
        # Wrap FEniCS jacobian into a block jacobian
        block_jacobian = self.block_backend.wrap_matrix(fenics_jacobian)
        # Assemble the block jacobian
        self._block_jacobian_matrix_assemble(block_jacobian, self.block_solution)
        # Apply boundary conditions
        if self.bcs is not None:
            self.bcs.apply(block_jacobian)
        
    def _block_jacobian_matrix_assemble(self, block_jacobian, block_solution):
        assert isinstance(self.jacobian_form_or_eval, (list, array, BlockForm2, types.FunctionType))
        if isinstance(self.jacobian_form_or_eval, (list, array, BlockForm2)):
            jacobian_form_or_matrix = self.jacobian_form_or_eval
        elif isinstance(self.jacobian_form_or_eval, types.FunctionType):
            jacobian_form_or_matrix = self.jacobian_form_or_eval(block_solution)
        else:
            raise AssertionError("Invalid case in BlockNonlinearProblem._block_jacobian_matrix_assemble.")
        assert isinstance(jacobian_form_or_matrix, (list, array, BlockForm2, GenericBlockMatrix))
        if isinstance(jacobian_form_or_matrix, (list, array, BlockForm2)):
            block_assemble(jacobian_form_or_matrix, block_tensor=block_jacobian)
        elif isinstance(jacobian_form_or_matrix, GenericBlockMatrix):
            block_jacobian.zero()
            block_jacobian += jacobian_form_or_matrix
        else:
            raise AssertionError("Invalid case in BlockNonlinearProblem._block_jacobian_matrix_assemble.")

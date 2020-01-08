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

from dolfin import NonlinearProblem
from multiphenics.fem import block_assemble
from multiphenics.la import BlockDefaultFactory

class BlockNonlinearProblem(NonlinearProblem):
    def __init__(self, residual_block_form, block_solution, bcs, jacobian_block_form):
        NonlinearProblem.__init__(self)
        # Store the input arguments
        self.residual_block_form = residual_block_form
        self.jacobian_block_form = jacobian_block_form
        self.block_solution = block_solution
        self.bcs = bcs
        # Create block backend for wrapping
        self.block_backend = BlockDefaultFactory()
        self.block_dof_map = self.block_solution.block_function_space().block_dofmap()
        
    def F(self, fenics_residual, _):
        # Update block solution subfunctions based on the third argument, which has already been
        # stored in self.block_solution.block_vector()
        self.block_solution.apply("to subfunctions")
        # Wrap FEniCS residual into a block residual
        block_residual = self.block_backend.wrap_vector(fenics_residual)
        # Assemble the block residual
        block_assemble(self.residual_block_form, block_tensor=block_residual)
        # Apply boundary conditions
        if self.bcs is not None:
            self.bcs.apply(block_residual, self.block_solution.block_vector())
        
    def J(self, fenics_jacobian, _):
        # No need to update block solution subfunctions, this has already been done in the residual
        # Wrap FEniCS jacobian into a block jacobian
        block_jacobian = self.block_backend.wrap_matrix(fenics_jacobian)
        # Assemble the block jacobian
        block_assemble(self.jacobian_block_form, block_tensor=block_jacobian, keep_diagonal=self.bcs is not None)
        # Apply boundary conditions
        if self.bcs is not None:
            self.bcs.apply(block_jacobian)

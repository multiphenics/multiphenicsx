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

from petsc4py import PETSc
from dolfin import NonlinearProblem
from multiphenics.fem import block_assemble, BlockDirichletBCLegacy

class BlockNonlinearProblem(NonlinearProblem):
    def __init__(self, residual_block_form, block_solution, bcs, jacobian_block_form):
        NonlinearProblem.__init__(self)
        # Store the input arguments
        self.residual_block_form = residual_block_form
        self.jacobian_block_form = jacobian_block_form
        self.block_solution = block_solution
        self.bcs = bcs
        # Storage for residual/jacobian tensors
        self.residual_block_vector = None
        self.jacobian_block_matrix = None
        
    def form(self, block_x):
        block_x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
    def F(self, _):
        # Update block solution subfunctions based on the last argument, which has already been
        # stored in self.block_solution.block_vector
        self.block_solution.apply("to subfunctions")
        # Assemble the block residual
        if self.residual_block_vector is None:
            self.residual_block_vector = block_assemble(self.residual_block_form)
        else:
            with self.residual_block_vector.localForm() as residual_block_vector_local:
                residual_block_vector_local.set(0.0)
            block_assemble(self.residual_block_form, block_tensor=self.residual_block_vector)
        # Apply boundary conditions
        if self.bcs is not None:
            BlockDirichletBCLegacy.apply(self.bcs, self.residual_block_vector, self.block_solution.block_vector)
        # Return
        return self.residual_block_vector
        
    def J(self, _):
        # No need to update block solution subfunctions, this has already been done in the residual
        # Assemble the block jacobian
        if self.jacobian_block_matrix is None:
            self.jacobian_block_matrix = block_assemble(self.jacobian_block_form)
        else:
            self.jacobian_block_matrix.zeroEntries()
            block_assemble(self.jacobian_block_form, block_tensor=self.jacobian_block_matrix)
        # Apply boundary conditions
        if self.bcs is not None:
            BlockDirichletBCLegacy.apply(self.bcs, self.jacobian_block_matrix, 1.)
        # Return
        return self.jacobian_block_matrix

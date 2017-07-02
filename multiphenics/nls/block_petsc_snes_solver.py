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
from dolfin import PETScSNESSolver
from multiphenics.la import as_backend_type, GenericBlockMatrix

class BlockPETScSNESSolver(PETScSNESSolver):
    def __init__(self, problem):
        PETScSNESSolver.__init__(self)
        self.problem = problem
        # =========== PETScSNESSolver::init() workaround for assembled matrices =========== #
        # Make sure to use a matrix with proper sparsity pattern if matrix_eval returns a matrix (rather than a Form)
        if isinstance(self.problem.jacobian_form_or_eval, (types.FunctionType, types.MethodType)):
            jacobian_form_or_matrix = self.problem.jacobian_form_or_eval(self.problem.block_solution)
            if isinstance(jacobian_form_or_matrix, GenericBlockMatrix):
                jacobian_matrix = as_backend_type(jacobian_form_or_matrix).mat().duplicate()
                self.snes().setJacobian(None, jacobian_matrix)
        # === end === PETScSNESSolver::init() workaround for assembled matrices === end === #
    
    def solve(self):
        PETScSNESSolver.solve(self, self.problem, self.problem.block_solution.block_vector())
        # Keep subfunctions up to date
        self.problem.block_solution.apply("to subfunctions")
        

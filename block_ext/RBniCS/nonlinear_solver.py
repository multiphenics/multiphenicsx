# Copyright (C) 2015-2017 by the RBniCS authors
# Copyright (C) 2016-2017 by the block_ext authors
#
# This file is part of the RBniCS interface to block_ext.
#
# RBniCS and block_ext are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and block_ext are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and block_ext. If not, see <http://www.gnu.org/licenses/>.
#

import types
from block_ext import BlockDirichletBC, BlockNonlinearProblem, BlockPETScSNESSolver
from RBniCS.backends.abstract import NonlinearSolver as AbstractNonlinearSolver
from block_ext.RBniCS.function import Function
from RBniCS.utils.decorators import BackendFor, Extends, override

@Extends(AbstractNonlinearSolver)
@BackendFor("block_ext", inputs=(types.FunctionType, Function.Type(), types.FunctionType, (BlockDirichletBC, None)))
class NonlinearSolver(AbstractNonlinearSolver):
    @override
    def __init__(self, block_jacobian_eval, block_solution, block_residual_eval, block_bcs=None):
        problem = BlockNonlinearProblem(block_residual_eval, block_solution, block_bcs, block_jacobian_eval)
        self.solver  = BlockPETScSNESSolver(problem)
            
    @override
    def set_parameters(self, parameters):
        self.solver.parameters.update(parameters)
        
    @override
    def solve(self):
        return self.solver.solve()
        

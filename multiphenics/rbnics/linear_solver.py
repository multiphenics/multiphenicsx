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

from multiphenics import BlockDirichletBC, block_solve
from rbnics.backends.abstract import LinearSolver as AbstractLinearSolver
from multiphenics.rbnics.matrix import Matrix
from multiphenics.rbnics.vector import Vector
from multiphenics.rbnics.function import Function
from rbnics.utils.decorators import BackendFor, dict_of, Extends, override

@Extends(AbstractLinearSolver)
@BackendFor("multiphenics", inputs=(Matrix.Type(), Function.Type(), Vector.Type(), (BlockDirichletBC, dict_of(str, BlockDirichletBC), None)))
class LinearSolver(AbstractLinearSolver):
    @override
    def __init__(self, lhs, solution, rhs, bcs=None):
        self.solution = solution
        if bcs is not None:
            # Create a copy of lhs and rhs, in order not to 
            # change the original references when applying bcs
            self.lhs = lhs.copy()
            self.lhs._block_discard_dofs = lhs._block_discard_dofs
            self.rhs = rhs.copy()
            self.rhs._block_discard_dofs = rhs._block_discard_dofs
            self.bcs = bcs
            # Apply BCs
            assert isinstance(self.bcs, (dict, BlockDirichletBC))
            if isinstance(self.bcs, BlockDirichletBC):
                self.bcs.apply(self.lhs)
                self.bcs.apply(self.rhs)
            elif isinstance(self.bcs, dict):
                for key in self.bcs:
                    self.bcs[key].apply(self.lhs)
                    self.bcs[key].apply(self.rhs)
            else:
                raise AssertionError("Invalid type for bcs.")
        else:
            self.lhs = lhs
            self.rhs = rhs
            self.bcs = None
        
    @override
    def set_parameters(self, parameters):
        assert len(parameters) == 0, "multiphenics linear solver does not accept parameters yet"
        
    @override
    def solve(self):
        block_solve(self.lhs, self.solution.block_vector(), self.rhs)
        return self.solution
        

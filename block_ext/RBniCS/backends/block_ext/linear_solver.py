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

from block_ext import BlockDirichletBC, block_solve
from RBniCS.backends.abstract import LinearSolver as AbstractLinearSolver
from block_ext.RBniCS.backends.block_ext.matrix import Matrix
from block_ext.RBniCS.backends.block_ext.vector import Vector
from block_ext.RBniCS.backends.block_ext.function import Function
from RBniCS.utils.decorators import BackendFor, Extends, list_of, override

@Extends(AbstractLinearSolver)
@BackendFor("block_ext", inputs=(Matrix.Type(), Function.Type(), Vector.Type(), (list_of(BlockDirichletBC), None)))
class LinearSolver(AbstractLinearSolver):
    @override
    def __init__(self, lhs, solution, rhs, bcs=None): # TODO deve mettere il block discard dofs?
        self.lhs = lhs
        self.solution = solution
        self.rhs = rhs
        self.bcs = bcs
        
    @override
    def solve(self):
        if self.bcs is not None:
            self.bcs.apply(self.lhs)
            self.bcs.apply(self.rhs)
        block_solve(self.lhs, self.solution.block_vector(), self.rhs)
        

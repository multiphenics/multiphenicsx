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

from block_trial_function import BlockTrialFunction
from block_test_function import BlockTestFunction
from block_split import block_split
from block_assemble import block_assemble
from block_derivative import block_derivative
from block_dirichlet_bc import BlockDirichletBC
from block_function import BlockFunction
from block_element import BlockElement
from block_function_space import BlockFunctionSpace
from block_discard_dofs import BlockDiscardDOFs
from block_solve import block_solve
from block_nonlinear_problem import BlockNonlinearProblem
from block_petsc_snes_solver import BlockPETScSNESSolver
from block_matlab_export import block_matlab_export

import sys, petsc4py
petsc4py.init(sys.argv)

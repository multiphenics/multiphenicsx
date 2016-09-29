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

import sys, petsc4py
petsc4py.init(sys.argv)

from block_ext.block_assemble import block_assemble
from block_ext.block_derivative import block_derivative
from block_ext.block_dirichlet_bc import BlockDirichletBC
from block_ext.block_discard_dofs import BlockDiscardDOFs
from block_ext.block_element import BlockElement
from block_ext.block_function import BlockFunction
from block_ext.block_function_space import BlockFunctionSpace
from block_ext.block_matlab_export import block_matlab_export
from block_ext.block_nonlinear_problem import BlockNonlinearProblem
from block_ext.block_petsc_snes_solver import BlockPETScSNESSolver
from block_ext.block_slepc_eigen_solver import BlockSLEPcEigenSolver
from block_ext.block_solve import block_solve
from block_ext.block_split import block_split
from block_ext.block_trial_function import BlockTrialFunction
from block_ext.block_test_function import BlockTestFunction

__all__ = [
    'block_assemble',
    'block_derivative',
    'BlockDirichletBC',
    'BlockDiscardDOFs',
    'BlockElement',
    'BlockFunction',
    'BlockFunctionSpace',
    'block_matlab_export',
    'BlockNonlinearProblem',
    'BlockPETScSNESSolver',
    'BlockSLEPcEigenSolver',
    'block_solve',
    'block_split',
    'BlockTrialFunction',
    'BlockTestFunction'
]

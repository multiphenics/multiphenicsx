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

from multiphenics.fem import block_adjoint, block_assemble, block_derivative, BlockDirichletBC, BlockForm, block_restrict, DirichletBC
from multiphenics.function import block_assign, BlockElement, BlockFunction, BlockFunctionSpace, block_split, BlockTestFunction, BlockTrialFunction, split, TestFunction, TrialFunction
from multiphenics.la import block_matlab_export, BlockSLEPcEigenSolver, block_solve, SLEPcEigenSolver
from multiphenics.mesh import MeshRestriction
from multiphenics.nls import BlockNonlinearProblem, BlockPETScSNESSolver

__all__ = [
    'block_adjoint',
    'block_assemble',
    'block_assign',
    'block_derivative',
    'BlockDirichletBC',
    'BlockElement',
    'BlockForm',
    'BlockFunction',
    'BlockFunctionSpace',
    'block_matlab_export',
    'BlockNonlinearProblem',
    'BlockPETScSNESSolver',
    'block_restrict',
    'BlockSLEPcEigenSolver',
    'block_solve',
    'block_split',
    'BlockTestFunction',
    'BlockTrialFunction',
    'DirichletBC',
    'MeshRestriction',
    'SLEPcEigenSolver',
    'split',
    'TestFunction',
    'TrialFunction'
]

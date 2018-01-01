# Copyright (C) 2016-2018 by the multiphenics authors
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

import multiphenics.python # TODO to patch DOLFIN with has_pybind11 function: remove after 2018.1.0 release  # noqa
from multiphenics.fem import block_adjoint, block_assemble, block_derivative, BlockDirichletBC, BlockForm, block_restrict, DirichletBC
from multiphenics.function import assign, block_assign, BlockElement, BlockFunction, BlockFunctionSpace, block_split, BlockTestFunction, BlockTrialFunction, split, TestFunction, TrialFunction
from multiphenics.io import File, plot, XDMFFile
from multiphenics.la import as_backend_type, block_matlab_export, BlockSLEPcEigenSolver, block_solve, SLEPcEigenSolver
from multiphenics.mesh import MeshRestriction
from multiphenics.nls import BlockNonlinearProblem, BlockPETScSNESSolver

__all__ = [
    'as_backend_type',
    'assign',
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
    'File',
    'MeshRestriction',
    'plot',
    'SLEPcEigenSolver',
    'split',
    'TestFunction',
    'TrialFunction',
    'XDMFFile'
]

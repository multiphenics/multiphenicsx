# Copyright (C) 2016-2019 by the multiphenics authors
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

from multiphenics.la.as_backend_type import as_backend_type
from multiphenics.la.block_default_factory import BlockDefaultFactory
from multiphenics.la.block_matlab_export import block_matlab_export
from multiphenics.la.block_petsc_factory import BlockPETScFactory
from multiphenics.la.block_petsc_matrix import BlockPETScMatrix
from multiphenics.la.block_petsc_sub_matrix import BlockPETScSubMatrix
from multiphenics.la.block_petsc_sub_vector import BlockPETScSubVector
from multiphenics.la.block_petsc_vector import BlockPETScVector
from multiphenics.la.block_slepc_eigen_solver import BlockSLEPcEigenSolver
from multiphenics.la.block_solve import block_solve
from multiphenics.la.generic_block_linear_algebra_factory import GenericBlockLinearAlgebraFactory
from multiphenics.la.generic_block_matrix import GenericBlockMatrix
from multiphenics.la.generic_block_vector import GenericBlockVector
from multiphenics.la.slepc_eigen_solver import SLEPcEigenSolver

__all__ = [
    'as_backend_type',
    'BlockDefaultFactory',
    'block_matlab_export',
    'BlockPETScFactory',
    'BlockPETScMatrix',
    'BlockPETScSubMatrix',
    'BlockPETScSubVector',
    'BlockPETScVector',
    'BlockSLEPcEigenSolver',
    'block_solve',
    'GenericBlockLinearAlgebraFactory',
    'GenericBlockMatrix',
    'GenericBlockVector',
    'SLEPcEigenSolver'
]

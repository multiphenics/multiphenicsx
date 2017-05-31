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

from multiphenics.block.la.as_backend_type import as_backend_type
from multiphenics.block.la.block_default_factory import BlockDefaultFactory
from multiphenics.block.la.block_matlab_export import block_matlab_export
from multiphenics.block.la.block_petsc_factory import BlockPETScFactory
from multiphenics.block.la.block_petsc_matrix import BlockPETScMatrix
from multiphenics.block.la.block_petsc_sub_matrix import BlockPETScSubMatrix
from multiphenics.block.la.block_petsc_sub_vector import BlockPETScSubVector
from multiphenics.block.la.block_petsc_vector import BlockPETScVector
from multiphenics.block.la.block_slepc_eigen_solver import BlockSLEPcEigenSolver
from multiphenics.block.la.block_solve import block_solve
from multiphenics.block.la.generic_block_linear_algebra_factory import GenericBlockLinearAlgebraFactory
from multiphenics.block.la.generic_block_matrix import GenericBlockMatrix
from multiphenics.block.la.generic_block_vector import GenericBlockVector
from multiphenics.block.la.get_tensor_type import get_tensor_type
from multiphenics.block.la.has_type import has_type
from multiphenics.block.la.slepc_eigen_solver import SLEPcEigenSolver

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
    'get_tensor_type',
    'has_type',
    'SLEPcEigenSolver'
]

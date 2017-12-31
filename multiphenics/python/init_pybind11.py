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

from multiphenics.python.multiphenics_compile_cpp_code import multiphenics_compile_cpp_code

# Compile cpp code
cpp = multiphenics_compile_cpp_code(
    "multiphenics",
    # Files are manually sorted to handle dependencies
    "log/log.cpp",
    "fem/BlockDofMap.cpp",
    "function/BlockFunctionSpace.cpp",
    "fem/BlockFormBase.cpp",
    "fem/BlockForm1.cpp",
    "fem/BlockForm2.cpp",
    "la/BlockMATLABExport.cpp",
    "la/BlockInsertMode.cpp",
    "la/GenericBlockVector.cpp",
    "la/GenericBlockMatrix.cpp",
    "la/BlockPETScVector.cpp",
    "la/BlockPETScMatrix.cpp",
    "la/BlockPETScSubMatrix.cpp",
    "la/BlockPETScSubVector.cpp",
    "la/GenericBlockLinearAlgebraFactory.cpp",
    "la/BlockDefaultFactory.cpp",
    "la/BlockPETScFactory.cpp",
    "function/BlockFunction.cpp",
    "fem/BlockAssemblerBase.cpp",
    "fem/BlockAssembler.cpp",
    "fem/BlockDirichletBC.cpp",
    "la/CondensedSLEPcEigenSolver.cpp",
    "la/CondensedBlockSLEPcEigenSolver.cpp",
)

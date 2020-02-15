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

import os
from multiphenics.cpp.compile_package import compile_package

# Compile package
cpp = compile_package(
    "multiphenics",
    os.path.dirname(os.path.abspath(__file__)),
    # Files are manually sorted to handle dependencies
    "fem/BlockDofMap.cpp",
    "function/BlockFunctionSpace.cpp",
    "fem/BlockForm1.cpp",
    "fem/BlockForm2.cpp",
    "la/BlockPETScSubMatrix.cpp",
    "la/BlockPETScSubVectorReadWrapper.cpp",
    "la/BlockPETScSubVectorWrapper.cpp",
    "function/BlockFunction.cpp",
    "fem/BlockSparsityPatternBuilder.cpp",
    "fem/block_assemble.cpp",
    "fem/DirichletBCLegacy.cpp",
    "fem/BlockDirichletBC.cpp",
    "fem/BlockDirichletBCLegacy.cpp",
    "la/CondensedSLEPcEigenSolver.cpp",
    "la/CondensedBlockSLEPcEigenSolver.cpp",
)

__all__ = [
    "cpp"
]

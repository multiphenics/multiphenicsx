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

from multiphenics.cpp import cpp

BlockDirichletBC_Base = cpp.fem.BlockDirichletBC

class BlockDirichletBC(BlockDirichletBC_Base):
    def __init__(self, bcs, block_function_space):
        # Store input arguments
        self._bcs = bcs
        self._block_function_space = block_function_space
        # Initialize C++ object
        BlockDirichletBC_Base.__init__(self, self._bcs, self._block_function_space._cpp_object)

    def __getitem__(self, key):
        return self._bcs[key]

    def __iter__(self):
        return self._bcs.__iter__()

    def __len__(self):
        return len(self._bcs)

    @property
    def block_function_space(self):
        return self._block_function_space

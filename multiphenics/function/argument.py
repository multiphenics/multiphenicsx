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

from ufl import Argument as ufl_Argument
from dolfinx import FunctionSpace
from multiphenics.function.block_function_space import BlockFunctionSpace

class Argument(ufl_Argument):
    def __init__(self, function_space, number, part=None, block_function_space=None, block_index=None):
        assert isinstance(function_space, FunctionSpace)
        assert isinstance(block_function_space, BlockFunctionSpace) or block_function_space is None
        assert isinstance(block_index, int) or block_index is None

        # Generate trial/test function
        ufl_Argument.__init__(self, function_space, number, part)

        # Store block function space
        assert (block_function_space is None) == (block_index is None)
        if block_function_space is not None:
            assert block_index is not None
            self._block_function_space = block_function_space
            self._block_index = block_index
        else:
            self._block_function_space = None
            self._block_index = None

    def block_function_space(self):
        if self._block_function_space is not None:
            return self._block_function_space
        else:
            raise AttributeError

    def block_index(self):
        if self.block_index is not None:
            return self._block_index
        else:
            raise AttributeError

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

from dolfin import FunctionSpace

class BlockFunctionSpace(list):
    def __init__(self, arg1, arg2=None):
        if isinstance(arg1, list):
            function_spaces = arg1
        else:
            assert arg2 is not None
            mesh = arg1
            block_element = arg2
            function_spaces = []
            for e in block_element:
                function_spaces.append(FunctionSpace(mesh, e))

        list.__init__(self, function_spaces)
        
    def sub(self, i):
        return self[i]
        

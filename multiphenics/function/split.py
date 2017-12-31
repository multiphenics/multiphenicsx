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

import types
import dolfin

def split(f):
    split_f = dolfin.split(f)
    for s in split_f:
        if hasattr(f, "block_function_space"):
            # Add a block_function_space method
            def block_function_space(self_):
                return f.block_function_space()
            s.block_function_space = types.MethodType(block_function_space, s)
        
        if hasattr(f, "block_index"):
            # Add a block_index method
            def block_index(self_):
                return f.block_index()
            s.block_index = types.MethodType(block_index, s)
    return split_f

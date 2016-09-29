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

from dolfin import File
from block_ext import BlockFunction

def function_load(directory, filename, block_V):
    fun = BlockFunction(block_V)
    for (block_index, block_fun) in enumerate(fun):
        full_filename = str(directory) + "/" + filename + "_block_" + str(block_index) + ".xml"
        file = File(full_filename)
        file >> block_fun
    return fun


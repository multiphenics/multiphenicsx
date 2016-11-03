# Copyright (C) 2015-2016 by the RBniCS authors
# Copyright (C) 2016 by the block_ext authors
#
# This file is part of the RBniCS interface to block_ext.
#
# RBniCS and block_ext are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and block_ext are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and block_ext. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import File

def function_save(fun, directory, filename):
    for (block_index, block_fun) in enumerate(fun):
        full_filename = str(directory) + "/" + filename + "_block_" + str(block_index) + ".pvd"
        file = File(full_filename, "compressed")
        file << block_fun
        full_filename = str(directory) + "/" + filename + "_block_" + str(block_index) + ".xml"
        file = File(full_filename)
        file << block_fun


# Copyright (C) 2015-2017 by the RBniCS authors
# Copyright (C) 2016-2017 by the block_ext authors
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

import os # for path
from dolfin import File
from block_ext import BlockFunction
from RBniCS.utils.mpi import is_io_process

def function_load(directory, filename, block_V, suffix=None):
    if suffix is not None:
        filename = filename + "." + str(suffix)
    fun = BlockFunction(block_V)
    for (block_index, block_fun) in enumerate(fun):
        full_filename = str(directory) + "/" + filename + "_block_" + str(block_index) + ".xml"
        file_exists = False
        if is_io_process() and os.path.exists(full_filename):
            file_exists = True
        file_exists = is_io_process.mpi_comm.bcast(file_exists, root=is_io_process.root)
        if file_exists:
            file = File(full_filename)
            file >> block_fun
        else:
            return False
    return True


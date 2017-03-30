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

from RBniCS.backends.basic import export as basic_export
import block_ext.RBniCS
from block_ext.RBniCS.function import Function
from block_ext.RBniCS.matrix import Matrix
from block_ext.RBniCS.vector import Vector
import block_ext.RBniCS.wrapping
from RBniCS.utils.decorators import backend_for
from RBniCS.utils.io import Folders

# Export a solution to file
@backend_for("block_ext", inputs=((Function.Type(), Matrix.Type(), Vector.Type()), (Folders.Folder, str), str, int, (int, str, None)))
def export(solution, directory, filename, suffix=None, component=None):
    basic_export(solution, directory, filename, suffix, component, block_ext.RBniCS, block_ext.RBniCS.wrapping)
    

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

from rbnics.backends.basic import copy as basic_copy
import block_ext.rbnics
from block_ext.rbnics.function import Function
from block_ext.rbnics.matrix import Matrix
from block_ext.rbnics.vector import Vector
import block_ext.rbnics.wrapping
from rbnics.utils.decorators import backend_for

@backend_for("block_ext", inputs=((Function.Type(), Matrix.Type(), Vector.Type()), ))
def copy(arg):
    return basic_copy(arg, block_ext.rbnics, block_ext.rbnics.wrapping)
    

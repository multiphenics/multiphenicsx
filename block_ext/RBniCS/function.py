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

from dolfin import Function
from block_ext import BlockFunction, BlockFunctionSpace
from RBniCS.utils.decorators import backend_for

_Function_Type = BlockFunction

@backend_for("block_ext", inputs=(BlockFunctionSpace, (int, None)), output=_Function_Type)
def Function(block_V, component=None):
    if component is None:
        return _Function_Type(block_V)
    else:
        return Function(block_V[component])
        
# Make BlockFunction hashable
def block_function__hash__(self):
    return hash(tuple(self)) ^ hash(type(self))
BlockFunction.__hash__ = block_function__hash__


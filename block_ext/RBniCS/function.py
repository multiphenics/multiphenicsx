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

from block_ext import BlockFunction, BlockFunctionSpace
from block_ext.RBniCS.wrapping.block_function_space import _convert_component_to_int_or_list_of_int
from RBniCS.utils.decorators import backend_for

_Function_Type = BlockFunction

@backend_for("block_ext", inputs=(BlockFunctionSpace, (str, None)), output=_Function_Type)
def Function(block_V, component=None):
    if component is None:
        return _Function_Type(block_V)
    else:
        from block_ext.RBniCS.wrapping.get_function_subspace import get_function_subspace
        return _Function_Type(get_function_subspace(block_V, component))
        
# Make BlockFunction hashable
def block_function__hash__(self):
    return hash(tuple(self)) ^ hash(type(self))
BlockFunction.__hash__ = block_function__hash__

# Make sure that _Function_Type.function_space() preserves component to index map
original__init__ = _Function_Type.__init__
def custom__init__(self, *args, **kwargs):
    if isinstance(args[0], BlockFunctionSpace) and hasattr(args[0], "_component_to_indices"):
        self._component_to_indices = args[0]._component_to_indices
    original__init__(self, *args, **kwargs)
_Function_Type.__init__ = custom__init__

# Also make _Function_Type.sub() aware of string components    
original_sub = _Function_Type.sub
def custom_sub(self, i, deepcopy=False):
    if hasattr(self, "_component_to_indices"):
        i_int = _convert_component_to_int_or_list_of_int(self, i)
        assert isinstance(i_int, (int, tuple))
        if isinstance(i_int, int):
            return original_sub(self, i_int, deepcopy)
        else:
            output = self
            for sub_i in i_int:
                output = output.sub(sub_i, deepcopy)
            return output
    else:
        return original_sub(self, i, deepcopy)
_Function_Type.sub = custom_sub


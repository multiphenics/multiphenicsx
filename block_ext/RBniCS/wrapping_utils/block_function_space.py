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

import types
from block_ext.block_function_space import BlockFunctionSpace as block_ext_BlockFunctionSpace
from RBniCS.backends.fenics.wrapping_utils.function_space import _enable_string_components, _convert_component_to_int

def BlockFunctionSpace(*args, **kwargs):
    if "components" in kwargs:
        components = kwargs["components"]
        del kwargs["components"]
    else:
        components = None
    output = block_ext_BlockFunctionSpace(*args, **kwargs)
    if components is not None:
        _enable_string_components(components, output)
    return output

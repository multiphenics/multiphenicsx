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

from dolfin import assign
from block_ext import BlockFunction
from block_ext.RBniCS.wrapping.function_copy import function_copy

def function_component(function, component, copy, weight):
    if component is None:
        if copy is True:
            output = function_copy(function)
            if weight is not None:
                output.vector()[:] *= weight
            return output
        else:
            assert weight is None, "It is not possible to weigh components without copying the vector"
            return function
    else:
        assert copy is True, "It is not possible to clear components without copying the vector"
        block_V = function.block_function_space()
        num_components = len(block_V)
        assert (
            (num_components == 1 and component == None)
                or
            (num_components > 1 and (component == None or component < num_components))
        )
        function_component = Function(block_V) # zero by default
        assign(function_component.sub(component), function.sub(component))
        if weight is not None:
            function_component.vector()[:] *= weight
        return function_component


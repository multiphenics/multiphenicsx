# Copyright (C) 2015-2017 by the RBniCS authors
# Copyright (C) 2016-2017 by the multiphenics authors
#
# This file is part of the RBniCS interface to multiphenics.
#
# RBniCS and multiphenics are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and multiphenics are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

import types
from multiphenics.block_function_space import BlockFunctionSpace as multiphenics_BlockFunctionSpace

def BlockFunctionSpace(*args, **kwargs):
    if "components" in kwargs:
        components = kwargs["components"]
        del kwargs["components"]
    else:
        components = None
    output = multiphenics_BlockFunctionSpace(*args, **kwargs)
    if components is not None:
        _enable_string_components(components, output)
    return output
    
def _enable_string_components(components, block_function_space):
    _init_component_to_indices(components, block_function_space)
    
    original_sub = block_function_space.sub
    def custom_sub(self_, i):
        assert isinstance(i, (str, int))
        i_int = _convert_component_to_int_or_list_of_int(self_, i)
        assert isinstance(i_int, (int, list))
        if isinstance(i_int, int):
            output = original_sub(i_int)
        else:
            assert isinstance(i_int, list)
            assert isinstance(i, str)
            output = list()
            for s in i_int:
                output.append(original_sub(s))
            output = BlockFunctionSpace(output, components=[i]*len(i_int))
        return output
    block_function_space.sub = types.MethodType(custom_sub, block_function_space)
    
def _init_component_to_indices(components, block_function_space):
    assert isinstance(components, list)
    block_function_space._component_to_indices = dict()
    for (index, component) in enumerate(components):
        _init_component_to_indices__recursive(component, block_function_space._component_to_indices, index)
    def component_to_indices(self_, i):
        return self_._component_to_indices[i]
    block_function_space.component_to_indices = types.MethodType(component_to_indices, block_function_space)
    
def _init_component_to_indices__recursive(components, component_to_indices, index):
    assert isinstance(components, (str, list))
    if isinstance(components, str):
        assert isinstance(index, int)
        if components not in component_to_indices:
            component_to_indices[components] = list()
        component_to_indices[components].append(index)
    elif isinstance(components, list):
        for component in components:
            _init_component_to_indices__recursive(component, component_to_indices, index)
            
def _convert_component_to_int_or_list_of_int(block_function_space, i):
    if isinstance(i, str):
        output = block_function_space._component_to_indices[i]
        if len(output) == 1:
            return output[0]
        else:
            return output
    else:
        return i

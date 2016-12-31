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

from collections import OrderedDict
from dolfin import assign
from block_ext.RBniCS.function import Function

def function_extend_or_restrict(block_function, block_function_components, block_V, block_V_components, weight, copy):
    block_function_block_V = block_function.block_function_space()
    if block_function_components is not None:
        assert isinstance(block_function_components, (int, list))
        assert not isinstance(block_function_components, tuple), "block_ext does not handle yet the case of sub components"
        if isinstance(block_function_components, int):
            block_function_block_V_index_list = [block_function_components]
        else:
            for block_function_component in block_function_components:
                assert isinstance(block_function_component, int)
                assert not isinstance(block_function_component, tuple), "block_ext does not handle yet the case of sub components"
            block_function_block_V_index_list = block_function_components
    else:
        block_function_block_V_index_list = range(len(block_function_block_V))
    if block_V_components is not None:
        assert isinstance(block_V_components, (int, list))
        assert not isinstance(block_V_components, tuple), "block_ext does not handle yet the case of sub components"
        if isinstance(block_V_components, int):
            block_V_index_list = [block_V_components]
        else:
            for block_V_component in block_V_components:
                assert isinstance(block_V_component, int)
                assert not isinstance(block_V_component, tuple), "block_ext does not handle yet the case of sub components"
            block_V_index_list = block_V_components
    else:
        block_V_index_list = range(len(block_V))
    
    block_V_to_block_function_block_V_mapping = dict()
    block_function_block_V_to_block_V_mapping = dict()
    
    if _block_function_spaces_eq(block_function_block_V, block_V, block_function_block_V_index_list, block_V_index_list):
        # Then block_function_block_V == block_V: do not need to extend nor restrict input block_function
        # Example of use case: block_function is the solution of an elliptic problem, block_V is the truth space
        if not copy:
            assert block_function_components is None, "It is not possible to extract block function components without copying the vector"
            assert block_V_components is None, "It is not possible to extract block function components without copying the vector"
            assert weight is None, "It is not possible to weigh components without copying the vector"
            return block_function
        else:
            output = Function(block_V) # zero by default
            for (block_V_index, block_function_block_V_index) in zip(block_V_index_list, block_function_block_V_index_list):
                assign(output[block_V_index], block_function[block_function_block_V_index])
                if weight is not None:
                    output[block_V_index].vector()[:] *= weight
            return output
    elif _block_function_spaces_lt(block_function_block_V, block_V, block_V_to_block_function_block_V_mapping, block_function_block_V_index_list, block_V_index_list):
        # Then block_function_block_V < block_V: need to extend input block_function
        # Example of use case: block_function is the solution of the supremizer problem of a Stokes problem,
        # block_V is the mixed (velocity, pressure) space, and you are interested in storing a extended block_function
        # (i.e. extended to zero on pressure DOFs) when defining basis functions for enriched velocity space
        assert copy is True, "It is not possible to extend block functions without copying the vector"
        extended_block_function = Function(block_V) # zero by default
        for (index_block_V, index_block_function_block_V) in block_V_to_block_function_block_V_mapping.iteritems():
            assign(extended_block_function[index_block_V], block_function[index_block_function_block_V])
            if weight is not None:
                extended_block_function[index_block_V].vector()[:] *= weight
        return extended_block_function
    elif _block_function_spaces_gt(block_function_block_V, block_V, block_function_block_V_to_block_V_mapping, block_function_block_V_index_list, block_V_index_list):
        # Then block_function_block_V > block_V: need to restrict input block_function
        # Example of use case: block_function = (y, u, p) is the solution of an elliptic optimal control problem,
        # block_V is the collapsed state (== adjoint) solution space, and you are
        # interested in storing snapshots of y or p components because of an aggregrated approach
        assert copy is True, "It is not possible to restrict block functions without copying the vector"
        restricted_block_function = Function(block_V) # zero by default
        for (index_block_function_block_V, index_block_V) in block_function_block_V_to_block_V_mapping.iteritems():
            assign(restricted_block_function[index_block_V], block_function[index_block_function_block_V])
            if weight is not None:
                restricted_block_function[index_block_V].vector()[:] *= weight
        return restricted_block_function
    
def _block_function_spaces_eq(block_V, block_W, index_block_V, index_block_W): # block_V == block_W
    block_V_sub_spaces = _get_sub_spaces(block_V, index_block_V)
    block_W_sub_spaces = _get_sub_spaces(block_W, index_block_W)
    if len(block_V_sub_spaces) != len(block_W_sub_spaces):
        return False
    for (sub_space_block_V, sub_space_block_W) in zip(block_V_sub_spaces, block_W_sub_spaces):
        assert sub_space_block_V.ufl_domain() == sub_space_block_W.ufl_domain()
        if sub_space_block_V.ufl_element() != sub_space_block_W.ufl_element():
            return False
    return True
    
def _block_function_spaces_lt(block_V, block_W, block_W_to_block_V_mapping, index_block_V, index_block_W): # block_V < block_W
    assert len(block_W_to_block_V_mapping) == 0
    block_V_sub_spaces = _get_sub_spaces(block_V, index_block_V)
    block_W_sub_spaces = _get_sub_spaces(block_W, index_block_W)
    block_W_sub_spaces_used = [False]*len(block_W_sub_spaces)
    should_return_False = False
    for (index_block_V, sub_space_block_V) in enumerate(block_V_sub_spaces):
        for (index_block_W, sub_space_block_W) in enumerate(block_W_sub_spaces):
            if (
                sub_space_block_V.ufl_domain() == sub_space_block_W.ufl_domain()
                    and
                sub_space_block_W.ufl_element() == sub_space_block_V.ufl_element()
                    and
                not block_W_sub_spaces_used[index_block_W]
            ):
                assert index_block_W not in block_W_to_block_V_mapping
                block_W_to_block_V_mapping[index_block_W] = index_block_V
                block_W_sub_spaces_used[index_block_W] = True
                break
        else: # for loop was not broken
            # There is a sub space in block_V which cannot be mapped to block_W, thus
            # block_V is larger than block_W
            should_return_False = True
            # Do not return immediately so that the map block_W_to_block_V_mapping
            # is filled in as best as possible
            
    if should_return_False:
        return False
            
    assert len(block_W_to_block_V_mapping) == len(block_V_sub_spaces) # all sub spaces were found
    
    # Avoid ambiguity that may arise if there were sub spaces of block_W that were not used but had 
    # the same type of used sub spaces
    for (index_block_W_used, sub_space_block_W_was_used) in enumerate(block_W_sub_spaces_used):
        if sub_space_block_W_was_used:
            for (index_block_W, sub_space_block_W) in enumerate(block_W_sub_spaces):
                if (
                    sub_space_block_W.ufl_domain() == block_W_sub_spaces[index_block_W_used].ufl_domain()
                        and
                    sub_space_block_W.ufl_element() == block_W_sub_spaces[index_block_W_used].ufl_element()
                        and
                    not block_W_sub_spaces_used[index_block_W]
                ):
                    raise RuntimeError("Ambiguity when querying _block_function_spaces_lt")
        
    return True
    
def _block_function_spaces_gt(block_V, block_W, block_V_to_block_W_mapping, index_block_V, index_block_W): # block_V > block_W
    return _block_function_spaces_lt(block_W, block_V, block_V_to_block_W_mapping, index_block_W, index_block_V)
    
def _get_sub_spaces(block_V, index_block_V):
    block_V_sub_spaces = [block_V[i] for i in index_block_V]
    return block_V_sub_spaces
    

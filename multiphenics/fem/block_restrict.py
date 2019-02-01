# Copyright (C) 2016-2019 by the multiphenics authors
#
# This file is part of multiphenics.
#
# multiphenics is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

from numpy import ndarray as array, zeros
from multiphenics.fem.block_dirichlet_bc import BlockDirichletBC
from multiphenics.fem.block_form import _block_form_preprocessing
from multiphenics.fem.block_form_1 import BlockForm1
from multiphenics.fem.block_form_2 import BlockForm2
from multiphenics.function import BlockFunctionSpace, BlockFunction

def block_restrict(block_input, block_function_sub_space):
    assert isinstance(block_input, (array, list, BlockDirichletBC, BlockForm1, BlockForm2, BlockFunction))
    if isinstance(block_input, (array, list, BlockForm1, BlockForm2)):
        if isinstance(block_input, (array, list)):
            (block_form, _, block_form_rank) = _block_form_preprocessing(block_input)
        elif isinstance(block_input, BlockForm2):
            block_form = block_input
            block_form_rank = 2
        elif isinstance(block_input, BlockForm1):
            block_form = block_input
            block_form_rank = 1
        if block_form_rank == 2:
            assert isinstance(block_function_sub_space, list)
            assert len(block_function_sub_space) == 2
            assert isinstance(block_function_sub_space[0], BlockFunctionSpace)
            assert isinstance(block_function_sub_space[1], BlockFunctionSpace)
            N_sub_space = block_function_sub_space[0].num_sub_spaces()
            M_sub_space = block_function_sub_space[1].num_sub_spaces()
            sub_block_form = zeros((N_sub_space, M_sub_space), dtype=object)
            for I_sub_space in range(N_sub_space):
                I_space = _sub_component_to_component(block_function_sub_space[0], I_sub_space)
                for J_sub_space in range(M_sub_space):
                    J_space = _sub_component_to_component(block_function_sub_space[1], J_sub_space)
                    sub_block_form[I_sub_space, J_sub_space] = block_form[I_space, J_space]
            return BlockForm2(sub_block_form, block_function_sub_space)
        elif block_form_rank == 1:
            assert isinstance(block_function_sub_space, (BlockFunctionSpace, list))
            if isinstance(block_function_sub_space, BlockFunctionSpace):
                block_function_sub_space = [block_function_sub_space]
            assert len(block_function_sub_space) == 1
            assert isinstance(block_function_sub_space[0], BlockFunctionSpace)
            N_sub_space = block_function_sub_space[0].num_sub_spaces()
            sub_block_form = zeros((N_sub_space, ), dtype=object)
            for I_sub_space in range(N_sub_space):
                I_space = _sub_component_to_component(block_function_sub_space[0], I_sub_space)
                sub_block_form[I_sub_space] = block_form[I_space]
            return BlockForm1(sub_block_form, block_function_sub_space)
    elif isinstance(block_input, BlockFunction):
        assert isinstance(block_function_sub_space, (BlockFunctionSpace, list))
        if isinstance(block_function_sub_space, list):
            assert len(block_function_sub_space) == 1
            block_function_sub_space = block_function_sub_space[0]
        assert isinstance(block_function_sub_space, BlockFunctionSpace)
        N_sub_space = block_function_sub_space.num_sub_spaces()
        sub_functions = list()
        for I_sub_space in range(N_sub_space):
            I_space = _sub_component_to_component(block_function_sub_space, I_sub_space)
            sub_functions.append(block_input[I_space])
        return BlockFunction(block_function_sub_space, sub_functions)
    elif isinstance(block_input, BlockDirichletBC):
        assert isinstance(block_function_sub_space, (BlockFunctionSpace, list))
        if isinstance(block_function_sub_space, list):
            assert len(block_function_sub_space) == 1
            block_function_sub_space = block_function_sub_space[0]
        assert isinstance(block_function_sub_space, BlockFunctionSpace)
        N_sub_space = block_function_sub_space.num_sub_spaces()
        sub_bcs = list()
        for I_sub_space in range(N_sub_space):
            I_space = _sub_component_to_component(block_function_sub_space, I_sub_space)
            sub_bcs.append(block_input[I_space])
        return BlockDirichletBC(sub_bcs, block_function_sub_space)
    else:
        raise AssertionError("Invalid arguments to block_restrict")
        
def _sub_component_to_component(block_function_sub_space, sub_component):
    if hasattr(block_function_sub_space, "sub_components_to_components"):
        return block_function_sub_space.sub_components_to_components[sub_component]
    else:
        return sub_component

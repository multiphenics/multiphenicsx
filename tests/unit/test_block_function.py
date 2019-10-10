# Copyright (C) 2016-2020 by the multiphenics authors
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

import pytest
from dolfin import MPI, UnitSquareMesh
from multiphenics import BlockFunctionSpace
from test_utils import assert_functions_manipulations, get_function_spaces_1, get_function_spaces_2, get_list_of_functions_1, get_list_of_functions_2, get_restrictions_1, get_restrictions_2

# Mesh
@pytest.fixture(scope="module")
def mesh():
    return UnitSquareMesh(MPI.comm_world, 4, 4)

# Single block, no restriction
@pytest.mark.parametrize("FunctionSpace", get_function_spaces_1())
def test_single_block_no_restriction(mesh, FunctionSpace):
    V = FunctionSpace(mesh)
    block_V = BlockFunctionSpace([V])
    functions = get_list_of_functions_1(block_V)
    assert_functions_manipulations(functions, block_V)

# Two blocks, no restriction
@pytest.mark.parametrize("FunctionSpaces", get_function_spaces_2())
def test_two_blocks_no_restriction(mesh, FunctionSpaces):
    (FunctionSpace1, FunctionSpace2) = FunctionSpaces
    V1 = FunctionSpace1(mesh)
    V2 = FunctionSpace2(mesh)
    block_V = BlockFunctionSpace([V1, V2])
    functions = get_list_of_functions_2(block_V)
    assert_functions_manipulations(functions, block_V)

# Single block, with restriction
@pytest.mark.parametrize("restriction", get_restrictions_1())
@pytest.mark.parametrize("FunctionSpace", get_function_spaces_1())
def test_single_block_with_restriction(mesh, restriction, FunctionSpace):
    V = FunctionSpace(mesh)
    block_V = BlockFunctionSpace([V], restrict=[restriction])
    functions = get_list_of_functions_1(block_V)
    assert_functions_manipulations(functions, block_V)

# Two blocks, with restrictions
@pytest.mark.parametrize("restriction", get_restrictions_2())
@pytest.mark.parametrize("FunctionSpaces", get_function_spaces_2())
def test_two_blocks_with_restriction(mesh, restriction, FunctionSpaces):
    (FunctionSpace1, FunctionSpace2) = FunctionSpaces
    V1 = FunctionSpace1(mesh)
    V2 = FunctionSpace2(mesh)
    block_V = BlockFunctionSpace([V1, V2], restrict=restriction)
    functions = get_list_of_functions_2(block_V)
    assert_functions_manipulations(functions, block_V)

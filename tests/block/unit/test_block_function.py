# Copyright (C) 2016-2017 by the multiphenics authors
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

from dolfin import *
from multiphenics import *
from test_utils import assert_functions_manipulations, get_function_spaces_1, get_function_spaces_2, get_list_of_functions_1, get_list_of_functions_2, get_restrictions_1, get_restrictions_2, project

# 0) Mesh definition
mesh = UnitSquareMesh(4, 4)

# 1) Single block, no restriction
log(PROGRESS, "Case 1")
for V in get_function_spaces_1(mesh):
    block_V = BlockFunctionSpace([V])
    functions = get_list_of_functions_1(block_V)
    assert_functions_manipulations(functions, block_V)

# 2) Two blocks, no restriction
log(PROGRESS, "Case 2")
for (V1, V2) in get_function_spaces_2(mesh):
    block_V = BlockFunctionSpace([V1, V2])
    functions = get_list_of_functions_2(block_V)
    assert_functions_manipulations(functions, block_V)

# 3) Single block, with restriction
for restriction in get_restrictions_1():
    log(PROGRESS, "Case 3")
    for V in get_function_spaces_1(mesh):
        block_V = BlockFunctionSpace([V], restrict=[restriction])
        functions = get_list_of_functions_1(block_V)
        assert_functions_manipulations(functions, block_V)

# 4) Two blocks, with restrictions
for restriction in get_restrictions_2():
    log(PROGRESS, "Case 4")
    for (V1, V2) in get_function_spaces_2(mesh):
        block_V = BlockFunctionSpace([V1, V2], restrict=restriction)
        functions = get_list_of_functions_2(block_V)
        assert_functions_manipulations(functions, block_V)
        

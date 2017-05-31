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
from test_utils import apply_bc_and_block_bc_matrix, assemble_and_block_assemble_matrix, assert_block_matrices_equal, get_block_bcs_1, get_block_bcs_2, get_function_spaces_1, get_function_spaces_2, get_lhs_block_form_1, get_lhs_block_form_2, get_restrictions_1, get_restrictions_2

# 0) Mesh definition
mesh = UnitSquareMesh(4, 4)

# 1) Single block, no restriction
log(PROGRESS, "Case 1")
for V in get_function_spaces_1(mesh):
    block_V = BlockFunctionSpace([V])
    block_form = get_lhs_block_form_1(block_V)
    (lhs, block_lhs) = assemble_and_block_assemble_matrix(block_form)
    assert_block_matrices_equal(lhs, block_lhs, block_V)
    for block_bcs in get_block_bcs_1(block_V):
        (lhs, block_lhs) = assemble_and_block_assemble_matrix(block_form)
        apply_bc_and_block_bc_matrix(lhs, block_lhs, block_bcs)
        assert_block_matrices_equal(lhs, block_lhs, block_V)

# 2) Two blocks, no restriction
log(PROGRESS, "Case 2")
for (V1, V2) in get_function_spaces_2(mesh):
    block_V = BlockFunctionSpace([V1, V2])
    block_form = get_lhs_block_form_2(block_V)
    (lhs, block_lhs) = assemble_and_block_assemble_matrix(block_form)
    assert_block_matrices_equal(lhs, block_lhs, block_V)
    for block_bcs in get_block_bcs_2(block_V):
        (lhs, block_lhs) = assemble_and_block_assemble_matrix(block_form)
        apply_bc_and_block_bc_matrix(lhs, block_lhs, block_bcs)
        assert_block_matrices_equal(lhs, block_lhs, block_V)

# 3) Single block, with restriction
for restriction in get_restrictions_1():
    log(PROGRESS, "Case 3")
    for V in get_function_spaces_1(mesh):
        block_V = BlockFunctionSpace([V], restrict=[restriction])
        block_form = get_lhs_block_form_1(block_V)
        (lhs, block_lhs) = assemble_and_block_assemble_matrix(block_form)
        assert_block_matrices_equal(lhs, block_lhs, block_V)
        for block_bcs in get_block_bcs_1(block_V):
            (lhs, block_lhs) = assemble_and_block_assemble_matrix(block_form)
            apply_bc_and_block_bc_matrix(lhs, block_lhs, block_bcs)
            assert_block_matrices_equal(lhs, block_lhs, block_V)

# 4) Two blocks, with restrictions
for restriction in get_restrictions_2():
    log(PROGRESS, "Case 4")
    for (V1, V2) in get_function_spaces_2(mesh):
        block_V = BlockFunctionSpace([V1, V2], restrict=restriction)
        block_form = get_lhs_block_form_2(block_V)
        (lhs, block_lhs) = assemble_and_block_assemble_matrix(block_form)
        assert_block_matrices_equal(lhs, block_lhs, block_V)
        for block_bcs in get_block_bcs_2(block_V):
            (lhs, block_lhs) = assemble_and_block_assemble_matrix(block_form)
            apply_bc_and_block_bc_matrix(lhs, block_lhs, block_bcs)
            assert_block_matrices_equal(lhs, block_lhs, block_V)
            

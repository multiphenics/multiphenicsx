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
from dolfinx import MPI, UnitSquareMesh
from multiphenics import BlockFunctionSpace
from test_utils import apply_bc_and_block_bc_vector, apply_bc_and_block_bc_vector_non_linear, assemble_and_block_assemble_vector, assert_block_functions_equal, assert_block_vectors_equal, get_block_bcs_1, get_block_bcs_2, get_function_spaces_1, get_function_spaces_2, get_rhs_block_form_1, get_rhs_block_form_2, get_subdomains_1, get_subdomains_2, Restriction

# Mesh
@pytest.fixture(scope="module")
def mesh():
    return UnitSquareMesh(MPI.comm_world, 4, 4)

# Single block, no restriction
@pytest.mark.parametrize("FunctionSpace", get_function_spaces_1())
@pytest.mark.parametrize("BlockBCs", get_block_bcs_1())
def test_single_block_no_restriction(mesh, FunctionSpace, BlockBCs):
    V = FunctionSpace(mesh)
    block_V = BlockFunctionSpace([V])
    block_form = get_rhs_block_form_1(block_V)
    block_bcs = BlockBCs(block_V)
    (rhs, block_rhs) = assemble_and_block_assemble_vector(block_form)
    apply_bc_and_block_bc_vector(rhs, block_rhs, block_bcs)
    assert_block_vectors_equal(rhs, block_rhs, block_V)
    (rhs, block_rhs) = assemble_and_block_assemble_vector(block_form)
    (function, block_function) = apply_bc_and_block_bc_vector_non_linear(rhs, block_rhs, block_bcs, block_V)
    assert_block_vectors_equal(rhs, block_rhs, block_V)
    assert_block_functions_equal(function, block_function, block_V)

# Two blocks, no restriction
@pytest.mark.parametrize("FunctionSpaces", get_function_spaces_2())
@pytest.mark.parametrize("BlockBCs", get_block_bcs_2())
def test_two_blocks_no_restriction(mesh, FunctionSpaces, BlockBCs):
    (FunctionSpace1, FunctionSpace2) = FunctionSpaces
    V1 = FunctionSpace1(mesh)
    V2 = FunctionSpace2(mesh)
    block_V = BlockFunctionSpace([V1, V2])
    block_form = get_rhs_block_form_2(block_V)
    block_bcs = BlockBCs(block_V)
    (rhs, block_rhs) = assemble_and_block_assemble_vector(block_form)
    apply_bc_and_block_bc_vector(rhs, block_rhs, block_bcs)
    assert_block_vectors_equal(rhs, block_rhs, block_V)
    (rhs, block_rhs) = assemble_and_block_assemble_vector(block_form)
    (function, block_function) = apply_bc_and_block_bc_vector_non_linear(rhs, block_rhs, block_bcs, block_V)
    assert_block_vectors_equal(rhs, block_rhs, block_V)
    assert_block_functions_equal(function, block_function, block_V)

# Single block, with restriction
@pytest.mark.parametrize("subdomain", get_subdomains_1())
@pytest.mark.parametrize("FunctionSpace", get_function_spaces_1())
@pytest.mark.parametrize("BlockBCs", get_block_bcs_1())
def test_single_block_with_restriction(mesh, subdomain, FunctionSpace, BlockBCs):
    V = FunctionSpace(mesh)
    restriction = Restriction(V, subdomain)
    block_V = BlockFunctionSpace([V], restrict=[restriction])
    block_form = get_rhs_block_form_1(block_V)
    block_bcs = BlockBCs(block_V)
    (rhs, block_rhs) = assemble_and_block_assemble_vector(block_form)
    apply_bc_and_block_bc_vector(rhs, block_rhs, block_bcs)
    assert_block_vectors_equal(rhs, block_rhs, block_V)
    (rhs, block_rhs) = assemble_and_block_assemble_vector(block_form)
    (function, block_function) = apply_bc_and_block_bc_vector_non_linear(rhs, block_rhs, block_bcs, block_V)
    assert_block_vectors_equal(rhs, block_rhs, block_V)
    assert_block_functions_equal(function, block_function, block_V)

# Two blocks, with restrictions
@pytest.mark.parametrize("subdomains", get_subdomains_2())
@pytest.mark.parametrize("FunctionSpaces", get_function_spaces_2())
@pytest.mark.parametrize("BlockBCs", get_block_bcs_2())
def test_two_blocks_with_restriction(mesh, subdomains, FunctionSpaces, BlockBCs):
    (FunctionSpace1, FunctionSpace2) = FunctionSpaces
    (subdomain1, subdomain2) = subdomains
    V1 = FunctionSpace1(mesh)
    V2 = FunctionSpace2(mesh)
    restriction1 = Restriction(V1, subdomain1)
    restriction2 = Restriction(V2, subdomain2)
    block_V = BlockFunctionSpace([V1, V2], restrict=[restriction1, restriction2])
    block_form = get_rhs_block_form_2(block_V)
    block_bcs = BlockBCs(block_V)
    (rhs, block_rhs) = assemble_and_block_assemble_vector(block_form)
    apply_bc_and_block_bc_vector(rhs, block_rhs, block_bcs)
    assert_block_vectors_equal(rhs, block_rhs, block_V)
    (rhs, block_rhs) = assemble_and_block_assemble_vector(block_form)
    (function, block_function) = apply_bc_and_block_bc_vector_non_linear(rhs, block_rhs, block_bcs, block_V)
    assert_block_vectors_equal(rhs, block_rhs, block_V)
    assert_block_functions_equal(function, block_function, block_V)

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

from block_ext.rbnics.wrapping.block_form_types import BlockFormTypes, TupleOfBlockFormTypes
from block_ext.rbnics.wrapping.block_function_space import BlockFunctionSpace
from block_ext.rbnics.wrapping.function_copy import function_copy
from block_ext.rbnics.wrapping.function_extend_or_restrict import function_extend_or_restrict
from block_ext.rbnics.wrapping.function_load import function_load
from block_ext.rbnics.wrapping.function_save import function_save
from block_ext.rbnics.wrapping.functions_list_basis_functions_matrix_mul import functions_list_basis_functions_matrix_mul_online_matrix, functions_list_basis_functions_matrix_mul_online_vector, functions_list_basis_functions_matrix_mul_online_function
from block_ext.rbnics.wrapping.get_mpi_comm import get_mpi_comm
from block_ext.rbnics.wrapping.get_function_subspace import get_function_subspace
from block_ext.rbnics.wrapping.gram_schmidt_projection_step import gram_schmidt_projection_step
from block_ext.rbnics.wrapping.matrix_mul import matrix_mul_vector, vectorized_matrix_inner_vectorized_matrix
from block_ext.rbnics.wrapping.tensor_copy import tensor_copy
from block_ext.rbnics.wrapping.tensor_load import tensor_load
from block_ext.rbnics.wrapping.tensor_save import tensor_save
from block_ext.rbnics.wrapping.tensors_list_mul import tensors_list_mul_online_function
from block_ext.rbnics.wrapping.vector_mul import vector_mul_vector

__all__ = [
    'BlockFormTypes',
    'BlockFunctionSpace',
    'function_copy',
    'function_extend_or_restrict',
    'function_load',
    'function_save',
    'functions_list_basis_functions_matrix_mul_online_matrix', 
    'functions_list_basis_functions_matrix_mul_online_vector', 
    'functions_list_basis_functions_matrix_mul_online_function',
    'get_function_subspace',
    'get_mpi_comm',
    'gram_schmidt_projection_step',
    'matrix_mul_vector',
    'tensor_copy',
    'tensor_load',
    'tensor_save',
    'tensors_list_mul_online_function',
    'TupleOfBlockFormTypes',
    'vector_mul_vector',
    'vectorized_matrix_inner_vectorized_matrix'
]


__overridden__ = [
    'BlockFunctionSpace'
]

# Copyright (C) 2016 by the block_ext authors
#
# This file is part of block_ext.
#
# block_ext is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# block_ext is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with block_ext. If not, see <http://www.gnu.org/licenses/>.
#

from block_ext.RBniCS.backends.block_ext.wrapping.function_component import function_component
from block_ext.RBniCS.backends.block_ext.wrapping.function_copy import function_copy
from block_ext.RBniCS.backends.block_ext.wrapping.function_load import function_load
from block_ext.RBniCS.backends.block_ext.wrapping.function_save import function_save
from block_ext.RBniCS.backends.block_ext.wrapping.functions_list_mul import functions_list_mul_online_matrix, functions_list_mul_online_vector, functions_list_mul_online_function
from block_ext.RBniCS.backends.block_ext.wrapping.get_mpi_comm import get_mpi_comm
from block_ext.RBniCS.backends.block_ext.wrapping.gram_schmidt_projection_step import gram_schmidt_projection_step
from block_ext.RBniCS.backends.block_ext.wrapping.matrix_mul import matrix_mul_vector, vectorized_matrix_inner_vectorized_matrix
from block_ext.RBniCS.backends.block_ext.wrapping.tensor_copy import tensor_copy
from block_ext.RBniCS.backends.block_ext.wrapping.tensor_load import tensor_load
from block_ext.RBniCS.backends.block_ext.wrapping.tensor_save import tensor_save
from block_ext.RBniCS.backends.block_ext.wrapping.tensors_list_mul import tensors_list_mul_online_function
from block_ext.RBniCS.backends.block_ext.wrapping.vector_mul import vector_mul_vector

__all__ = [
    'function_component',
    'function_copy',
    'function_load',
    'function_save',
    'functions_list_mul_online_matrix', 
    'functions_list_mul_online_vector', 
    'functions_list_mul_online_function',
    'get_mpi_comm',
    'gram_schmidt_projection_step',
    'matrix_mul_vector',
    'tensor_copy',
    'tensor_load',
    'tensor_save',
    'tensors_list_mul_online_function',
    'vector_mul_vector',
    'vectorized_matrix_inner_vectorized_matrix'
]

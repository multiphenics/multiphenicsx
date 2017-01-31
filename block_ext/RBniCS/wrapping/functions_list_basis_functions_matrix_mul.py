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

from block_ext import BlockFunctionSpace
from block_ext.RBniCS.wrapping.function_copy import function_copy
from RBniCS.backends.basic.wrapping import functions_list_basis_functions_matrix_adapter
from RBniCS.backends.numpy.matrix import Matrix as OnlineMatrix
from RBniCS.backends.numpy.vector import Vector as OnlineVector
from RBniCS.backends.numpy.function import Function as OnlineFunction

def functions_list_basis_functions_matrix_mul_online_matrix(functions_list_basis_functions_matrix, online_matrix, FunctionsListBasisFunctionsMatrixType, backend):
    block_V = functions_list_basis_functions_matrix.V_or_Z
    (functions, _) = functions_list_basis_functions_matrix_adapter(functions_list_basis_functions_matrix, backend)
    assert isinstance(block_V, BlockFunctionSpace)
    assert isinstance(online_matrix, OnlineMatrix.Type())
    
    output = FunctionsListBasisFunctionsMatrixType(block_V)
    dim = online_matrix.shape[1]
    for j in range(dim):
        assert len(online_matrix[:, j]) == len(functions)
        output_j = function_copy(functions[0])
        output_j.block_vector().zero()
        for (block_index, block_output_j) in enumerate(output_j):
            for (i, fun_i) in enumerate(functions):
                block_output_j.vector().add_local(fun_i.block_vector()[block_index].array()*online_matrix.item((i, j)))
            block_output_j.vector().apply("add")
        output.enrich(output_j)
    return output

def functions_list_basis_functions_matrix_mul_online_vector(functions_list_basis_functions_matrix, online_vector, backend):
    (functions, _) = functions_list_basis_functions_matrix_adapter(functions_list_basis_functions_matrix, backend)
    assert isinstance(online_vector, (OnlineVector.Type(), tuple))
    
    output = function_copy(functions[0])
    output.block_vector().zero()
    
    for (block_index, block_output) in enumerate(output):
        if isinstance(online_vector, OnlineVector.Type()):
            for (i, fun_i) in enumerate(functions):
                block_output.vector().add_local(fun_i.block_vector()[block_index].array()*online_vector.item(i))
        elif isinstance(online_vector, tuple):
            for (i, fun_i) in enumerate(functions):
                block_output.vector().add_local(fun_i.block_vector()[block_index].array()*online_vector[i])
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in functions_list_basis_functions_matrix_mul_online_vector.")
        
        block_output.vector().apply("add")
        
    return output
    
def functions_list_basis_functions_matrix_mul_online_function(functions_list_basis_functions_matrix, online_function, backend):
    assert isinstance(online_function, OnlineFunction.Type())
    
    return functions_list_basis_functions_matrix_mul_online_vector(functions_list_basis_functions_matrix, online_function.vector(), backend)
    

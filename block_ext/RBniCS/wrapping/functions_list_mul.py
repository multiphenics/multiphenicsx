# Copyright (C) 2015-2016 by the RBniCS authors
# Copyright (C) 2016 by the block_ext authors
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
from RBniCS.backends.numpy.matrix import Matrix as OnlineMatrix
from RBniCS.backends.numpy.vector import Vector as OnlineVector
from RBniCS.backends.numpy.function import Function as OnlineFunction

def functions_list_mul_online_matrix(functions_list, online_matrix, FunctionsListType):
    block_V = functions_list.V_or_Z
    assert isinstance(block_V, BlockFunctionSpace)
    assert isinstance(online_matrix, OnlineMatrix.Type())
    
    output = FunctionsListType(block_V)
    dim = online_matrix.shape[1]
    for j in range(dim):
        assert len(online_matrix[:, j]) == len(functions_list._list)
        output_j = function_copy(functions_list._list[0])
        output_j.block_vector().zero()
        for (i, fun_i) in enumerate(functions_list._list):
            output_j.block_vector().add_local(fun_i.block_vector().block_array()*online_matrix.item((i, j)))
        output_j.block_vector().apply("add")
        output.enrich(output_j)
    return output

def functions_list_mul_online_vector(functions_list, online_vector):
    assert isinstance(online_vector, (OnlineVector.Type(), tuple))
    
    output = function_copy(functions_list._list[0])
    output.block_vector().zero()
    if isinstance(online_vector, OnlineVector.Type()):
        for (i, fun_i) in enumerate(functions_list._list):
            output.block_vector().add_local(fun_i.block_vector().block_array()*online_vector.item(i))
    elif isinstance(online_vector, tuple):
        for (i, fun_i) in enumerate(functions_list._list):
            output.block_vector().add_local(fun_i.block_vector().block_array()*online_vector[i])
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in functions_list_mul_online_vector.")
        
    output.block_vector().apply("add")
    return output
    
def functions_list_mul_online_function(functions_list, online_function):
    assert isinstance(online_function, OnlineFunction.Type())
    
    return functions_list_mul_online_vector(functions_list, online_function.block_vector())
    

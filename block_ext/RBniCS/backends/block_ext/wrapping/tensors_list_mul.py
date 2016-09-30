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

from block_ext.RBniCS.backends.block_ext.wrapping.tensor_copy import tensor_copy
from block_ext.RBniCS.backends.block_ext.matrix import Matrix
from block_ext.RBniCS.backends.block_ext.vector import Vector
from RBniCS.backends.numpy.function import Function as OnlineFunction

def tensors_list_mul_online_function(tensors_list, online_function):
    assert isinstance(online_function, OnlineFunction.Type())
    online_vector = online_function.block_vector()
    
    output = tensor_copy(tensors_list._list[0])
    output.zero()
    assert isinstance(output, (Matrix.Type(), Vector.Type()))
    if isinstance(output, Matrix.Type()):
        for (i, matrix_i) in enumerate(tensors_list._list):
            output += matrix_i*online_vector.item(i)
    elif isinstance(output, Vector.Type()):
        for (i, vector_i) in enumerate(tensors_list._list):
            output.add_local(vector_i.block_array()*online_vector.item(i))
        output.apply("add")
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in tensors_list_mul_online_function.")
    return output
    

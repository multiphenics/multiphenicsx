# Copyright (C) 2015-2017 by the RBniCS authors
# Copyright (C) 2016-2017 by the multiphenics authors
#
# This file is part of the RBniCS interface to multiphenics.
#
# RBniCS and multiphenics are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and multiphenics are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

from multiphenics.rbnics.wrapping.tensor_copy import tensor_copy
import multiphenics.rbnics # avoid circular imports when importing rbnics backend
import rbnics.backends # avoid circular imports when importing numpy backend

def tensors_list_mul_online_function(tensors_list, online_function):
    assert isinstance(online_function, rbnics.backends.numpy.Function.Type())
    online_vector = online_function.block_vector()
    
    output = tensor_copy(tensors_list._list[0])
    assert isinstance(output, (multiphenics.rbnics.Matrix.Type(), multiphenics.rbnics.Vector.Type()))
    if isinstance(output, multiphenics.rbnics.Matrix.Type()):
        for (block_index_I, block_output_I) in enumerate(output):
            for (block_index_I, block_output_IJ) in enumerate(block_output_I):
                block_output_IJ.zero()
                for (i, matrix_i) in enumerate(tensors_list._list):
                    block_output_IJ += matrix_i[block_index_I, block_index_J]*online_vector.item(i)
    elif isinstance(output, multiphenics.rbnics.Vector.Type()):
        for (block_index, block_output) in enumerate(output):
            block_output.zero()
            for (i, vector_i) in enumerate(tensors_list._list):
                block_output.add_local(vector_i[block_index].array()*online_vector.item(i))
        output.apply("add")
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in tensors_list_mul_online_function.")
    return output
    

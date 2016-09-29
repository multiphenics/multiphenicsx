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

from block_ext.RBniCS.backends.block_ext.function import Function
from block_ext.RBniCS.backends.block_ext.matrix import Matrix
from block_ext.RBniCS.backends.block_ext.vector import Vector
from block_ext.RBniCS.backends.block_ext.wrapping import function_copy, tensor_copy
from RBniCS.utils.decorators import backend_for

# Compute the difference between two solutions
@backend_for("block_ext", inputs=((Function.Type(), Matrix.Type(), Vector.Type()), (Function.Type(), Matrix.Type(), Vector.Type())))
def difference(solution1, solution2):
    assert (
        (isinstance(solution1, Function.Type()) and isinstance(solution2, Function.Type()))
            or
        (isinstance(solution1, (Matrix.Type(), Vector.Type())) and isinstance(solution2, (Matrix.Type(), Vector.Type())))
    )
    if isinstance(solution1, Function.Type()):
        output = function_copy(solution1)
        output.block_vector().add_local( - solution2.block_vector().block_array() )
        output.block_vector().apply("add")
        return output
    elif isinstance(solution1, Matrix.Type()):
        output = tensor_copy(solution1)
        output -= solution2
        return output
    elif isinstance(solution1, Vector.Type()):
        output = tensor_copy(solution1)
        output.add_local( - solution2.block_array() )
        output.apply("add")
        return output
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in difference.")
    

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
from block_ext.RBniCS.backends.block_ext.wrapping import function_save, tensor_save
from RBniCS.utils.decorators import backend_for
from RBniCS.utils.io import Folders

# Export a solution to file
@backend_for("block_ext", inputs=((Function.Type(), Matrix.Type(), Vector.Type()), (Folders.Folder, str), str))
def export(solution, directory, filename):
    assert isinstance(solution, (Function.Type(), Matrix.Type(), Vector.Type()))
    if isinstance(solution, Function.Type()):
        function_save(solution, directory, filename)
    elif isinstance(solution, (Matrix.Type(), Vector.Type())):
        tensor_save(solution, directory, filename)
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in export.")
    

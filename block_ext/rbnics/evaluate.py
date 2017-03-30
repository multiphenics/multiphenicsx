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

from block_ext.rbnics.matrix import Matrix
from block_ext.rbnics.vector import Vector
from block_ext.rbnics.function import Function
from block_ext.rbnics.functions_list import FunctionsList
from block_ext.rbnics.tensors_list import TensorsList
from block_ext.rbnics.parametrized_tensor_factory import ParametrizedTensorFactory
from block_ext.rbnics.parametrized_expression_factory import ParametrizedExpressionFactory
from block_ext.rbnics.reduced_mesh import ReducedMesh
from block_ext.rbnics.reduced_vertices import ReducedVertices
from rbnics.utils.decorators import backend_for, tuple_of

# Evaluate a parametrized expression, possibly at a specific location
#@backend_for("block_ext", inputs=((Matrix.Type(), Vector.Type(), Function.Type(), TensorsList, FunctionsList, ParametrizedTensorFactory, ParametrizedExpressionFactory), (ReducedMesh, ReducedVertices, None)))
def evaluate(expression_, at=None):
    pass # TODO
    

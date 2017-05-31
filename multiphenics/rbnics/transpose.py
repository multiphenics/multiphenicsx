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

from rbnics.backends.basic import transpose as basic_transpose
import multiphenics.rbnics
from multiphenics.rbnics.basis_functions_matrix import BasisFunctionsMatrix
from multiphenics.rbnics.function import Function
from multiphenics.rbnics.functions_list import FunctionsList
from multiphenics.rbnics.vector import Vector
import multiphenics.rbnics.wrapping
import rbnics.backends.numpy
from rbnics.utils.decorators import backend_for

@backend_for("multiphenics", online_backend="numpy", inputs=((BasisFunctionsMatrix, Function.Type(), FunctionsList, Vector.Type()), ))
def transpose(arg):
    return basic_transpose(arg, multiphenics.rbnics, multiphenics.rbnics.wrapping, rbnics.backends.numpy)
    

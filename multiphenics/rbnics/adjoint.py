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

from numpy import ndarray as array
from dolfin import adjoint as dolfin_adjoint
from rbnics.utils.decorators import backend_for
from multiphenics.block_adjoint import block_adjoint
from multiphenics.rbnics.wrapping import BlockFormTypes, TupleOfBlockFormTypes

@backend_for("multiphenics", inputs=(BlockFormTypes + TupleOfBlockFormTypes, ))
def adjoint(arg):
    assert isinstance(arg, (array, list, tuple))
    if not isinstance(arg, tuple):
        return block_adjoint(arg)
    else:
        adjoint_arg = list()
        for a in arg:
            adjoint_arg.append(block_adjoint(a))
        return tuple(adjoint_arg)
        

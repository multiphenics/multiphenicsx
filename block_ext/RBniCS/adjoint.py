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

from numpy import ndarray as array
from ufl import Form
from dolfin import adjoint as dolfin_adjoint
from block_ext.block_adjoint import block_adjoint
from RBniCS.utils.decorators import array_of, backend_for, list_of, tuple_of

@backend_for("block_ext", inputs=((list_of(Form), list_of(list_of(Form)), array_of(Form), array_of(array_of(Form)), tuple_of(list_of(Form)), tuple_of(list_of(list_of(Form))), tuple_of(array_of(Form)), tuple_of(array_of(array_of(Form)))), ))
def adjoint(arg):
    assert isinstance(arg, (array, list, tuple))
    if not isinstance(arg, tuple):
        return block_adjoint(arg)
    else:
        adjoint_arg = list()
        for a in arg:
            adjoint_arg.append(block_adjoint(a))
        return tuple(adjoint_arg)
        

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

from ufl import Form
from RBniCS.utils.decorators import array_of, list_of, tuple_of

BlockFormTypes = (
    list_of(Form),
    list_of(list_of(Form)), 
    array_of(Form), 
    array_of(array_of(Form)), 
    # Forms with 0 placeholders
    list_of((Form, int)),
    list_of(list_of((Form, int))), 
    array_of((Form, int)), 
    array_of(array_of((Form, int))), 
    # Forms with 0. placeholders
    list_of((Form, float)),
    list_of(list_of((Form, float))), 
    array_of((Form, float)), 
    array_of(array_of((Form, float))), 
    # Forms with 0. and 0 placeholders
    list_of((Form, float, int)),
    list_of(list_of((Form, float, int))), 
    array_of((Form, float, int)), 
    array_of(array_of((Form, float, int))), 
)

TupleOfBlockFormTypes = tuple(tuple_of(BlockFormType) for BlockFormType in BlockFormTypes)


# Copyright (C) 2016-2019 by the multiphenics authors
#
# This file is part of multiphenics.
#
# multiphenics is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import as_backend_type, assign
from multiphenics.function.block_function import BlockFunction

def block_assign(object_to, object_from):
    assert isinstance(object_to, BlockFunction) and isinstance(object_from, BlockFunction)
    as_backend_type(object_from.block_vector()).vec().copy(as_backend_type(object_to.block_vector()).vec())
    object_to.block_vector().apply("insert")
    for (function_to, function_from) in zip(object_to, object_from):
        assign(function_to, function_from)

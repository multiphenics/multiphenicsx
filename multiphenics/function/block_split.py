# Copyright (C) 2016-2020 by the multiphenics authors
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

from multiphenics.function.block_function import BlockFunction
from multiphenics.function.block_test_function import BlockTestFunction
from multiphenics.function.block_trial_function import BlockTrialFunction

def block_split(f):
    assert isinstance(f, (BlockFunction, BlockTestFunction, BlockTrialFunction))
    if isinstance(f, (BlockTestFunction, BlockTrialFunction)):
        assert isinstance(f, tuple)
        return f
    elif isinstance(f, BlockFunction):
        return tuple(subf for subf in f)

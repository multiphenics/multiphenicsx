# Copyright (C) 2016-2017 by the block_ext authors
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

from dolfin import TestFunction
from block_ext.block_test_trial_function_base import BlockTestTrialFunction_Base

class BlockTestFunction(BlockTestTrialFunction_Base):
    def __new__(cls, arg1):
        return BlockTestTrialFunction_Base.__new__(cls, arg1, TestFunction)
        
    def __init__(self, arg1):
        BlockTestTrialFunction_Base.__init__(self, arg1, TestFunction)

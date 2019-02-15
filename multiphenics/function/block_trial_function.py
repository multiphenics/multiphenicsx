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

from multiphenics.function.block_test_trial_function_base import BlockTestTrialFunction_Base
from multiphenics.function.trial_function import TrialFunction

class BlockTrialFunction(BlockTestTrialFunction_Base):
    def __new__(cls, arg1):
        return BlockTestTrialFunction_Base.__new__(cls, arg1, TrialFunction)
        
    def __init__(self, arg1):
        BlockTestTrialFunction_Base.__init__(self, arg1, TrialFunction)

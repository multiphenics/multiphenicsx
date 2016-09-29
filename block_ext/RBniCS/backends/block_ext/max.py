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

from block_ext.RBniCS.backends.block_ext.abs import AbsOutput
from RBniCS.utils.decorators import backend_for

# max function to compute the maximum absolute value of entries in EIM. To be used in combination with abs,
# even though abs actually carries out both the max and the abs!
@backend_for("block_ext", inputs=(AbsOutput, ))
def max(abs_output):
    return (abs_output.max_abs_return_value, abs_output.max_abs_return_location)
        

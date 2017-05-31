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

from multiphenics.rbnics.abs import AbsOutput
from rbnics.utils.decorators import backend_for

# max function to compute the maximum absolute value of entries in EIM. To be used in combination with abs,
# even though abs actually carries out both the max and the abs!
@backend_for("multiphenics", inputs=(AbsOutput, ))
def max(abs_output):
    return (abs_output.max_abs_return_value, abs_output.max_abs_return_location)
        

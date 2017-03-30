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

import block_ext.RBniCS # avoid circular imports when importing RBniCS backend

def vector_mul_vector(vector1, vector2):
    if isinstance(vector1, block_ext.RBniCS.Function.Type()):
        vector1 = vector1.block_vector()
    if isinstance(vector2, block_ext.RBniCS.Function.Type()):
        vector2 = vector2.block_vector()
    return vector1.inner(vector2)


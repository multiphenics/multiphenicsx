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

# Make sure that the parent block_ext package is imported (rather than the one
# in this folder which has the same name!)
from __future__ import absolute_import

# Import RBniCS backends
import RBniCS.backends

# Make sure to import block_ext backend, so that it is added to the factory storage
import block_ext.RBniCS.backends.block_ext

# Enable block_ext backend
from RBniCS.utils.factories import enable_backend
enable_backend("block_ext")

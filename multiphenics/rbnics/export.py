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

from rbnics.backends.basic import export as basic_export
import multiphenics.rbnics
from multiphenics.rbnics.function import Function
from multiphenics.rbnics.matrix import Matrix
from multiphenics.rbnics.vector import Vector
import multiphenics.rbnics.wrapping
from rbnics.utils.decorators import backend_for
from rbnics.utils.io import Folders

# Export a solution to file
@backend_for("multiphenics", inputs=((Function.Type(), Matrix.Type(), Vector.Type()), (Folders.Folder, str), str, int, (int, str, None)))
def export(solution, directory, filename, suffix=None, component=None):
    basic_export(solution, directory, filename, suffix, component, multiphenics.rbnics, multiphenics.rbnics.wrapping)
    

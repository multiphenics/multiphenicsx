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

import dolfin
from multiphenics.mesh import MeshRestriction

def plot(obj, *args, **kwargs):
    if isinstance(obj, MeshRestriction):
        for (d, mesh_function_d) in enumerate(obj):
            dolfin.plot(mesh_function_d, title="MeshFunction of dimension " + str(d), *args, **kwargs)
    else:
        dolfin.plot(obj, *args, **kwargs)

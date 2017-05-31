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

from multiphenics.block_function_space import BlockFunctionSpace
from rbnics.backends.fenics import ReducedMesh as FEniCSReducedMesh
from rbnics.utils.decorators import BackendFor, Extends, override

@Extends(FEniCSReducedMesh)
@BackendFor("multiphenics", inputs=(BlockFunctionSpace, ))
class ReducedMesh(FEniCSReducedMesh):
    def __init__(self, V, subdomain_data=None, **kwargs):
        FEniCSReducedMesh.__init__(self, V, subdomain_data, **kwargs)
        
        

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

import types
from dolfinx import Mesh, MeshFunction
from dolfinx.cpp.mesh import MeshFunctionSizet

class MeshRestriction(object):
    def __init__(self, mesh, mesh_functions=None):
        # Store mesh
        assert isinstance(mesh, Mesh)
        self._mesh = mesh
        # Initialize mesh functions
        D = mesh.topology.dim
        if mesh_functions is not None:
            assert isinstance(mesh_functions, list)
            assert len(mesh_functions) == D + 1
            assert all([isinstance(mesh_function, MeshFunctionSizet) for mesh_function in mesh_functions])
            assert all([mesh_functions[d].dim == d for d in range(D + 1)])
            self._mesh_functions = mesh_functions
        else:
            self._mesh_functions = [MeshFunction("size_t", mesh, d, 0) for d in range(D + 1)]
            
    def mesh(self):
        return self._mesh
        
    def __getitem__(self, dim):
        return self._mesh_functions[dim]
        
    def mark(self, marking_function):
        assert isinstance(marking_function, types.FunctionType)
        D = self._mesh.topology.dim
        for d in range(D + 1):
            self._mesh_functions[d].mark(marking_function, 1)

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

from dolfin import Mesh, MeshFunction, SubDomain

class MeshRestriction(list):
    """
    This type converts a SubDomain into a hierarchy of MeshFunctions.
    """
    def __init__(self, mesh, arg=None):
        # Initialize empty list
        list.__init__(self)
        # Process depending on the second argument
        assert isinstance(mesh, Mesh)
        assert isinstance(arg, (list, SubDomain)) or arg is None
        if isinstance(arg, list):
            assert all([isinstance(arg_i, SubDomain) for arg_i in arg])
        if arg is None:
            pass # leave the list empty
        elif isinstance(arg, (list, SubDomain)):
            D = mesh.topology.dim
            for d in range(D + 1):
                mesh_function_d = MeshFunction("bool", mesh, d, False)
                if isinstance(arg, SubDomain):
                    arg.mark(mesh_function_d, True)
                else:
                    for arg_i in arg:
                        arg_i.mark(mesh_function_d, True)
                self.append(mesh_function_d)
        else:
            raise TypeError("Invalid second argument provided to MeshRestriction")

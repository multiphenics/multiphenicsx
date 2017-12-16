# Copyright (C) 2016-2017 by the multiphenics authors
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

import os
from dolfin import File, Mesh, MeshFunction, SubDomain, XDMFFile

class MeshRestriction(list):
    """
    This type converts a SubDomain into a hierarchy of MeshFunctions.
    """
    def __init__(self, mesh, arg):
        # Initialize empty list
        list.__init__(self)
        # Process depending on the second argument
        assert isinstance(mesh, Mesh)
        assert isinstance(arg, (list, str, SubDomain)) or arg is None
        if isinstance(arg, list):
            assert all([isinstance(arg_i, SubDomain) for arg_i in arg])
        if arg is None:
            pass # leave the list empty
        elif isinstance(arg, (list, SubDomain)):
            D = mesh.topology().dim()
            for d in range(D + 1):
                mesh_function_d = MeshFunction("bool", mesh, d)
                mesh_function_d.set_all(False)
                if isinstance(arg, SubDomain):
                    arg.mark(mesh_function_d, True)
                else:
                    for arg_i in arg:
                        arg_i.mark(mesh_function_d, True)
                self.append(mesh_function_d)
        elif isinstance(arg, str):
            self._read(mesh, arg)
            
    def _read(self, mesh, filename, encoding=None):
        assert filename.endswith(".rtc.xml") or filename.endswith(".rtc.xdmf")
        # Read in MeshFunctions
        D = mesh.topology().dim()
        for d in range(D + 1):
            if filename.endswith(".rtc.xml"):
                mesh_function_d = MeshFunction("bool", mesh, filename + "/mesh_function_" + str(d) + ".xml")
            else:
                mesh_function_d = MeshFunction("bool", mesh, d)
                xdmf_file = XDMFFile(filename + "/mesh_function_" + str(d) + ".xdmf")
                if encoding is not None:
                    xdmf_file.read(mesh_function_d, encoding)
                else:
                    xdmf_file.read(mesh_function_d)
            self.append(mesh_function_d)
            
    def _write(self, filename, encoding=None):
        assert filename.endswith(".rtc.xml") or filename.endswith(".rtc.xdmf")
        # Create output folder
        try:
            os.makedirs(filename)
        except OSError:
            if not os.path.isdir(filename):
                raise
        # Write out MeshFunctions
        for (d, mesh_function_d) in enumerate(self):
            if filename.endswith(".rtc.xml"):
                File(filename + "/mesh_function_" + str(d) + ".xml") << mesh_function_d
            else:
                xdmf_file = XDMFFile(filename + "/mesh_function_" + str(d) + ".xdmf")
                if encoding is not None:
                    xdmf_file.write(mesh_function_d, encoding)
                else:
                    xdmf_file.write(mesh_function_d)

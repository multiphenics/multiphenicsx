# Copyright (C) 2016-2018 by the multiphenics authors
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

class MeshRestrictionXDMFFile(object):
    def __init__(self, filename):
        self.filename = filename
    
    def write(self, content, encoding=None):
        content._write(self.filename, encoding)
            
def XDMFFile(arg1, arg2=None):
    if arg2 is None:
        assert isinstance(arg1, str)
        filename = arg1
        if filename.endswith(".rtc.xdmf"):
            return MeshRestrictionXDMFFile(filename)
        else:
            return dolfin.XDMFFile(filename)
    else:
        assert isinstance(arg2, str)
        mpi_comm = arg1
        filename = arg2
        assert not filename.endswith(".rtc.xdmf")
        return dolfin.XDMFFile(mpi_comm, filename)

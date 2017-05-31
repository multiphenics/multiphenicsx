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

import dolfin
from multiphenics.block.mesh import MeshRestriction

class File(object):
    def __init__(self, filename, encoding=None):
        if filename.endswith(".rtc"):
            self.filename = filename
        else:
            if encoding is not None:
                self.backend = dolfin.File(filename, encoding)
            else:
                self.backend = dolfin.File(filename)
        
    def __lshift__(self, content):
        if isinstance(content, MeshRestriction):
            content._write(self.filename)
        else:
            self.backend.__lshift__(content)
            

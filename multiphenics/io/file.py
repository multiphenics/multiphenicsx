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

class MeshRestrictionXMLFile(object):
    def __init__(self, filename, encoding=None):
        self.filename = filename
    
    def write(self, content):
        content._write(self.filename)
            
    def __lshift__(self, content):
        self.write(content)
            
def File(filename, encoding=None):
    if filename.endswith(".rtc.xml"):
        assert encoding is None
        return MeshRestrictionXMLFile(filename)
    else:
        if encoding is not None:
            return dolfin.File(filename, encoding)
        else:
            return dolfin.File(filename)

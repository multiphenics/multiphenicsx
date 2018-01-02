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

try:
    from dolfin import has_pybind11
except ImportError:
    from dolfin import __version__ as dolfin_version
    if dolfin_version.startswith("2018.1.0"):
        def has_pybind11():
            return True
    else:
        def has_pybind11():
            return False
    import dolfin
    dolfin.has_pybind11 = has_pybind11

if has_pybind11():
    from multiphenics.python.init_pybind11 import cpp
else:
    from multiphenics.python.init_swig import cpp
    
__all__ = [
    'cpp'
]
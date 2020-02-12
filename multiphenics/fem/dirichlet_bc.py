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

from dolfinx import DirichletBC as dolfinx_DirichletBC

class DirichletBC(dolfinx_DirichletBC):
    def __init__(self, value, dofs, V=None):
        # Call parent constructor
        dolfinx_DirichletBC.__init__(self, value, dofs, V)
        # Store the (python) function space. This is already available as a property in the public interface,
        # but it casts the function space to a C++ FunctionSpace and then wraps it into a python FunctionSpace,
        # losing all the customization that we have done in the block_function_space.py file (most notably, the
        # block_function_space() method)
        if V is None:
            self._function_space = value.function_space
        else:
            self._function_space = V
        
    @property
    def function_space(self):
        return self._function_space

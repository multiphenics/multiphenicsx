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

import types
from dolfin import DirichletBC as dolfin_DirichletBC

def DirichletBC(*args, **kwargs):
    # Call the constructor
    output = dolfin_DirichletBC(*args, **kwargs)
    # Deduce private variable values from arguments
    if len(args) == 1 and isinstance(args[0], dolfin_DirichletBC):
        assert len(kwargs) == 0
        _function_space = args[0]._function_space
    else:
        _function_space = args[0]
    # Override the function_space() method. This is already available in the public interface,
    # but it casts the function space to a C++ FunctionSpace and then wraps it into a python FunctionSpace,
    # losing all the customization that we have done in the function_space.py file (most notably, the
    # block_function_space() method)
    output._function_space = _function_space
    def function_space(self_):  # nopep8
        return self_._function_space
    output.function_space = types.MethodType(function_space, output)
    # Return
    return output

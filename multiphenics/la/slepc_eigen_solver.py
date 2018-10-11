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
from dolfin import DirichletBC, Function
import dolfin.cpp
from multiphenics.python import cpp
    
def DecorateGetEigenPair(SLEPcEigenSolver):
    class DecoratedSLEPcEigenSolver(SLEPcEigenSolver):
        def get_eigenpair(self, r_fun, c_fun, i):
            assert isinstance(r_fun, Function)
            assert isinstance(c_fun, Function)
            (lr, lc, _, _) = SLEPcEigenSolver.get_eigenpair(self, r_fun._cpp_object, c_fun._cpp_object, i)
            return (lr, lc, r_fun, c_fun)
    
    return DecoratedSLEPcEigenSolver
    
def SLEPcEigenSolver(A, B=None, bcs=None):
    if bcs is None:
        EigenSolver = DecorateGetEigenPair(dolfin.SLEPcEigenSolver)
        return EigenSolver(A, B)
    else:
        assert isinstance(bcs, (DirichletBC, list))
        if isinstance(bcs, DirichletBC):
            bcs = [bcs]
        else:
            assert all([isinstance(bc, DirichletBC) for bc in bcs])
        EigenSolver = DecorateGetEigenPair(cpp.la.CondensedSLEPcEigenSolver)
        return EigenSolver(A, B, bcs)

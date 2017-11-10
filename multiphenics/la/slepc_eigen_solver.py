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
from dolfin import DirichletBC, Function, has_pybind11
from multiphenics.python import cpp

def DecorateGetEigenPair(SLEPcEigenSolver):
    class DecoratedSLEPcEigenSolver(SLEPcEigenSolver):
        def get_eigenpair(self, i, r_vec=None, c_vec=None):
            if isinstance(r_vec, Function):
                r_vec_in = None # cannot use r_vec due to different ghosting
            else:
                r_vec_in = r_vec
            if isinstance(c_vec, Function):
                c_vec_in = None # cannot use r_vec due to different ghosting
            else:
                c_vec_in = c_vec
            (lr, lc, r_vec_out, c_vec_out) = SLEPcEigenSolver.get_eigenpair(self, i, r_vec_in, c_vec_in)
            if isinstance(r_vec, Function):
                r_vec.vector().set_local(r_vec_out.array())
                r_vec.vector().apply("insert")
                r_vec_out = r_vec
            if isinstance(c_vec, Function):
                c_vec.vector().set_local(c_vec_out.array())
                c_vec.vector().apply("insert")
                c_vec_out = c_vec
            return (lr, lc, r_vec_out, c_vec_out)
    
    return DecoratedSLEPcEigenSolver

def SLEPcEigenSolver(A, B=None, bcs=None):
    if bcs is None:
        EigenSolver = DecorateGetEigenPair(dolfin.SLEPcEigenSolver)
        return EigenSolver(A, B)
    else:
        assert isinstance(bcs, (DirichletBC, list))
        if has_pybind11():
            EigenSolver = DecorateGetEigenPair(cpp.la.CondensedSLEPcEigenSolver)
        else:
            EigenSolver = DecorateGetEigenPair(cpp.CondensedSLEPcEigenSolver)
        if isinstance(bcs, DirichletBC):
            return EigenSolver(A, B, [bcs])
        else:
            assert all([isinstance(bc, DirichletBC) for bc in bcs])
            return EigenSolver(A, B, bcs)

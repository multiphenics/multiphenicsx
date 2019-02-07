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

from dolfin import DirichletBC
from multiphenics.python import cpp
    
def SLEPcEigenSolver(A, B=None, bcs=None):
    mpi_comm = A.getComm()    
    
    if bcs is None:
        eigen_solver = cpp.la.SLEPcEigenSolver(mpi_comm)
    else:
        assert isinstance(bcs, (DirichletBC, list))
        if isinstance(bcs, DirichletBC):
            bcs = [bcs]
        else:
            assert all([isinstance(bc, DirichletBC) for bc in bcs])
        eigen_solver = cpp.la.CondensedSLEPcEigenSolver(mpi_comm)
        eigen_solver.set_boundary_conditions(bcs)
        
    eigen_solver.set_operators(A, B)
    return eigen_solver

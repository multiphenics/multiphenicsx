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

from petsc4py import PETSc
from multiphenics.la.as_backend_type import as_backend_type
from multiphenics.la.generic_block_matrix import GenericBlockMatrix
from multiphenics.la.generic_block_vector import GenericBlockVector

def block_matlab_export(block_tensor, name_tensor):
    block_tensor = as_backend_type(block_tensor)
    assert isinstance(block_tensor, (GenericBlockMatrix, GenericBlockVector))
    if isinstance(block_tensor, GenericBlockMatrix):
        viewer = PETSc.Viewer().createASCII(name_tensor + ".m", comm= PETSc.COMM_WORLD)
        viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB)
        viewer.view(block_tensor.mat())
        viewer.popFormat()
    elif isinstance(block_tensor, GenericBlockVector):
        viewer = PETSc.Viewer().createASCII(name_tensor + ".m", comm= PETSc.COMM_WORLD)
        viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB)
        viewer.view(block_tensor.vec())
        viewer.popFormat()
    else:
        raise AssertionError("Invalid arguments provided to MATLAB export")


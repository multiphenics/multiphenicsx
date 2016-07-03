# Copyright (C) 2016 by the block_ext authors
#
# This file is part of block_ext.
#
# block_ext is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# block_ext is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with block_ext. If not, see <http://www.gnu.org/licenses/>.
#

from block_matrix import BlockMatrix
from block_vector import BlockVector
from monolithic_matrix import MonolithicMatrix
from block_discard_dofs import BlockDiscardDOFs
from petsc4py import PETSc

def block_matlab_export(block_A, name_A, block_b, name_b, block_discard_dofs=None):
    assert isinstance(block_A, BlockMatrix)
    assert isinstance(block_b, BlockVector)
    # Init monolithic matrix/vector corresponding to block matrix/vector
    A = MonolithicMatrix(block_A, block_discard_dofs=block_discard_dofs)
    b = A.create_monolithic_vector_left(block_b)
    # Copy values from block matrix/vector to monolithic matrix/vector
    A.zero(); A.block_add(block_A)
    b.zero(); b.block_add(block_b)
    # Export
    A_viewer = PETSc.Viewer().createASCII(name_A + ".m", comm= PETSc.COMM_WORLD)
    A_viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB)
    A_viewer.view(A.mat())
    A_viewer.popFormat()
    b_viewer = PETSc.Viewer().createASCII(name_b + ".m", comm= PETSc.COMM_WORLD)
    b_viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB)
    b_viewer.view(b.vec())
    b_viewer.popFormat()

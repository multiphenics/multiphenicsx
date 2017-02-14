# Copyright (C) 2016-2017 by the block_ext authors
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

from block_ext.block_matrix import BlockMatrix
from block_ext.block_vector import BlockVector
from block_ext.monolithic_matrix import MonolithicMatrix
from petsc4py import PETSc

def block_matlab_export(block_A, name_A, block_b=None, name_b=None, block_x=None, name_x=None):
    assert isinstance(block_A, BlockMatrix)
    assert isinstance(block_b, BlockVector) or block_b is None
    if block_b is not None:
        if block_b._block_discard_dofs is None:
            assert block_A._block_discard_dofs[0] is None
            assert block_A._block_discard_dofs[1] is None
        else:
            assert block_A._block_discard_dofs[0] == block_b._block_discard_dofs
    block_discard_dofs = block_A._block_discard_dofs
    # Init monolithic matrix/vector corresponding to block matrix/vector
    A = MonolithicMatrix(block_A, block_discard_dofs=block_discard_dofs)
    if block_b is not None and block_x is None:
        b = A.create_monolithic_vector_left(block_b)
    elif block_b is not None and block_x is not None:
        x, b = A.create_monolithic_vectors(block_x, block_b)
    # Copy values from block matrix/vector to monolithic matrix/vector
    A.zero(); A.block_add(block_A)
    if block_b is not None:
        b.zero(); b.block_add(block_b)
    if block_x is not None:
        x.zero(); x.block_add(block_x)
    # Export
    A_viewer = PETSc.Viewer().createASCII(name_A + ".m", comm= PETSc.COMM_WORLD)
    A_viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB)
    A_viewer.view(A.mat())
    A_viewer.popFormat()
    if block_b is not None:
        b_viewer = PETSc.Viewer().createASCII(name_b + ".m", comm= PETSc.COMM_WORLD)
        b_viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB)
        b_viewer.view(b.vec())
        b_viewer.popFormat()
    if block_x is not None:
        x_viewer = PETSc.Viewer().createASCII(name_x + ".m", comm= PETSc.COMM_WORLD)
        x_viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB)
        x_viewer.view(x.vec())
        x_viewer.popFormat()

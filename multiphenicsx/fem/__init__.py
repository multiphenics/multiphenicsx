# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tools for assembling and manipulating finite element forms"""

from multiphenicsx.fem.assemble import (
    apply_lifting, apply_lifting_nest,
    assemble_matrix, assemble_matrix_block,
    assemble_matrix_nest, assemble_scalar,
    assemble_vector, assemble_vector_block,
    assemble_vector_nest, create_matrix,
    BlockMatSubMatrixWrapper,
    BlockVecSubVectorReadWrapper, BlockVecSubVectorWrapper,
    create_matrix_block, create_matrix_nest,
    create_vector, create_vector_block,
    create_vector_nest, set_bc, set_bc_nest,
    NestVecSubVectorReadWrapper, NestVecSubVectorWrapper,
    MatSubMatrixWrapper, NestMatSubMatrixWrapper,
    VecSubVectorReadWrapper, VecSubVectorWrapper)
from multiphenicsx.fem.dofmap_restriction import DofMapRestriction

__all__ = [
    "create_vector", "create_vector_block", "create_vector_nest",
    "create_matrix", "create_matrix_block", "create_matrix_nest",
    "apply_lifting", "apply_lifting_nest", "assemble_scalar", "assemble_vector",
    "assemble_vector_block", "assemble_vector_nest",
    "assemble_matrix_block", "assemble_matrix_nest",
    "assemble_matrix", "set_bc", "set_bc_nest",
    "VecSubVectorReadWrapper", "VecSubVectorWrapper",
    "BlockVecSubVectorReadWrapper", "BlockVecSubVectorWrapper",
    "NestVecSubVectorReadWrapper", "NestVecSubVectorWrapper",
    "MatSubMatrixWrapper", "BlockMatSubMatrixWrapper", "NestMatSubMatrixWrapper",
    "DofMapRestriction"
]

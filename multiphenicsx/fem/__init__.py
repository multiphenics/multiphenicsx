# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tools for assembling finite element forms with restrictions."""

from multiphenicsx.fem.assemble import (
    apply_lifting, apply_lifting_nest, assemble_matrix, assemble_matrix_block, assemble_matrix_nest, assemble_vector,
    assemble_vector_block, assemble_vector_nest, BlockMatSubMatrixWrapper, BlockVecSubVectorReadWrapper,
    BlockVecSubVectorWrapper, create_matrix, create_matrix_block, create_matrix_nest, create_vector,
    create_vector_block, create_vector_nest, MatSubMatrixWrapper, NestMatSubMatrixWrapper, NestVecSubVectorReadWrapper,
    NestVecSubVectorWrapper, set_bc, set_bc_nest, VecSubVectorReadWrapper, VecSubVectorWrapper)
from multiphenicsx.fem.dofmap_restriction import DofMapRestriction

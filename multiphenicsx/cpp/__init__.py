# Copyright (C) 2016-2021 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from mpi4py import MPI
from multiphenicsx.cpp.compile_package import compile_package

# Compile package
cpp = compile_package(
    MPI.COMM_WORLD,
    "multiphenicsx",
    os.path.dirname(os.path.abspath(__file__)),
    # Files are manually sorted to handle dependencies
    "fem/DofMapRestriction.cpp",
    "fem/sparsitybuild.cpp",
    "fem/petsc.cpp",
    "fem/utils.cpp",
    "la/petsc.cpp"
)

__all__ = [
    "cpp"
]

# Copyright (C) 2016-2023 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""multiphenicsx cpp module."""

import os

import mpi4py.MPI

from multiphenicsx.cpp.compile_code import compile_code
from multiphenicsx.cpp.compile_package import compile_package

cpp_library = compile_package(
    mpi4py.MPI.COMM_WORLD,
    "multiphenicsx",
    os.path.dirname(os.path.abspath(__file__)),
    # Files are manually sorted to handle dependencies
    "fem/DofMapRestriction.cpp",
    "fem/sparsitybuild.cpp",
    "fem/petsc.cpp",
    "fem/utils.cpp",
    "la/petsc.cpp"
)

# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""multiphenicsx main module."""

# Simplify import of mpi4py.MPI and petsc4py.PETSc in internal modules (and slepc4py.SLEPc
# in tutorials, if available) by importing them here once and for all. Internal modules will
# now only need to import the main packages mpi4py and petsc4py (and slepc4py, if available).
import mpi4py
import mpi4py.MPI  # noqa: F401
import petsc4py
import petsc4py.PETSc  # noqa: F401

try:
    import slepc4py
except ImportError:  # pragma: no cover
    pass
else:
    import slepc4py.SLEPc  # noqa: F401


# We require a very small subset of the numpy typing library, namely NDArray. To avoid enforcing
# a requirement numpy>=1.21.0, we mock numpy.typing.NDArray for older numpy versions.
import types
import typing

import numpy

try:
    import numpy.typing
except ImportError:  # pragma: no cover
    numpy.typing = types.ModuleType("typing", "Mock numpy.typing module")
finally:
    if not hasattr(numpy.typing, "NDArray"):  # pragma: no cover
        numpy.typing.NDArray = typing.Iterable

# Clean up imported names so that they are not visible to end users
del typing
del types

del mpi4py
del numpy
del petsc4py

try:
    del slepc4py
except NameError:  # pragma: no cover
    pass

# Copyright (C) 2016-2025 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Restriction of a DofMap to a list of active degrees of freedom."""


import dolfinx.cpp as dcpp
import dolfinx.fem
import numpy as np
import numpy.typing as npt

from multiphenicsx.cpp import cpp_library as mcpp


class DofMapRestriction(mcpp.fem.DofMapRestriction):  # type: ignore[misc, no-any-unimported]
    """Restriction of a DofMap to a list of active degrees of freedom."""

    def __init__(  # type: ignore[no-any-unimported]
        self,
        dofmap: dcpp.fem.DofMap | dolfinx.fem.DofMap,
        restriction: npt.NDArray[np.int32]
    ) -> None:
        # Extract cpp dofmap
        try:
            _dofmap = dofmap._cpp_object
        except AttributeError:  # pragma: no cover
            _dofmap = dofmap
        super().__init__(_dofmap, restriction)

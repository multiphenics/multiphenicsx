# Copyright (C) 2016-2023 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Restriction of a DofMap to a list of active degrees of freedom."""

import typing

import dolfinx.cpp as dcpp
import dolfinx.fem
import numpy as np
import numpy.typing

from multiphenicsx.cpp import cpp_library as mcpp


class DofMapRestriction(mcpp.fem.DofMapRestriction):  # type: ignore[misc, no-any-unimported]
    """Restriction of a DofMap to a list of active degrees of freedom."""

    def __init__(  # type: ignore[no-any-unimported]
        self,
        dofmap: typing.Union[dcpp.fem.DofMap, dolfinx.fem.DofMap],
        restriction: np.typing.NDArray[np.int32],
        legacy: bool = False
    ) -> None:
        # Extract cpp dofmap
        try:
            _dofmap = dofmap._cpp_object
        except AttributeError:  # pragma: no cover
            _dofmap = dofmap
        super().__init__(_dofmap, restriction, legacy)

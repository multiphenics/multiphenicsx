# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Restriction of a DofMap to a list of active degrees of freedom."""

import typing

import dolfinx.cpp as dcpp
import dolfinx.fem

from multiphenicsx.cpp import cpp_library as mcpp


class DofMapRestriction(mcpp.fem.DofMapRestriction):
    """Restriction of a DofMap to a list of active degrees of freedom."""

    def __init__(
        self,
        dofmap: typing.Union[dcpp.fem.DofMap, dolfinx.fem.DofMap],
        restriction: typing.List[int]
    ) -> None:
        # Extract cpp dofmap
        try:
            _dofmap = dofmap._cpp_object
        except AttributeError:  # pragma: no cover
            _dofmap = dofmap
        super().__init__(_dofmap, restriction)

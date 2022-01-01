# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import typing

import dolfinx.cpp as dcpp
from dolfinx.fem.dofmap import DofMap

from multiphenicsx.cpp import cpp as mcpp


class DofMapRestriction(mcpp.fem.DofMapRestriction):
    def __init__(
            self,
            dofmap: typing.Union[dcpp.fem.DofMap, DofMap],
            restriction: typing.List[int]):
        """Restriction of a DofMap to a list of active degrees of freedom
        """
        # Extract cpp dofmap
        try:
            _dofmap = dofmap._cpp_object
        except AttributeError:
            _dofmap = dofmap
        super().__init__(_dofmap, restriction)

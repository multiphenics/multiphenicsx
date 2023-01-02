# Copyright (C) 2016-2023 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tools for assembling finite element forms with restrictions."""


import multiphenicsx.fem.petsc  # import module rather than its content to be consistent with dolfinx.fem.petsc
from multiphenicsx.fem.dofmap_restriction import DofMapRestriction

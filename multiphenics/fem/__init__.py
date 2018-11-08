# Copyright (C) 2016-2020 by the multiphenics authors
#
# This file is part of multiphenics.
#
# multiphenics is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

from multiphenics.fem.block_adjoint import block_adjoint
from multiphenics.fem.block_assemble import block_assemble
from multiphenics.fem.block_derivative import block_derivative
from multiphenics.fem.block_dirichlet_bc import BlockDirichletBC
from multiphenics.fem.block_dof_map import BlockDofMap
from multiphenics.fem.block_flatten_nested import block_flatten_nested
from multiphenics.fem.block_form import BlockForm
from multiphenics.fem.block_form_1 import BlockForm1
from multiphenics.fem.block_form_2 import BlockForm2
from multiphenics.fem.block_replace_zero import block_replace_zero
from multiphenics.fem.block_restrict import block_restrict
from multiphenics.fem.dirichlet_bc import DirichletBC

__all__ = [
    'block_adjoint',
    'block_assemble',
    'block_derivative',
    'BlockDirichletBC',
    'BlockDofMap',
    'block_flatten_nested',
    'BlockForm',
    'BlockForm1',
    'BlockForm2',
    'block_replace_zero',
    'block_restrict',
    'DirichletBC'
]

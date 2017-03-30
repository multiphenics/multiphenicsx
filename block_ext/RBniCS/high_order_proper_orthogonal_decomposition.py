# Copyright (C) 2015-2017 by the RBniCS authors
# Copyright (C) 2016-2017 by the block_ext authors
#
# This file is part of the RBniCS interface to block_ext.
#
# RBniCS and block_ext are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and block_ext are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and block_ext. If not, see <http://www.gnu.org/licenses/>.
#

from block_ext import BlockFunctionSpace
from RBniCS.backends.abstract import HighOrderProperOrthogonalDecomposition as AbstractHighOrderProperOrthogonalDecomposition
from RBniCS.backends.basic import ProperOrthogonalDecompositionBase as BasicHighOrderProperOrthogonalDecomposition
import block_ext.RBniCS
import block_ext.RBniCS.wrapping
from RBniCS.utils.decorators import BackendFor, Extends, override

HighOrderProperOrthogonalDecompositionBase = BasicHighOrderProperOrthogonalDecomposition(AbstractHighOrderProperOrthogonalDecomposition)

@Extends(HighOrderProperOrthogonalDecompositionBase)
@BackendFor("block_ext", inputs=(BlockFunctionSpace, ))
class HighOrderProperOrthogonalDecomposition(HighOrderProperOrthogonalDecompositionBase):
    @override
    def __init__(self, V, empty_tensor):
        HighOrderProperOrthogonalDecompositionBase.__init__(self, V, None, empty_tensor, block_ext.RBniCS, block_ext.RBniCS.wrapping, block_ext.RBniCS.TensorSnapshotsList, block_ext.RBniCS.TensorBasisList)
        
    @override
    def store_snapshot(self, snapshot):
        self.snapshots_matrix.enrich(snapshot)
        

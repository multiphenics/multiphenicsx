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

from rbnics.backends.basic import GramSchmidt as BasicGramSchmidt
import block_ext.rbnics
from block_ext.rbnics.matrix import Matrix
import block_ext.rbnics.wrapping
from rbnics.utils.decorators import BackendFor, Extends, override

@Extends(BasicGramSchmidt)
@BackendFor("block_ext", inputs=(Matrix.Type(), ))
class GramSchmidt(BasicGramSchmidt):
    @override
    def __init__(self, X):
        BasicGramSchmidt.__init__(self, X, block_ext.rbnics, block_ext.rbnics.wrapping)
        

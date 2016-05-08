# Copyright (C) 2016 by the block_ext authors
#
# This file is part of block_ext.
#
# block_ext is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# block_ext is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with block_ext. If not, see <http://www.gnu.org/licenses/>.
#

class MonolithicIndexHelper(object):
    def __init__(self, row_start, N, n):
        self.row_start = row_start
        self.N = N
        self.n = n
    
    def block_index_init(self):
        row_start = self.row_start
        N = self.N
        n = self.n
        # The block index is equal to the smallest integer I for which
        # the integer division row_start/sum(n[:(I+1)]) is zero
        for I in range(N):
            if row_start/sum(n[:(I+1)]) == 0:
                break
        return I
    
    def local_index_init(self, I):
        row_start = self.row_start
        n = self.n
        return row_start - sum(n[:I])
        
    def indices_increment(self, I, i):
        n = self.n
        i = i + 1
        if i < n[I]:
            return I, i
        else:
            return I+1, 0
        

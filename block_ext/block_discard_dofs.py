# Copyright (C) 2016-2017 by the block_ext authors
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

from dolfin import Function, LagrangeInterpolator
import numpy as np
from mpi4py import MPI

# Assume to have a function space W \subset V
# which are conforming (same degree, conforming mesh)
# but defined on different (conforming) mesh,
# where in particular mesh(W) \subset mesh(V)
# This class allows you to assemble on the larger space V
# (otherwise FEniCS would give an error) and then discard
# unnecessary dofs
class _BlockDiscardDOFs(object):
    def __init__(self, V_subspaces, V_spaces):
        assert len(V_subspaces) == len(V_spaces)
        N = len(V_subspaces)
        
        # Detect if dofs should be discarded or not, based on the fact that
        # space and subspace are different
        self.need_to_discard_dofs = []
        for I in range(N):
            if V_subspaces[I] is not V_spaces[I]:
                self.need_to_discard_dofs.append(True)
            else:
                self.need_to_discard_dofs.append(False)
        
        # Interpolator
        interpolator = LagrangeInterpolator()
        
        # Create a map from space dofs to subspace dofs,
        # and precompute and store the number of DOFs to be discarded by row/columns
        self.subspace_dofs_extended = list()
        self.dofs_to_be_discarded = list()
        for I in range(N):
            if self.need_to_discard_dofs[I]:
                local_subspace_dofs = np.array(range(*V_subspaces[I].dofmap().ownership_range()), dtype=np.float)
                subspace_dofs = Function(V_subspaces[I])
                subspace_dofs.vector().set_local(local_subspace_dofs)
                subspace_dofs.vector().apply("")
                subspace_dofs.vector()[:] += 1 # otherwise you cannot distinguish dof 0 from the value=0. which is assigned on space \setminus subspace
                subspace_dofs_extended = Function(V_spaces[I])
                interpolator.interpolate(subspace_dofs_extended, subspace_dofs)
                subspace_dofs.vector()[:] -= 1
                subspace_dofs_extended.vector()[:] -= 1 # in this way dofs which are not in the subspace are marked by -1
                subspace_dofs_extended_rounded = np.round(subspace_dofs_extended.vector().array())
                assert np.max(np.abs( subspace_dofs_extended_rounded - subspace_dofs_extended.vector().array() )) < 1.e-10, \
                    "Extension has produced non-integer DOF IDs. Are you sure that the function space and subspace are conforming?"
                    
                subspace_dofs_extended_integer = subspace_dofs_extended_rounded.astype('i')
                dofs_to_be_kept_local_indices = np.where(subspace_dofs_extended_integer >= 0)[0]
                dofs_to_be_kept_global_indices = [V_spaces[I].dofmap().local_to_global_index(local_dof) for local_dof in dofs_to_be_kept_local_indices]
                assert len(np.unique(subspace_dofs_extended_integer[dofs_to_be_kept_local_indices])) == len(subspace_dofs_extended_integer[dofs_to_be_kept_local_indices])
                subspace_dofs_extended_dict = dict(zip(dofs_to_be_kept_global_indices, subspace_dofs_extended_integer[dofs_to_be_kept_local_indices]))
                
                # Need to (all_)gather the array because row indices operator only local dofs, but
                # col indices operate on local and non-local dofs
                comm = V_subspaces[I].mesh().mpi_comm().tompi4py()
                allgathered_subspace_dofs_extended_dict = comm.bcast(subspace_dofs_extended_dict, root=0)
                for r in range(1, comm.size):
                    allgathered_subspace_dofs_extended_dict.update( comm.bcast(subspace_dofs_extended_dict, root=r) )
                    
                self.subspace_dofs_extended.append(allgathered_subspace_dofs_extended_dict)
                
                # In contrast, negative dofs should be discarded
                dofs_to_be_discarded_local_indices = np.where(subspace_dofs_extended_integer < 0)[0]
                dofs_to_be_discarded_global_indices = [V_spaces[I].dofmap().local_to_global_index(local_dof) for local_dof in dofs_to_be_discarded_local_indices]
                assert len(np.unique(dofs_to_be_discarded_global_indices)) == len(dofs_to_be_discarded_global_indices)
                allgathered_dofs_to_be_discarded = comm.bcast(dofs_to_be_discarded_global_indices, root=0)
                for r in range(1, comm.size):
                    allgathered_dofs_to_be_discarded.extend( comm.bcast(dofs_to_be_discarded_global_indices, root=r) )
                self.dofs_to_be_discarded.append(set(allgathered_dofs_to_be_discarded))
            else:
                self.subspace_dofs_extended.append(None)
                self.dofs_to_be_discarded.append(None)
            
        # In a similar way, create a map from subspace dofs to space dofs
        self.space_dofs_restricted = list()
        for I in range(N):
            if self.need_to_discard_dofs[I]:
                self.space_dofs_restricted.append(dict(zip(self.subspace_dofs_extended[I].values(), self.subspace_dofs_extended[I].keys())))
            else:
                self.space_dofs_restricted.append(None)
                
def BlockDiscardDOFs(V_subspaces, V_spaces):
    if (V_subspaces, V_spaces) not in _all_block_discard_dofs:
        _all_block_discard_dofs[(V_subspaces, V_spaces)] = _BlockDiscardDOFs(V_subspaces, V_spaces)
    return _all_block_discard_dofs[(V_subspaces, V_spaces)]

_all_block_discard_dofs = dict()

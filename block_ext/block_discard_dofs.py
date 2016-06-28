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
class BlockDiscardDOFs(list):
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
        
        # Create a map between subspace dofs and space dofs
        self.subspace_dofs_extended = []
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
                # Need to (all_)gather the array because row indices operator only local dofs, but
                # col indices operate on local and non-local dofs
                comm = V_subspaces[I].mesh().mpi_comm().tompi4py()
                subspace_dofs_extended = comm.bcast(subspace_dofs_extended_rounded, root=0)
                for r in range(1, comm.size):
                    subspace_dofs_extended = np.append( subspace_dofs_extended, comm.bcast(subspace_dofs_extended_rounded, root=r) )

                self.subspace_dofs_extended.append(subspace_dofs_extended.astype('i'))
            else:
                self.subspace_dofs_extended.append(None)
            
        # In a similar way, create a map between space dofs and subspace dofs
        self.space_dofs_restricted = []
        for I in range(N):
            if self.need_to_discard_dofs[I]:
                local_space_dofs = np.array(range(*V_spaces[I].dofmap().ownership_range()), dtype=np.float)
                space_dofs = Function(V_spaces[I])
                space_dofs.vector().set_local(local_space_dofs)
                space_dofs.vector().apply("")
                space_dofs_restricted = Function(V_subspaces[I])
                interpolator.interpolate(space_dofs_restricted, space_dofs)
                space_dofs_restricted_rounded = np.round(space_dofs_restricted.vector().array())
                assert np.max(np.abs( space_dofs_restricted_rounded - space_dofs_restricted.vector().array() )) < 1.e-10, \
                    "Restriction has produced non-integer DOF IDs. Are you sure that the function space and subspace are conforming?"
                # No need to gather the result, since we only use this in for local row indices
                self.space_dofs_restricted.append(space_dofs_restricted_rounded.astype('i'))
            else:
                self.space_dofs_restricted.append(None)
        
        # Precompute and store the number of DOFs to be discarded by row/columns
        self.dofs_to_be_discarded = []
        for I in range(N):
            if self.need_to_discard_dofs[I]:
                self.dofs_to_be_discarded.append( np.where(self.subspace_dofs_extended[I] < 0)[0] )
                # Non-discarded DOFs should be consecutive numbers, starting from 0
                assert len(self.dofs_to_be_discarded[I]) + np.max(self.subspace_dofs_extended[I]) + 1 == len(self.subspace_dofs_extended[I])
            else:
                self.dofs_to_be_discarded.append(None)
        

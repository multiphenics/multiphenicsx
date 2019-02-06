// Copyright (C) 2016-2020 by the multiphenics authors
//
// This file is part of multiphenics.
//
// multiphenics is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// multiphenics is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
//

#include <multiphenics/la/BlockPETScSubMatrix.h>
#include <multiphenics/log/log.h>

using namespace multiphenics;
using namespace multiphenics::la;

using dolfin::la::petsc_error;
using multiphenics::fem::BlockDofMap;

//-----------------------------------------------------------------------------
BlockPETScSubMatrix::BlockPETScSubMatrix(
  Mat A,
  std::array<std::size_t, 2> block_indices,
  std::array<std::shared_ptr<const multiphenics::fem::BlockDofMap>, 2> block_dofmaps
) : _global_matrix(A)
{
  PetscErrorCode ierr;
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  ierr = PetscObjectGetComm((PetscObject) A, &mpi_comm);
  if (ierr != 0) petsc_error(ierr, __FILE__, "PetscObjectGetComm");
  
  // Extract sub matrix
  for (std::size_t i = 0; i < 2; ++i)
  {
    const auto & block_owned_dofs__local_numbering = block_dofmaps[i]->block_owned_dofs__local_numbering(block_indices[i]);
    ierr = ISCreateGeneral(mpi_comm, block_owned_dofs__local_numbering.size(), block_owned_dofs__local_numbering.data(),
                           PETSC_USE_POINTER, &_is[i]);
    if (ierr != 0) petsc_error(ierr, __FILE__, "ISCreateGeneral");
  }
  ierr = MatGetLocalSubMatrix(_global_matrix, _is[0], _is[1], &_A);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatGetLocalSubMatrix");
  
  // Initialization of local to global PETSc map.
  // Here "local" is intended with respect to the original space, while
  // "global" is intended with respect to original matrix _global_matrix
  // for entries in restriction, -1 for entries not in restriction.
  
  // Allocate data for local-to-global map in an STL vector
  std::array<std::vector<PetscInt>, 2> stl_local_to_global;
  for (std::size_t i = 0; i < 2; ++i)
  {
    const auto & original_to_block = block_dofmaps[i]->original_to_block(block_indices[i]);
    const auto block_index_map = block_dofmaps[i]->index_map();
    const auto original_index_map = block_dofmaps[i]->dofmaps()[block_indices[i]]->index_map();
    std::size_t unrestricted_ghosted_size = original_index_map->block_size()*(original_index_map->size_local() + original_index_map->num_ghosts());
    
    auto & map = stl_local_to_global[i];
    map.resize(unrestricted_ghosted_size);
    for (std::size_t original_index = 0; original_index < unrestricted_ghosted_size; original_index++)
    {
      if (original_to_block.count(original_index) > 0)
      {
        map[original_index] = block_index_map->local_to_global(original_to_block.at(original_index));
      }
      else
      {
        map[original_index] = -1; // entries corresponding to this index will be discarded
      }
    }
  }
  
  // Create PETSc local-to-global map as index set
  std::array<ISLocalToGlobalMapping, 2> petsc_local_to_global;
  for (std::size_t i = 0; i < 2; ++i)
  {
    ierr = ISLocalToGlobalMappingCreate(mpi_comm, 1, stl_local_to_global[i].size(), stl_local_to_global[i].data(),
                                        PETSC_COPY_VALUES, &petsc_local_to_global[i]);
    if (ierr != 0) petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");
  }
  
  // Set matrix local-to-global maps
  ierr = MatSetLocalToGlobalMapping(_A, petsc_local_to_global[0], petsc_local_to_global[1]);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetLocalToGlobalMapping");
  
  // Clean up local-to-global maps
  for (std::size_t i = 0; i < 2; ++i)
  {
    ierr = ISLocalToGlobalMappingDestroy(&petsc_local_to_global[i]);
    if (ierr != 0) petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");
  }
}
//-----------------------------------------------------------------------------
BlockPETScSubMatrix::~BlockPETScSubMatrix()
{
  PetscErrorCode ierr;
  
  // --- restore the global matrix --- //
  ierr = MatRestoreLocalSubMatrix(_global_matrix, _is[0], _is[1], &_A);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatRestoreLocalSubMatrix");
  ierr = ISDestroy(&_is[0]);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISDestroy");
  ierr = ISDestroy(&_is[1]);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISDestroy");
  // --- end --- restore the global matrix --- end --- //
}
//-----------------------------------------------------------------------------
Mat BlockPETScSubMatrix::mat() const
{
  return _A;
}
//-----------------------------------------------------------------------------

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

#include <dolfinx/la/utils.h>
#include <multiphenics/la/BlockPETScSubVectorWrapper.h>

using namespace multiphenics;
using namespace multiphenics::la;

using dolfinx::la::petsc_error;
using multiphenics::fem::BlockDofMap;

//-----------------------------------------------------------------------------
BlockPETScSubVectorWrapper::BlockPETScSubVectorWrapper(
  Vec x,
  std::size_t block_index,
  std::shared_ptr<const BlockDofMap> block_dofmap,
  InsertMode insert_mode
) : BlockPETScSubVectorReadWrapper(x, block_index, block_dofmap),
    _global_vector(x),
    _original_to_block(block_dofmap->original_to_block(block_index)),
    _sub_block_to_original(block_dofmap->sub_block_to_original(block_index)),
    _insert_mode(insert_mode)
{
  // Nothing to be done
}
//-----------------------------------------------------------------------------
BlockPETScSubVectorWrapper::~BlockPETScSubVectorWrapper()
{
  // Restrict values from content private attribute
  std::vector<PetscInt> restricted_rows(_sub_block_to_original.size());
  std::vector<PetscScalar> restricted_values(_sub_block_to_original.size());
  for (auto & sub_block_to_original_it: _sub_block_to_original)
  {
    auto sub_block_index = sub_block_to_original_it.first;
    auto original_index = sub_block_to_original_it.second;
    auto block_index = _original_to_block.at(original_index);
    restricted_rows[sub_block_index] = block_index;
    restricted_values[sub_block_index] = content[original_index];
  }

  // Insert values calling PETSc API
  PetscErrorCode ierr;
  Vec global_vector_local_form;
  PetscScalar* array_local_form;
  ierr = VecGhostGetLocalForm(_global_vector, &global_vector_local_form);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostGetLocalForm");
  ierr = VecGetArray(global_vector_local_form, &array_local_form);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetArray");
  for (std::size_t sub_block_index = 0; sub_block_index < restricted_rows.size(); ++sub_block_index)
  {
    array_local_form[restricted_rows[sub_block_index]] = restricted_values[sub_block_index];
  }
  ierr = VecRestoreArray(global_vector_local_form, &array_local_form);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecRestoreArray");
  ierr = VecGhostRestoreLocalForm(_global_vector, &global_vector_local_form);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostRestoreLocalForm");
}
//-----------------------------------------------------------------------------

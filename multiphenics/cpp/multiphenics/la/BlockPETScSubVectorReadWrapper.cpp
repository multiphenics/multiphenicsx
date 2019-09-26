// Copyright (C) 2016-2019 by the multiphenics authors
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

#include <dolfin/common/IndexMap.h>
#include <dolfin/la/utils.h>
#include <multiphenics/la/BlockPETScSubVectorReadWrapper.h>

using namespace multiphenics;
using namespace multiphenics::la;

using dolfin::la::petsc_error;
using multiphenics::fem::BlockDofMap;

//-----------------------------------------------------------------------------
BlockPETScSubVectorReadWrapper::BlockPETScSubVectorReadWrapper(
  Vec x,
  std::size_t block_index,
  std::shared_ptr<const BlockDofMap> block_dofmap
) : content(nullptr, 0)
{
  // Fetch vector content
  const auto & original_to_block(block_dofmap->original_to_block(block_index));
  const auto & sub_block_to_original(block_dofmap->sub_block_to_original(block_index));
  std::vector<PetscInt> original_indices(sub_block_to_original.size());
  std::vector<PetscInt> block_indices(original_to_block.size());
  for (auto & sub_block_to_original_it: sub_block_to_original)
  {
    auto sub_block_index = sub_block_to_original_it.first;
    auto original_index = sub_block_to_original_it.second;
    auto block_index = original_to_block.at(original_index);
    original_indices[sub_block_index] = original_index;
    block_indices[sub_block_index] = block_index;
  }
  std::vector<PetscScalar> block_content(block_indices.size());
  PetscErrorCode ierr;
  Vec x_local_form;
  ierr = VecGhostGetLocalForm(x, &x_local_form);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostGetLocalForm");
  ierr = VecGetValues(x_local_form, block_indices.size(), block_indices.data(), block_content.data());
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetValues");
  ierr = VecGhostRestoreLocalForm(x, &x_local_form);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostRestoreLocalForm");
  
  // Assign to an eigen matrix (indexed with respect to original indices)
  const auto original_index_map = block_dofmap->dofmaps()[block_index]->index_map;
  _content.resize(original_index_map->block_size*(original_index_map->size_local() + original_index_map->num_ghosts()), 0.);
  for (auto & sub_block_to_original_it: sub_block_to_original)
  {
    auto sub_block_index = sub_block_to_original_it.first;
    auto original_index = sub_block_to_original_it.second;
    _content[original_index] = block_content[sub_block_index];
  }
  new (&content) Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>(_content.data(), _content.size());
}
//-----------------------------------------------------------------------------
BlockPETScSubVectorReadWrapper::~BlockPETScSubVectorReadWrapper()
{
  // Nothing to be done
}
//-----------------------------------------------------------------------------

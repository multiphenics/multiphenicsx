// Copyright (C) 2016-2021 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <dolfinx/la/PETScVector.h>  // for dolfinx::la::petsc_error
#include <multiphenicsx/la/PETScVector.h>

using namespace dolfinx;
using dolfinx::la::petsc_error;
using multiphenicsx::la::VecSubVectorReadWrapper;
using multiphenicsx::la::VecSubVectorWrapper;

//-----------------------------------------------------------------------------
std::vector<IS> multiphenicsx::la::create_petsc_index_sets(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps,
    const std::vector<int> is_bs, bool ghosted, GhostBlockLayout ghost_block_layout)
{
  assert(maps.size() == is_bs.size());
  std::vector<std::int32_t> size_local(maps.size());
  std::vector<std::int32_t> size_ghost(maps.size());
  std::vector<int> bs(maps.size());
  std::generate(size_local.begin(), size_local.end(), [i = 0, maps] () mutable {
    return maps[i++].first.get().size_local();
  });
  std::generate(size_ghost.begin(), size_ghost.end(), [i = 0, maps] () mutable {
    return maps[i++].first.get().num_ghosts();
  });
  std::generate(bs.begin(), bs.end(), [i = 0, maps, is_bs] () mutable {
    auto bs_i = maps[i].second;
    auto is_bs_i = is_bs[i];
    i++;
    assert(is_bs_i == bs_i || is_bs_i == 1);
    if (is_bs_i == 1)
      return bs_i;
    else
      return 1;
  });

  // Initialize storage for indices
  std::vector<std::vector<PetscInt>> index(maps.size());
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    if (ghosted)
    {
      index[i].resize(bs[i] * (size_local[i] + size_ghost[i]));
    }
    else
    {
      index[i].resize(bs[i] * size_local[i]);
    }
  }

  // Compute indices and offset
  std::size_t offset = 0;
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    if (ghosted && ghost_block_layout == GhostBlockLayout::intertwined)
    {
      std::iota(index[i].begin(), std::next(index[i].begin(), bs[i] * (size_local[i] + size_ghost[i])), offset);
    }
    else
    {
      std::iota(index[i].begin(), std::next(index[i].begin(), bs[i] * size_local[i]), offset);
    }

    offset += bs[i] * size_local[i];
    if (ghost_block_layout == GhostBlockLayout::intertwined)
    {
      offset += bs[i] * size_ghost[i];
    }
  }
  if (ghosted && ghost_block_layout == GhostBlockLayout::trailing)
  {
    for (std::size_t i = 0; i < maps.size(); ++i)
    {
      std::iota(std::next(index[i].begin(), bs[i] * size_local[i]),
                std::next(index[i].begin(), bs[i] * (size_local[i] + size_ghost[i])), offset);

      offset += bs[i] * size_ghost[i];
    }
  }

  // Initialize PETSc IS objects
  std::vector<IS> is(maps.size());
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    ISCreateBlock(PETSC_COMM_SELF, is_bs[i], index[i].size(), index[i].data(),
                  PETSC_COPY_VALUES, &is[i]);
  }
  return is;
}
//-----------------------------------------------------------------------------
VecSubVectorReadWrapper::VecSubVectorReadWrapper(
  Vec x,
  IS index_set,
  bool ghosted
) : _ghosted(ghosted)
{
  PetscErrorCode ierr;

  // Get number of entries to extract from x
  PetscInt is_size;
  ierr = ISGetLocalSize(index_set, &is_size);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISGetLocalSize");

  // Get indices of entries to extract from x
  const PetscInt *indices;
  ierr = ISGetIndices(index_set, &indices);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISGetIndices");

  // Fetch vector content from x
  Vec x_local_form;
  if (_ghosted)
  {
    ierr = VecGhostGetLocalForm(x, &x_local_form);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostGetLocalForm");
  }
  else
  {
    x_local_form = x;
  }
  _content.resize(is_size, 0.);
  ierr = VecGetValues(x_local_form, is_size, indices, _content.data());
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetValues");
  if (_ghosted)
  {
    ierr = VecGhostRestoreLocalForm(x, &x_local_form);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostRestoreLocalForm");
  }

  // Restore indices
  ierr = ISRestoreIndices(index_set, &indices);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISRestoreIndices");
}
//-----------------------------------------------------------------------------
VecSubVectorReadWrapper::VecSubVectorReadWrapper(
  Vec x,
  IS unrestricted_index_set,
  IS restricted_index_set,
  const std::map<std::int32_t, std::int32_t>& unrestricted_to_restricted,
  int unrestricted_to_restricted_bs,
  bool ghosted)
  : _ghosted(ghosted)
{
  PetscErrorCode ierr;

  // Get number of entries to extract from x
  PetscInt restricted_is_size;
  ierr = ISGetLocalSize(restricted_index_set, &restricted_is_size);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISGetLocalSize");

  // Get indices of entries to extract from x
  const PetscInt *restricted_indices;
  ierr = ISGetIndices(restricted_index_set, &restricted_indices);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISGetIndices");

  // Fetch vector content from x
  Vec x_local_form;
  if (_ghosted)
  {
    ierr = VecGhostGetLocalForm(x, &x_local_form);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostGetLocalForm");
  }
  else
  {
    x_local_form = x;
  }
  std::vector<PetscScalar> restricted_content(restricted_is_size, 0.);
  ierr = VecGetValues(x_local_form, restricted_is_size, restricted_indices, restricted_content.data());
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetValues");
  if (_ghosted)
  {
    ierr = VecGhostRestoreLocalForm(x, &x_local_form);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostRestoreLocalForm");
  }

  // Restore indices
  ierr = ISRestoreIndices(restricted_index_set, &restricted_indices);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISRestoreIndices");

  // Get number of entries to be stored in _content
  PetscInt unrestricted_is_size;
  ierr = ISGetLocalSize(unrestricted_index_set, &unrestricted_is_size);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISGetLocalSize");

  // Assign vector content to an STL vector indexed with respect to the unrestricted index set
  _content.resize(unrestricted_is_size, 0.);
  for (PetscInt unrestricted_index = 0; unrestricted_index < unrestricted_is_size; unrestricted_index++)
  {
    if (unrestricted_to_restricted.count(unrestricted_index/unrestricted_to_restricted_bs) > 0)
    {
      _content[unrestricted_index] = restricted_content[
        unrestricted_to_restricted_bs*unrestricted_to_restricted.at(
          unrestricted_index/unrestricted_to_restricted_bs) +
            unrestricted_index%unrestricted_to_restricted_bs];
    }
  }
}
//-----------------------------------------------------------------------------
VecSubVectorReadWrapper::~VecSubVectorReadWrapper()
{
  // Nothing to be done
}
//-----------------------------------------------------------------------------
VecSubVectorWrapper::VecSubVectorWrapper(
  Vec x,
  IS index_set,
  bool ghosted)
  : VecSubVectorReadWrapper(x, index_set, ghosted),
    _global_vector(x), _is(index_set)
{
  PetscErrorCode ierr;

  // Get number of entries stored in _content
  PetscInt is_size;
  ierr = ISGetLocalSize(index_set, &is_size);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISGetLocalSize");

  // Fill in _restricted_to_unrestricted attribute with the identity map
  for (PetscInt index = 0; index < is_size; index++)
  {
    _restricted_to_unrestricted[index] = index;
  }
}
//-----------------------------------------------------------------------------
VecSubVectorWrapper::VecSubVectorWrapper(
  Vec x,
  IS unrestricted_index_set,
  IS restricted_index_set,
  const std::map<std::int32_t, std::int32_t>& unrestricted_to_restricted,
  int unrestricted_to_restricted_bs,
  bool ghosted)
  : VecSubVectorReadWrapper(x, unrestricted_index_set, restricted_index_set, unrestricted_to_restricted,
                            unrestricted_to_restricted_bs, ghosted),
    _global_vector(x), _is(restricted_index_set)
{
  PetscErrorCode ierr;

  // Get number of entries stored in _content
  PetscInt unrestricted_is_size;
  ierr = ISGetLocalSize(unrestricted_index_set, &unrestricted_is_size);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISGetLocalSize");

  // Fill in _restricted_to_unrestricted attribute
  for (PetscInt unrestricted_index = 0; unrestricted_index < unrestricted_is_size; unrestricted_index++)
  {
    if (unrestricted_to_restricted.count(unrestricted_index/unrestricted_to_restricted_bs) > 0)
    {
      _restricted_to_unrestricted[
        unrestricted_to_restricted_bs*unrestricted_to_restricted.at(
          unrestricted_index/unrestricted_to_restricted_bs) +
            unrestricted_index%unrestricted_to_restricted_bs] = unrestricted_index;
    }
  }
}
//-----------------------------------------------------------------------------
VecSubVectorWrapper::~VecSubVectorWrapper()
{
  // Sub vector should have been restored before destroying object
  assert(!_is);
  assert(_restricted_to_unrestricted.size() == 0);
  assert(_content.size() == 0);
}
//-----------------------------------------------------------------------------
void VecSubVectorWrapper::restore()
{
  PetscErrorCode ierr;

  // Get indices of entries to restore in x
  const PetscInt *restricted_indices;
  ierr = ISGetIndices(_is, &restricted_indices);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISGetIndices");

  // Restrict values from content attribute
  std::vector<PetscScalar> restricted_values(_restricted_to_unrestricted.size());
  for (auto& restricted_to_unrestricted_it: _restricted_to_unrestricted)
  {
    auto restricted_index = restricted_to_unrestricted_it.first;
    auto unrestricted_index = restricted_to_unrestricted_it.second;
    restricted_values[restricted_index] = _content[unrestricted_index];
  }

  // Insert values calling PETSc API
  Vec global_vector_local_form;
  if (_ghosted)
  {
    ierr = VecGhostGetLocalForm(_global_vector, &global_vector_local_form);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostGetLocalForm");
  }
  else
  {
    global_vector_local_form = _global_vector;
  }
  PetscScalar* array_local_form;
  ierr = VecGetArray(global_vector_local_form, &array_local_form);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetArray");
  for (std::size_t i = 0; i < restricted_values.size(); ++i)
    array_local_form[restricted_indices[i]] = restricted_values[i];
  ierr = VecRestoreArray(global_vector_local_form, &array_local_form);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecRestoreArray");
  if (_ghosted)
  {
    ierr = VecGhostRestoreLocalForm(_global_vector, &global_vector_local_form);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostRestoreLocalForm");
  }

  // Restore indices
  ierr = ISRestoreIndices(_is, &restricted_indices);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISRestoreIndices");

  // Clear storage
  _is = nullptr;
  _restricted_to_unrestricted.clear();
  _content.clear();
}
//-----------------------------------------------------------------------------

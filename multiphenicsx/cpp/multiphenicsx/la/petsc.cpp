// Copyright (C) 2016-2026 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <cassert>
#include <dolfinx/common/petsc.h>
#include <multiphenicsx/la/petsc.h>
#include <vector>

using namespace dolfinx;
using multiphenicsx::la::petsc::MatSubMatrixWrapper;
using multiphenicsx::la::petsc::VecSubVectorReadWrapper;
using multiphenicsx::la::petsc::VecSubVectorWrapper;

//-----------------------------------------------------------------------------
std::vector<IS> multiphenicsx::la::petsc::create_index_sets(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps,
    const std::vector<int> is_bs, bool ghosted,
    GhostBlockLayout ghost_block_layout)
{
  assert(maps.size() == is_bs.size());
  std::vector<std::int32_t> size_local(maps.size());
  std::vector<std::int32_t> size_ghost(maps.size());
  std::vector<int> bs(maps.size());
  std::generate(size_local.begin(), size_local.end(), [i = 0, maps]() mutable
                { return maps[i++].first.get().size_local(); });
  std::generate(size_ghost.begin(), size_ghost.end(), [i = 0, maps]() mutable
                { return maps[i++].first.get().num_ghosts(); });
  std::generate(bs.begin(), bs.end(),
                [i = 0, maps, is_bs]() mutable
                {
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
      std::iota(
          index[i].begin(),
          std::next(index[i].begin(), bs[i] * (size_local[i] + size_ghost[i])),
          offset);
    }
    else
    {
      std::iota(index[i].begin(),
                std::next(index[i].begin(), bs[i] * size_local[i]), offset);
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
      std::iota(
          std::next(index[i].begin(), bs[i] * size_local[i]),
          std::next(index[i].begin(), bs[i] * (size_local[i] + size_ghost[i])),
          offset);

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
MatSubMatrixWrapper::MatSubMatrixWrapper(Mat A, std::array<IS, 2> index_sets)
    : _global_matrix(A), _is(index_sets)
{
  // Get communicator from matrix object
  MPI_Comm comm = MPI_COMM_NULL;
  common::petsc::check(PetscObjectGetComm((PetscObject)A, &comm),
                       "PetscObjectGetComm");

  // Sub matrix inherits block size of the index sets. Check that they
  // are consistent with the ones of the global matrix.
  std::vector<PetscInt> bs_A(2);
  common::petsc::check(MatGetBlockSizes(A, &bs_A[0], &bs_A[1]),
                       "MatGetBlockSizes");
  std::vector<PetscInt> bs_is(2);
  common::petsc::check(ISGetBlockSize(_is[0], &bs_is[0]), "ISGetBlockSize");
  common::petsc::check(ISGetBlockSize(_is[1], &bs_is[1]), "ISGetBlockSize");
  assert(bs_A[0] == bs_is[0]);
  assert(bs_A[1] == bs_is[1]);

  // Extract sub matrix
  common::petsc::check(MatGetLocalSubMatrix(A, _is[0], _is[1], &_sub_matrix),
                       "MatGetLocalSubMatrix");
}
//-----------------------------------------------------------------------------
MatSubMatrixWrapper::MatSubMatrixWrapper(
    Mat A, std::array<IS, 2> unrestricted_index_sets,
    std::array<IS, 2> restricted_index_sets,
    std::array<std::unordered_map<std::int32_t, std::int32_t>, 2>
        unrestricted_to_restricted,
    std::array<int, 2> unrestricted_to_restricted_bs)
    : MatSubMatrixWrapper(A, restricted_index_sets)
{
  // Initialization of custom local to global PETSc map.
  // In order not to change the assembly routines, here "local" is intended
  // with respect to the *unrestricted* index sets (which where generated using
  // the index map that will be passed to the assembly routines). Instead,
  // "global" is intended with respect to the *restricted* index sets for
  // entries in the restriction, while it is set to -1 (i.e., values
  // corresponding to those indices will be discarded) for entries not in the
  // restriction.

  // Get sub matrix (i.e., index sets) block sizes
  std::vector<PetscInt> bs(2);
  common::petsc::check(MatGetBlockSizes(_sub_matrix, &bs[0], &bs[1]),
                       "MatGetBlockSizes");

  // Compare sub matrix block sizes with unrestricted_to_restricted_bs:
  // they should either be the same (typically the case of restricted matrices
  // or restricted nest matrices) or unrestricted_to_restricted_bs may be larger
  // than the sub matrix block sizes (typically the case of restricted block
  // matrices, because bs is forced to one).
  assert(bs[0] == unrestricted_to_restricted_bs[0]
         || (bs[0] == 1 && unrestricted_to_restricted_bs[0] > 1));
  assert(bs[1] == unrestricted_to_restricted_bs[1]
         || (bs[1] == 1 && unrestricted_to_restricted_bs[1] > 1));
  std::vector<PetscInt> unrestricted_to_restricted_correction(2);
  for (std::size_t i = 0; i < 2; ++i)
  {
    if (bs[i] == unrestricted_to_restricted_bs[i])
    {
      unrestricted_to_restricted_correction[i] = 1;
    }
    else
    {
      assert(bs[i] == 1);
      unrestricted_to_restricted_correction[i]
          = unrestricted_to_restricted_bs[i];
    }
  }

  // Get matrix local-to-global map
  std::array<ISLocalToGlobalMapping, 2> petsc_local_to_global_matrix;
  common::petsc::check(
      MatGetLocalToGlobalMapping(A, &petsc_local_to_global_matrix[0],
                                 &petsc_local_to_global_matrix[1]),
      "MatGetLocalToGlobalMapping");

  // Allocate data for submatrix local-to-global maps in an STL vector
  std::array<std::vector<PetscInt>, 2> stl_local_to_global_submatrix;
  for (std::size_t i = 0; i < 2; ++i)
  {
    PetscInt unrestricted_is_size;
    common::petsc::check(
        ISBlockGetLocalSize(unrestricted_index_sets[i], &unrestricted_is_size),
        "ISBlockGetLocalSize");
    stl_local_to_global_submatrix[i].resize(unrestricted_is_size);

    const PetscInt* restricted_indices;
    common::petsc::check(
        ISBlockGetIndices(restricted_index_sets[i], &restricted_indices),
        "ISBlockGetIndices");

    std::vector<PetscInt> restricted_local_index(1);
    std::vector<PetscInt> restricted_global_index(1);

    for (PetscInt unrestricted_index = 0;
         unrestricted_index < unrestricted_is_size; unrestricted_index++)
    {
      if (unrestricted_to_restricted[i].count(
              unrestricted_index / unrestricted_to_restricted_correction[i])
          > 0)
      {
        restricted_local_index[0] = restricted_indices
            [unrestricted_to_restricted_correction[i]
                 * unrestricted_to_restricted[i].at(
                     unrestricted_index
                     / unrestricted_to_restricted_correction[i])
             + unrestricted_index % unrestricted_to_restricted_correction[i]];
        ISLocalToGlobalMappingApplyBlock(
            petsc_local_to_global_matrix[i], restricted_local_index.size(),
            restricted_local_index.data(), restricted_global_index.data());
        stl_local_to_global_submatrix[i][unrestricted_index]
            = restricted_global_index[0];
      }
      else
      {
        stl_local_to_global_submatrix[i][unrestricted_index] = -1;
      }
    }

    common::petsc::check(
        ISBlockRestoreIndices(restricted_index_sets[i], &restricted_indices),
        "ISBlockRestoreIndices");
  }

  // Get communicator from submatrix object
  MPI_Comm comm = MPI_COMM_NULL;
  common::petsc::check(PetscObjectGetComm((PetscObject)_sub_matrix, &comm),
                       "PetscObjectGetComm");

  // Create submatrix local-to-global maps as index set
  std::array<ISLocalToGlobalMapping, 2> petsc_local_to_global_submatrix;
  for (std::size_t i = 0; i < 2; ++i)
  {
    common::petsc::check(
        ISLocalToGlobalMappingCreate(
            comm, bs[i], stl_local_to_global_submatrix[i].size(),
            stl_local_to_global_submatrix[i].data(), PETSC_COPY_VALUES,
            &petsc_local_to_global_submatrix[i]),
        "ISLocalToGlobalMappingCreate");
  }

  // Set submatrix local-to-global maps
  common::petsc::check(MatSetLocalToGlobalMapping(
                           _sub_matrix, petsc_local_to_global_submatrix[0],
                           petsc_local_to_global_submatrix[1]),
                       "MatSetLocalToGlobalMapping");

  // Clean up submatrix local-to-global maps
  for (std::size_t i = 0; i < 2; ++i)
  {
    common::petsc::check(
        ISLocalToGlobalMappingDestroy(&petsc_local_to_global_submatrix[i]),
        "ISLocalToGlobalMappingDestroy");
  }
}
//-----------------------------------------------------------------------------
MatSubMatrixWrapper::~MatSubMatrixWrapper()
{
  // Sub matrix should have been restored before destroying object
  assert(!_sub_matrix);
  assert(!_is[0]);
  assert(!_is[1]);
}
//-----------------------------------------------------------------------------
void MatSubMatrixWrapper::restore()
{
  // Restore the global matrix
  assert(_sub_matrix);
  common::petsc::check(
      MatRestoreLocalSubMatrix(_global_matrix, _is[0], _is[1], &_sub_matrix),
      "MatRestoreLocalSubMatrix");

  // Clear pointers
  _sub_matrix = nullptr;
  _is.fill(nullptr);
}
//-----------------------------------------------------------------------------
Mat MatSubMatrixWrapper::mat() const { return _sub_matrix; }
//-----------------------------------------------------------------------------
VecSubVectorReadWrapper::VecSubVectorReadWrapper(Vec x, IS index_set,
                                                 bool ghosted)
    : _ghosted(ghosted)
{
  // Get number of entries to extract from x
  PetscInt is_size;
  common::petsc::check(ISGetLocalSize(index_set, &is_size), "ISGetLocalSize");

  // Get indices of entries to extract from x
  const PetscInt* indices;
  common::petsc::check(ISGetIndices(index_set, &indices), "ISGetIndices");

  // Fetch vector content from x
  Vec x_local_form;
  if (_ghosted)
  {
    common::petsc::check(VecGhostGetLocalForm(x, &x_local_form),
                         "VecGhostGetLocalForm");
  }
  else
  {
    x_local_form = x;
  }
  _content.resize(is_size, 0.);
  common::petsc::check(
      VecGetValues(x_local_form, is_size, indices, _content.data()),
      "VecGetValues");
  if (_ghosted)
  {
    common::petsc::check(VecGhostRestoreLocalForm(x, &x_local_form),
                         "VecGhostRestoreLocalForm");
  }

  // Restore indices
  common::petsc::check(ISRestoreIndices(index_set, &indices),
                       "ISRestoreIndices");
}
//-----------------------------------------------------------------------------
VecSubVectorReadWrapper::VecSubVectorReadWrapper(
    Vec x, IS unrestricted_index_set, IS restricted_index_set,
    const std::unordered_map<std::int32_t, std::int32_t>&
        unrestricted_to_restricted,
    int unrestricted_to_restricted_bs, bool ghosted)
    : _ghosted(ghosted)
{
  // Get number of entries to extract from x
  PetscInt restricted_is_size;
  common::petsc::check(
      ISGetLocalSize(restricted_index_set, &restricted_is_size),
      "ISGetLocalSize");

  // Get indices of entries to extract from x
  const PetscInt* restricted_indices;
  common::petsc::check(ISGetIndices(restricted_index_set, &restricted_indices),
                       "ISGetIndices");

  // Fetch vector content from x
  Vec x_local_form;
  if (_ghosted)
  {
    common::petsc::check(VecGhostGetLocalForm(x, &x_local_form),
                         "VecGhostGetLocalForm");
  }
  else
  {
    x_local_form = x;
  }
  std::vector<PetscScalar> restricted_content(restricted_is_size, 0.);
  common::petsc::check(VecGetValues(x_local_form, restricted_is_size,
                                    restricted_indices,
                                    restricted_content.data()),
                       "VecGetValues");
  if (_ghosted)
  {
    common::petsc::check(VecGhostRestoreLocalForm(x, &x_local_form),
                         "VecGhostRestoreLocalForm");
  }

  // Restore indices
  common::petsc::check(
      ISRestoreIndices(restricted_index_set, &restricted_indices),
      "ISRestoreIndices");

  // Get number of entries to be stored in _content
  PetscInt unrestricted_is_size;
  common::petsc::check(
      ISGetLocalSize(unrestricted_index_set, &unrestricted_is_size),
      "ISGetLocalSize");

  // Assign vector content to an STL vector indexed with respect to the
  // unrestricted index set
  _content.resize(unrestricted_is_size, 0.);
  for (PetscInt unrestricted_index = 0;
       unrestricted_index < unrestricted_is_size; unrestricted_index++)
  {
    if (unrestricted_to_restricted.count(unrestricted_index
                                         / unrestricted_to_restricted_bs)
        > 0)
    {
      _content[unrestricted_index] = restricted_content
          [unrestricted_to_restricted_bs
               * unrestricted_to_restricted.at(unrestricted_index
                                               / unrestricted_to_restricted_bs)
           + unrestricted_index % unrestricted_to_restricted_bs];
    }
  }
}
//-----------------------------------------------------------------------------
VecSubVectorReadWrapper::~VecSubVectorReadWrapper()
{
  // Nothing to be done
}
//-----------------------------------------------------------------------------
VecSubVectorWrapper::VecSubVectorWrapper(Vec x, IS index_set, bool ghosted)
    : VecSubVectorReadWrapper(x, index_set, ghosted), _global_vector(x),
      _is(index_set)
{
  // Get number of entries stored in _content
  PetscInt is_size;
  common::petsc::check(ISGetLocalSize(index_set, &is_size), "ISGetLocalSize");

  // Fill in _restricted_to_unrestricted attribute with the identity map
  for (PetscInt index = 0; index < is_size; index++)
  {
    _restricted_to_unrestricted[index] = index;
  }
}
//-----------------------------------------------------------------------------
VecSubVectorWrapper::VecSubVectorWrapper(
    Vec x, IS unrestricted_index_set, IS restricted_index_set,
    const std::unordered_map<std::int32_t, std::int32_t>&
        unrestricted_to_restricted,
    int unrestricted_to_restricted_bs, bool ghosted)
    : VecSubVectorReadWrapper(x, unrestricted_index_set, restricted_index_set,
                              unrestricted_to_restricted,
                              unrestricted_to_restricted_bs, ghosted),
      _global_vector(x), _is(restricted_index_set)
{
  // Get number of entries stored in _content
  PetscInt unrestricted_is_size;
  common::petsc::check(
      ISGetLocalSize(unrestricted_index_set, &unrestricted_is_size),
      "ISGetLocalSize");

  // Fill in _restricted_to_unrestricted attribute
  for (PetscInt unrestricted_index = 0;
       unrestricted_index < unrestricted_is_size; unrestricted_index++)
  {
    if (unrestricted_to_restricted.count(unrestricted_index
                                         / unrestricted_to_restricted_bs)
        > 0)
    {
      _restricted_to_unrestricted
          [unrestricted_to_restricted_bs
               * unrestricted_to_restricted.at(unrestricted_index
                                               / unrestricted_to_restricted_bs)
           + unrestricted_index % unrestricted_to_restricted_bs]
          = unrestricted_index;
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
  // Get indices of entries to restore in x
  const PetscInt* restricted_indices;
  common::petsc::check(ISGetIndices(_is, &restricted_indices), "ISGetIndices");

  // Restrict values from content attribute
  std::vector<PetscScalar> restricted_values(
      _restricted_to_unrestricted.size());
  for (auto& restricted_to_unrestricted_it : _restricted_to_unrestricted)
  {
    auto restricted_index = restricted_to_unrestricted_it.first;
    auto unrestricted_index = restricted_to_unrestricted_it.second;
    restricted_values[restricted_index] = _content[unrestricted_index];
  }

  // Insert values calling PETSc API
  Vec global_vector_local_form;
  if (_ghosted)
  {
    common::petsc::check(
        VecGhostGetLocalForm(_global_vector, &global_vector_local_form),
        "VecGhostGetLocalForm");
  }
  else
  {
    global_vector_local_form = _global_vector;
  }
  PetscScalar* array_local_form;
  common::petsc::check(VecGetArray(global_vector_local_form, &array_local_form),
                       "VecGetArray");
  for (std::size_t i = 0; i < restricted_values.size(); ++i)
    array_local_form[restricted_indices[i]] = restricted_values[i];
  common::petsc::check(
      VecRestoreArray(global_vector_local_form, &array_local_form),
      "VecRestoreArray");
  if (_ghosted)
  {
    common::petsc::check(
        VecGhostRestoreLocalForm(_global_vector, &global_vector_local_form),
        "VecGhostRestoreLocalForm");
  }

  // Restore indices
  common::petsc::check(ISRestoreIndices(_is, &restricted_indices),
                       "ISRestoreIndices");

  // Clear storage
  _is = nullptr;
  _restricted_to_unrestricted.clear();
  _content.clear();
}
//-----------------------------------------------------------------------------

// Copyright (C) 2016-2021 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <map>
#include <vector>
#include <petscvec.h>
#include <dolfinx/common/IndexMap.h>

namespace multiphenicsx
{

namespace la
{
/// Ghost block layout types
enum class GhostBlockLayout
{
  intertwined,  // [owned_0, ghost_0, owned_1, ghost_1, ..., ...], as used by block matrices
  trailing      // [owned_0, owned_1, ..., ghost_0, ghost_1, ...], as used by block vectors
};

/// @todo This function could take just the local sizes
///
/// Compute PETSc IndexSets (IS) for a stack of index maps. E.g., if
/// `map[0] = {0, 1, 2, 3, 4, 5, 6}` and `map[1] = {0, 1, 2, 4}` (in
/// local indices) then `IS[0] = {0, 1, 2, 3, 4, 5, 6}` and `IS[1] = {7, 8,
/// 9, 10}`.
///
/// The caller is responsible for destruction of each IS.
///
/// @param[in] maps Vector of IndexMaps and corresponding block sizes
/// @param[in] is_bs Requested block sizes for the output vector of PETSc Index Sets
/// @param[in] ghosted Include ghost indices in the computed IS only,
///                    if provided the bool true as input.
/// @param[in] ghost_block_layout Ghost block layout type. IS used for block matrices
///                               should provide GhostBlockLayout::intertwined.
///                               IS used for block vectors should provide
///                               GhostBlockLayout::trailing.
/// @returns Vector of PETSc Index Sets, created on` PETSC_COMM_SELF`
std::vector<IS> create_petsc_index_sets(
    const std::vector<
        std::pair<std::reference_wrapper<const dolfinx::common::IndexMap>, int>>& maps,
    const std::vector<int> is_bs, bool ghosted = true,
    GhostBlockLayout ghost_block_layout = GhostBlockLayout::intertwined);

/// Read-only wrapper around a local subvector of a Vec object, used in combination with DofMapRestriction
class VecSubVectorReadWrapper
{
public:
  /// Constructor (for cases without restriction)
  VecSubVectorReadWrapper(Vec x,
                          IS index_set,
                          bool ghosted = true),

  /// Constructor (for cases with restriction)
  VecSubVectorReadWrapper(Vec x,
                          IS unrestricted_index_set,
                          IS restricted_index_set,
                          const std::map<std::int32_t, std::int32_t>& unrestricted_to_restricted,
                          int unrestricted_to_restricted_bs,
                          bool ghosted = true);

  /// Destructor
  ~VecSubVectorReadWrapper();

  /// Copy constructor (deleted)
  VecSubVectorReadWrapper(const VecSubVectorReadWrapper&) = delete;

  /// Move constructor (deleted)
  VecSubVectorReadWrapper(VecSubVectorReadWrapper&&) = delete;

  // Assignment operator (deleted)
  VecSubVectorReadWrapper& operator=(const VecSubVectorReadWrapper&) = delete;

  /// Move assignment operator (deleted)
  VecSubVectorReadWrapper& operator=(VecSubVectorReadWrapper&&) = delete;

  /// Get content
  std::vector<PetscScalar>& mutable_content() { return _content; }

protected:
  std::vector<PetscScalar> _content;
  bool _ghosted;
};

/// Wrapper around a local subvector of a Vec object, used in combination with DofMapRestriction
class VecSubVectorWrapper: public VecSubVectorReadWrapper
{
public:
  /// Constructor (for cases without restriction)
  VecSubVectorWrapper(Vec x,
                      IS index_set,
                      bool ghosted = true),

  /// Constructor (for cases with restriction)
  VecSubVectorWrapper(Vec x,
                      IS unrestricted_index_set,
                      IS restricted_index_set,
                      const std::map<std::int32_t, std::int32_t>& unrestricted_to_restricted,
                      int unrestricted_to_restricted_bs,
                      bool ghosted = true);

  /// Destructor
  ~VecSubVectorWrapper();

  /// Copy constructor (deleted)
  VecSubVectorWrapper(const VecSubVectorWrapper&) = delete;

  /// Move constructor (deleted)
  VecSubVectorWrapper(VecSubVectorWrapper&&) = delete;

  // Assignment operator (deleted)
  VecSubVectorWrapper& operator=(const VecSubVectorWrapper&) = delete;

  /// Move assignment operator (deleted)
  VecSubVectorWrapper& operator=(VecSubVectorWrapper&&) = delete;

  /// Restore PETSc Vec object
  void restore();

private:
  Vec _global_vector;
  IS _is;
  std::map<std::int32_t, std::int32_t> _restricted_to_unrestricted;
};
} // namespace la
} // namespace multiphenicsx

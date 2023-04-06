// Copyright (C) 2016-2023 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <memory>
#include <petscmat.h>
#include <petscvec.h>
#include <vector>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/la/SparsityPattern.h>
#include <multiphenicsx/fem/utils.h>

namespace multiphenicsx
{

namespace fem
{

/// Helper functions for assembly into PETSc data structures
namespace petsc
{

/// Create a matrix
/// @param[in] a A bilinear form
/// @param[in] index_maps A pair of index maps. Row index map is given by index_maps[0], column index map is given
/// by index_maps[1].
/// @param[in] index_maps_bs A pair of int, representing the block size of index_maps.
/// @param[in] dofmaps A pair of AdjacencyList containing the dofmaps. Row dofmap is given by dofmaps[0], while
/// column dofmap is given by dofmaps[1].
/// @param[in] matrix_type The PETSc matrix type to create
/// @return A sparse matrix with a layout and sparsity that matches the
/// bilinear form. The caller is responsible for destroying the Mat
/// object.
template <std::floating_point T>
Mat create_matrix(
  const dolfinx::fem::Form<PetscScalar, T>& a,
  std::array<std::reference_wrapper<const dolfinx::common::IndexMap>, 2> index_maps,
  const std::array<int, 2> index_maps_bs,
  std::array<const dolfinx::graph::AdjacencyList<std::int32_t>*, 2> dofmaps,
  const std::string& matrix_type = std::string())
{
  dolfinx::la::SparsityPattern pattern = multiphenicsx::fem::create_sparsity_pattern(
    a, index_maps, index_maps_bs, dofmaps);
  pattern.assemble();
  return dolfinx::la::petsc::create_matrix(a.mesh()->comm(), pattern, matrix_type);
}

/// Initialise a monolithic matrix for an array of bilinear forms
/// @param[in] a Rectangular array of bilinear forms. The `a(i, j)` form
/// will correspond to the `(i, j)` block in the returned matrix
/// @param[in] index_maps A pair of vectors of index maps. Index maps for block (i, j) will be
/// constructed from (index_maps[0][i], index_maps[1][j]).
/// @param[in] index_maps_bs A pair of vectors of int, representing the block size of the
/// corresponding entry in index_maps.
/// @param[in] dofmaps A pair of vectors of AdjacencyList containing the list dofmaps for each block.
/// The dofmap pair for block (i, j) will be constructed from (dofmaps[0][i], dofmaps[1][j]).
/// @param[in] matrix_type The type of PETSc Mat. If empty the PETSc default is
/// used.
/// @return A sparse matrix  with a layout and sparsity that matches the
/// bilinear forms. The caller is responsible for destroying the Mat
/// object.
template <std::floating_point T>
Mat create_matrix_block(
  const std::vector<std::vector<const dolfinx::fem::Form<PetscScalar, T>*>>& a,
  std::array<std::vector<std::reference_wrapper<const dolfinx::common::IndexMap>>, 2> index_maps,
  const std::array<std::vector<int>, 2> index_maps_bs,
  const std::array<std::vector<const dolfinx::graph::AdjacencyList<std::int32_t>*>, 2>& dofmaps,
  const std::string& matrix_type = std::string())
{
  std::size_t rows = index_maps[0].size();
  assert(index_maps_bs[0].size() == rows);
  assert(dofmaps[0].size() == rows);
  std::size_t cols = index_maps[1].size();
  assert(index_maps_bs[1].size() == cols);
  assert(dofmaps[1].size() == cols);

  // Build sparsity pattern for each block
  std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh;
  std::vector<std::vector<std::unique_ptr<dolfinx::la::SparsityPattern>>> patterns(
      rows);
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < cols; ++col)
    {
      if (const dolfinx::fem::Form<PetscScalar, T>* form = a[row][col]; form)
      {
        patterns[row].push_back(std::make_unique<la::SparsityPattern>(
            multiphenicsx::fem::create_sparsity_pattern(
              *form, {{index_maps[0][row], index_maps[1][col]}}, {{index_maps_bs[0][row], index_maps_bs[1][col]}},
              {{dofmaps[0][row], dofmaps[1][col]}})));
        if (!mesh)
          mesh = form->mesh();
      }
      else
        patterns[row].push_back(nullptr);
    }
  }

  if (!mesh)
    throw std::runtime_error("Could not find a Mesh.");

  // Compute offsets for the fields
  std::array<std::vector<std::pair<
                 std::reference_wrapper<const dolfinx::common::IndexMap>, int>>,
             2>
      maps_and_bs;
  for (std::size_t d = 0; d < 2; ++d)
  {
    for (std::size_t f = 0; f < index_maps[d].size(); ++f)
    {
      maps_and_bs[d].emplace_back(
        index_maps[d][f], index_maps_bs[d][f]);
    }
  }

  // Create merged sparsity pattern
  std::vector<std::vector<const dolfinx::la::SparsityPattern*>> p(rows);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t col = 0; col < cols; ++col)
      p[row].push_back(patterns[row][col].get());

  dolfinx::la::SparsityPattern pattern(mesh->comm(), p, maps_and_bs, index_maps_bs);
  pattern.assemble();

  // FIXME: Add option to pass customised local-to-global map to PETSc
  // Mat constructor

  // Initialise matrix
  Mat A = dolfinx::la::petsc::create_matrix(mesh->comm(), pattern, matrix_type);

  // Create row and column local-to-global maps (field0, field1, field2,
  // etc), i.e. ghosts of field0 appear before owned indices of field1
  std::array<std::vector<PetscInt>, 2> _maps;
  for (int d = 0; d < 2; ++d)
  {
    // FIXME: Index map concatenation has already been computed inside
    // the SparsityPattern constructor, but we also need it here to
    // build the PETSc local-to-global map. Compute outside and pass
    // into SparsityPattern constructor.

    // FIXME: avoid concatenating the same maps twice in case that V[0]
    // == V[1].

    // Concatenate the block index map in the row and column directions
    auto [rank_offset, local_offset, ghosts, _]
        = dolfinx::common::stack_index_maps(maps_and_bs[d]);
    for (std::size_t f = 0; f < index_maps[d].size(); ++f)
    {
      const dolfinx::common::IndexMap& map = index_maps[d][f].get();
      const int bs = index_maps_bs[d][f];
      const std::int32_t size_local = bs * map.size_local();
      const std::vector global = map.global_indices();
      for (std::int32_t i = 0; i < size_local; ++i)
        _maps[d].push_back(i + rank_offset + local_offset[f]);
      for (std::size_t i = size_local; i < bs * global.size(); ++i)
        _maps[d].push_back(ghosts[f][i - size_local]);
    }
  }

  // Create PETSc local-to-global map/index sets and attach to matrix
  ISLocalToGlobalMapping petsc_local_to_global0;
  ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 1, _maps[0].size(),
                               _maps[0].data(), PETSC_COPY_VALUES,
                               &petsc_local_to_global0);
  if (dofmaps[0] == dofmaps[1])
  {
    MatSetLocalToGlobalMapping(A, petsc_local_to_global0,
                               petsc_local_to_global0);
    ISLocalToGlobalMappingDestroy(&petsc_local_to_global0);
  }
  else
  {
    ISLocalToGlobalMapping petsc_local_to_global1;
    ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 1, _maps[1].size(),
                                 _maps[1].data(), PETSC_COPY_VALUES,
                                 &petsc_local_to_global1);
    MatSetLocalToGlobalMapping(A, petsc_local_to_global0,
                               petsc_local_to_global1);
    ISLocalToGlobalMappingDestroy(&petsc_local_to_global0);
    ISLocalToGlobalMappingDestroy(&petsc_local_to_global1);
  }

  return A;
}

/// @brief Create nested (MatNest) matrix.
///
/// @note The caller is responsible for destroying the Mat object.
template <std::floating_point T>
Mat create_matrix_nest(
  const std::vector<std::vector<const dolfinx::fem::Form<PetscScalar, T>*>>& a,
  std::array<std::vector<std::reference_wrapper<const dolfinx::common::IndexMap>>, 2> index_maps,
  const std::array<std::vector<int>, 2> index_maps_bs,
  const std::array<std::vector<const dolfinx::graph::AdjacencyList<std::int32_t>*>, 2>& dofmaps,
  const std::vector<std::vector<std::string>>& matrix_types)
{
  std::size_t rows = index_maps[0].size();
  assert(index_maps_bs[0].size() == rows);
  assert(dofmaps[0].size() == rows);
  std::size_t cols = index_maps[1].size();
  assert(index_maps_bs[1].size() == cols);
  assert(dofmaps[1].size() == cols);
  std::vector<std::vector<std::string>> _matrix_types(
      rows, std::vector<std::string>(cols));
  if (!matrix_types.empty())
    _matrix_types = matrix_types;

  // Loop over each form and create matrix
  std::vector<Mat> mats(rows * cols, nullptr);
  std::shared_ptr<const mesh::Mesh<T>> mesh;
  for (std::size_t i = 0; i < rows; ++i)
  {
    for (std::size_t j = 0; j < cols; ++j)
    {
      if (const dolfinx::fem::Form<PetscScalar, T>* form = a[i][j]; form)
      {
        mats[i * cols + j] = multiphenicsx::fem::petsc::create_matrix(
          *form, {{index_maps[0][i], index_maps[1][j]}}, {{index_maps_bs[0][i], index_maps_bs[1][j]}},
          {{dofmaps[0][i], dofmaps[1][j]}}, _matrix_types[i][j]);
        mesh = form->mesh();
      }
    }
  }

  if (!mesh)
    throw std::runtime_error("Could not find a Mesh.");

  // Initialise block (MatNest) matrix
  Mat A;
  MatCreate(mesh->comm(), &A);
  MatSetType(A, MATNEST);
  MatNestSetSubMats(A, rows, nullptr, cols, nullptr, mats.data());
  MatSetUp(A);

  // De-reference Mat objects
  for (std::size_t i = 0; i < mats.size(); ++i)
  {
    if (mats[i])
      MatDestroy(&mats[i]);
  }

  return A;
}

} // namespace petsc
} // namespace fem
} // namespace multiphenicsx

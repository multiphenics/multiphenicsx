// Copyright (C) 2016-2023 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <memory>
#include <petscmat.h>
#include <petscvec.h>
#include <set>
#include <vector>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <multiphenicsx/fem/sparsitybuild.h>

namespace multiphenicsx
{

namespace fem
{

using dolfinx::fem::IntegralType;

/// Helper functions for assembly into PETSc data structures
namespace petsc
{

/// Create a matrix.
/// @param[in] mesh The mesh
/// @param[in] index_maps A pair of index maps. Row index map is given by index_maps[0], column index map is given
///                       by index_maps[1].
/// @param[in] index_maps_bs A pair of int, representing the block size of index_maps.
/// @param[in] integral_types Required integral types
/// @param[in] dofmaps A pair of AdjacencyList containing the dofmaps. Row dofmap is given by dofmaps[0], while
///                    column dofmap is given by dofmaps[1].
/// @param[in] matrix_type The PETSc matrix type to create
/// @return A sparse matrix with a layout and sparsity that matches the one required by the provided index maps
/// integral types and dofmaps. The caller is responsible for destroying the Mat object.
template <std::floating_point T>
Mat create_matrix(
    const dolfinx::mesh::Mesh<T>& mesh,
    std::array<std::reference_wrapper<const dolfinx::common::IndexMap>, 2> index_maps,
    const std::array<int, 2> index_maps_bs,
    const std::set<dolfinx::fem::IntegralType>& integral_types,
    std::array<const dolfinx::graph::AdjacencyList<std::int32_t>*, 2> dofmaps,
    const std::string& matrix_type = std::string())
{
  const int tdim = mesh.topology()->dim();

  // Build sparsity pattern
  const std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> index_maps_shared_ptr
    {{std::shared_ptr<const dolfinx::common::IndexMap>(&index_maps[0].get(), [](const dolfinx::common::IndexMap*){}),
      std::shared_ptr<const dolfinx::common::IndexMap>(&index_maps[1].get(), [](const dolfinx::common::IndexMap*){})}};
  dolfinx::la::SparsityPattern pattern(mesh.comm(), index_maps_shared_ptr, index_maps_bs);
  for (auto integral_type : integral_types)
  {
    switch (integral_type)
    {
    case IntegralType::cell:
      sparsitybuild::cells(pattern, *mesh.topology(), dofmaps);
      break;
    case IntegralType::interior_facet:
      mesh.topology_mutable()->create_entities(tdim - 1);
      mesh.topology_mutable()->create_connectivity(tdim - 1, tdim);
      sparsitybuild::interior_facets(pattern, *mesh.topology(), dofmaps);
      break;
    case IntegralType::exterior_facet:
      mesh.topology_mutable()->create_entities(tdim - 1);
      mesh.topology_mutable()->create_connectivity(tdim - 1, tdim);
      sparsitybuild::exterior_facets(pattern, *mesh.topology(), dofmaps);
      break;
    default:
      throw std::runtime_error("Unsupported integral type");
    }
  }

  // Finalise communication
  pattern.assemble();

  return dolfinx::la::petsc::create_matrix(mesh.comm(), pattern, matrix_type);
}

/// Initialise monolithic matrix for an array for bilinear forms. Matrix
/// is not zeroed.
/// @param[in] mesh The mesh
/// @param[in] index_maps A pair of vectors of index maps. Index maps for block (i, j) will be
///                       constructed from (index_maps[0][i], index_maps[1][j]).
/// @param[in] index_maps_bs A pair of vectors of int, representing the block size of the
///                          corresponding entry in index_maps.
/// @param[in] integral_types A matrix of required integral types. Required integral types
///                                 for block (i, j) are deduced from integral_types[i, j].
/// @param[in] dofmaps A pair of vectors of AdjacencyList containing the list dofmaps for each block.
///                    The dofmap pair for block (i, j) will be constructed from
///                    (dofmaps[0][i], dofmaps[1][j]).
/// @param[in] matrix_type The type of PETSc Mat. If empty the PETSc default is
///                 used.
/// @return A sparse matrix with a layout and sparsity that matches the one required by the provided index maps
/// integral types and dofmaps. The caller is responsible for destroying the Mat object.
template <std::floating_point T>
Mat create_matrix_block(
    const dolfinx::mesh::Mesh<T>& mesh,
    std::array<std::vector<std::reference_wrapper<const dolfinx::common::IndexMap>>, 2> index_maps,
    const std::array<std::vector<int>, 2> index_maps_bs,
    const std::vector<std::vector<std::set<dolfinx::fem::IntegralType>>>& integral_types,
    const std::array<std::vector<const dolfinx::graph::AdjacencyList<std::int32_t>*>, 2>& dofmaps,
    const std::string& matrix_type = std::string())
{
  std::size_t rows = index_maps[0].size();
  assert(index_maps_bs[0].size() == rows);
  assert(integral_types.size() == rows);
  assert(dofmaps[0].size() == rows);
  std::size_t cols = index_maps[1].size();
  assert(index_maps_bs[1].size() == cols);
  assert(std::all_of(integral_types.begin(), integral_types.end(),
    [&cols](const std::vector<std::set<fem::IntegralType>>& integral_types_){
    return integral_types_.size() == cols;}));
  assert(index_maps_bs[1].size() == cols);
  assert(dofmaps[1].size() == cols);

  // Build sparsity pattern for each block
  std::vector<std::vector<std::unique_ptr<dolfinx::la::SparsityPattern>>> patterns(
      rows);
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < cols; ++col)
    {
      int at_least_one_integral_owned = (integral_types[row][col].size() > 0);
      int at_least_one_integral_global = 0;
      MPI_Allreduce(&at_least_one_integral_owned, &at_least_one_integral_global, 1, MPI_INT, MPI_SUM, mesh.comm());
      if (at_least_one_integral_global > 0)
      {
        // Create sparsity pattern for block
        const std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> index_maps_row_col
          {{std::shared_ptr<const dolfinx::common::IndexMap>(
                &index_maps[0][row].get(), [](const dolfinx::common::IndexMap*){}),
            std::shared_ptr<const dolfinx::common::IndexMap>(
                &index_maps[1][col].get(), [](const dolfinx::common::IndexMap*){})}};
        const std::array<int, 2> index_maps_bs_row_col
          {{index_maps_bs[0][row], index_maps_bs[1][col]}};
        patterns[row].push_back(
            std::make_unique<dolfinx::la::SparsityPattern>(mesh.comm(), index_maps_row_col, index_maps_bs_row_col));
        assert(patterns[row].back());

        auto& sp = patterns[row].back();
        assert(sp);

        // Build sparsity pattern for block
        if (integral_types[row][col].count(IntegralType::cell) > 0)
        {
          sparsitybuild::cells(*sp, *mesh.topology(),
                               {{dofmaps[0][row], dofmaps[1][col]}});
        }
        if (integral_types[row][col].count(IntegralType::interior_facet) > 0
            or integral_types[row][col].count(IntegralType::exterior_facet) > 0)
        {
          const int tdim = mesh.topology()->dim();
          mesh.topology_mutable()->create_entities(tdim - 1);
          mesh.topology_mutable()->create_connectivity(tdim - 1, tdim);
          if (integral_types[row][col].count(IntegralType::interior_facet) > 0)
          {
            sparsitybuild::interior_facets(*sp, *mesh.topology(),
                                           {{dofmaps[0][row], dofmaps[1][col]}});
          }
          if (integral_types[row][col].count(IntegralType::exterior_facet) > 0)
          {
            sparsitybuild::exterior_facets(*sp, *mesh.topology(),
                                           {{dofmaps[0][row], dofmaps[1][col]}});
          }
        }
      }
      else
        patterns[row].push_back(nullptr);
    }
  }

  // Compute offsets for the fields
  std::array<std::vector<std::pair<
                 std::reference_wrapper<const dolfinx::common::IndexMap>, int>>,
             2>
      maps_and_bs;
  for (std::size_t d = 0; d < 2; ++d)
  {
    for (std::size_t f = 0; f < index_maps[d].size(); ++f)
    {
      maps_and_bs[d].push_back(
          {index_maps[d][f], index_maps_bs[d][f]});
    }
  }

  // Create merged sparsity pattern
  std::vector<std::vector<const dolfinx::la::SparsityPattern*>> p(rows);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t col = 0; col < cols; ++col)
      p[row].push_back(patterns[row][col].get());
  dolfinx::la::SparsityPattern pattern(mesh.comm(), p, maps_and_bs, index_maps_bs);
  pattern.assemble();

  // FIXME: Add option to pass customised local-to-global map to PETSc
  // Mat constructor

  // Initialise matrix
  Mat A = dolfinx::la::petsc::create_matrix(mesh.comm(), pattern, matrix_type);

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
      const std::vector<std::int64_t> global = map.global_indices();
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

/// Create nested (MatNest) matrix. Matrix is not zeroed.
/// @param[in] mesh The mesh
/// @param[in] index_maps A pair of vectors of index maps. Index maps for block (i, j) will be
///                       constructed from (index_maps[0][i], index_maps[1][j]).
/// @param[in] index_maps_bs A pair of vectors of int, representing the block size of the
///                          corresponding entry in index_maps.
/// @param[in] integral_types A matrix of required integral types. Required integral types
///                           for block (i, j) are deduced from integral_types[i, j].
/// @param[in] dofmaps A pair of vectors of AdjacencyList containing the list dofmaps for each block.
///                    The dofmap pair for block (i, j) will be constructed from
///                    (dofmaps[0][i], dofmaps[1][j]).
/// @param[in] matrix_types A matrix of PETSc Mat types.
/// @return A sparse matrix with a layout and sparsity that matches the one required by the provided index maps
/// integral types and dofmaps. The caller is responsible for destroying the Mat object.
template <std::floating_point T>
Mat create_matrix_nest(
    const dolfinx::mesh::Mesh<T>& mesh,
    std::array<std::vector<std::reference_wrapper<const dolfinx::common::IndexMap>>, 2> index_maps,
    const std::array<std::vector<int>, 2> index_maps_bs,
    const std::vector<std::vector<std::set<dolfinx::fem::IntegralType>>>& integral_types,
    const std::array<std::vector<const dolfinx::graph::AdjacencyList<std::int32_t>*>, 2>& dofmaps,
    const std::vector<std::vector<std::string>>& matrix_types)
{
  std::size_t rows = index_maps[0].size();
  assert(index_maps_bs[0].size() == rows);
  assert(integral_types.size() == rows);
  assert(dofmaps[0].size() == rows);
  std::size_t cols = index_maps[1].size();
  assert(index_maps_bs[1].size() == cols);
  assert(std::all_of(integral_types.begin(), integral_types.end(),
    [&cols](const std::vector<std::set<fem::IntegralType>>& integral_types_){
    return integral_types_.size() == cols;}));
  assert(dofmaps[1].size() == cols);
  std::vector<std::vector<std::string>> _matrix_types(
      rows, std::vector<std::string>(cols));
  if (!matrix_types.empty())
    _matrix_types = matrix_types;

  // Loop over each form and create matrix
  std::vector<Mat> mats(rows * cols, nullptr);
  for (std::size_t i = 0; i < rows; ++i)
  {
    for (std::size_t j = 0; j < cols; ++j)
    {
      if (integral_types[i][j].size() > 0)
      {
        mats[i * cols + j] = multiphenicsx::fem::petsc::create_matrix(
          mesh, {{index_maps[0][i], index_maps[1][j]}},
          {{index_maps_bs[0][i], index_maps_bs[1][j]}},
          integral_types[i][j],
          {{dofmaps[0][i], dofmaps[1][j]}},
          _matrix_types[i][j]);
      }
    }
  }

  // Initialise block (MatNest) matrix
  Mat A;
  MatCreate(mesh.comm(), &A);
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

// Copyright (C) 2016-2022 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/petsc.h>
#include <multiphenicsx/fem/petsc.h>
#include <multiphenicsx/fem/sparsitybuild.h>

using namespace dolfinx;
using dolfinx::fem::IntegralType;
namespace sparsitybuild = multiphenicsx::fem::sparsitybuild;

//-----------------------------------------------------------------------------
Mat multiphenicsx::fem::petsc::create_matrix(
    const mesh::Mesh& mesh,
    std::array<std::reference_wrapper<const common::IndexMap>, 2> index_maps,
    const std::array<int, 2> index_maps_bs,
    const std::set<fem::IntegralType>& integral_types,
    std::array<const graph::AdjacencyList<std::int32_t>*, 2> dofmaps,
    const std::string& matrix_type)
{
  // Build sparsity pattern
  const std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps_shared_ptr
    {{std::shared_ptr<const common::IndexMap>(&index_maps[0].get(), [](const common::IndexMap*){}),
      std::shared_ptr<const common::IndexMap>(&index_maps[1].get(), [](const common::IndexMap*){})}};
  la::SparsityPattern pattern(mesh.comm(), index_maps_shared_ptr, index_maps_bs);
  if (integral_types.count(fem::IntegralType::cell) > 0)
  {
    sparsitybuild::cells(pattern, mesh.topology(), dofmaps);
  }
  if (integral_types.count(fem::IntegralType::interior_facet) > 0
      or integral_types.count(fem::IntegralType::exterior_facet) > 0)
  {
    // FIXME: cleanup these calls? Some of the happen internally again.
    const int tdim = mesh.topology().dim();
    mesh.topology_mutable().create_entities(tdim - 1);
    mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
    if (integral_types.count(fem::IntegralType::interior_facet) > 0)
    {
      sparsitybuild::interior_facets(pattern, mesh.topology(), dofmaps);
    }
    if (integral_types.count(fem::IntegralType::exterior_facet) > 0)
    {
      sparsitybuild::exterior_facets(pattern, mesh.topology(), dofmaps);
    }
  }

  // Finalise communication
  pattern.assemble();

  return la::petsc::create_matrix(mesh.comm(), pattern, matrix_type);
}
//-----------------------------------------------------------------------------
Mat multiphenicsx::fem::petsc::create_matrix_block(
    const mesh::Mesh& mesh,
    std::array<std::vector<std::reference_wrapper<const common::IndexMap>>, 2> index_maps,
    const std::array<std::vector<int>, 2> index_maps_bs,
    const std::vector<std::vector<std::set<fem::IntegralType>>>& integral_types,
    const std::array<std::vector<const graph::AdjacencyList<std::int32_t>*>, 2>& dofmaps,
    const std::string& matrix_type)
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
  std::vector<std::vector<std::unique_ptr<la::SparsityPattern>>> patterns(
      rows);
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < cols; ++col)
    {
      if (integral_types[row][col].size() > 0)
      {
        // Create sparsity pattern for block
        const std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps_row_col
          {{std::shared_ptr<const common::IndexMap>(&index_maps[0][row].get(), [](const common::IndexMap*){}),
            std::shared_ptr<const common::IndexMap>(&index_maps[1][col].get(), [](const common::IndexMap*){})}};
        const std::array<int, 2> index_maps_bs_row_col
          {{index_maps_bs[0][row], index_maps_bs[1][col]}};
        patterns[row].push_back(
            std::make_unique<la::SparsityPattern>(mesh.comm(), index_maps_row_col, index_maps_bs_row_col));
        assert(patterns[row].back());

        auto& sp = patterns[row].back();
        assert(sp);

        // Build sparsity pattern for block
        if (integral_types[row][col].count(IntegralType::cell) > 0)
        {
          sparsitybuild::cells(*sp, mesh.topology(),
                               {{dofmaps[0][row], dofmaps[1][col]}});
        }
        if (integral_types[row][col].count(IntegralType::interior_facet) > 0
            or integral_types[row][col].count(IntegralType::exterior_facet) > 0)
        {
          // FIXME: cleanup these calls? Some of the happen internally again.
          const int tdim = mesh.topology().dim();
          mesh.topology_mutable().create_entities(tdim - 1);
          mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
          if (integral_types[row][col].count(IntegralType::interior_facet) > 0)
          {
            sparsitybuild::interior_facets(*sp, mesh.topology(),
                                           {{dofmaps[0][row], dofmaps[1][col]}});
          }
          if (integral_types[row][col].count(IntegralType::exterior_facet) > 0)
          {
            sparsitybuild::exterior_facets(*sp, mesh.topology(),
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
                 std::reference_wrapper<const common::IndexMap>, int>>,
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
  std::vector<std::vector<const la::SparsityPattern*>> p(rows);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t col = 0; col < cols; ++col)
      p[row].push_back(patterns[row][col].get());
  la::SparsityPattern pattern(mesh.comm(), p, maps_and_bs, index_maps_bs);
  pattern.assemble();

  // FIXME: Add option to pass customised local-to-global map to PETSc
  // Mat constructor

  // Initialise matrix
  Mat A = la::petsc::create_matrix(mesh.comm(), pattern, matrix_type);

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
        = common::stack_index_maps(maps_and_bs[d]);
    for (std::size_t f = 0; f < index_maps[d].size(); ++f)
    {
      const common::IndexMap& map = index_maps[d][f].get();
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
//-----------------------------------------------------------------------------
Mat multiphenicsx::fem::petsc::create_matrix_nest(
    const mesh::Mesh& mesh,
    std::array<std::vector<std::reference_wrapper<const common::IndexMap>>, 2> index_maps,
    const std::array<std::vector<int>, 2> index_maps_bs,
    const std::vector<std::vector<std::set<fem::IntegralType>>>& integral_types,
    const std::array<std::vector<const graph::AdjacencyList<std::int32_t>*>, 2>& dofmaps,
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
//-----------------------------------------------------------------------------

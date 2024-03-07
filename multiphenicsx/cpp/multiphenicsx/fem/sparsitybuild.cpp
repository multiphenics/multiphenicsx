// Copyright (C) 2016-2024 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Topology.h>
#include <multiphenicsx/fem/sparsitybuild.h>

using namespace dolfinx;
namespace sparsitybuild = multiphenicsx::fem::sparsitybuild;

//-----------------------------------------------------------------------------
void sparsitybuild::cells(
    la::SparsityPattern& pattern, std::span<const std::int32_t> cells,
    std::array<std::span<const std::int32_t>, 2> dofmaps_list,
    std::array<std::span<const std::size_t>, 2> dofmaps_bounds)
{
  for (auto c : cells)
  {
    auto cell_dofs_0
        = std::span(dofmaps_list[0].data() + dofmaps_bounds[0][c],
                    dofmaps_bounds[0][c + 1] - dofmaps_bounds[0][c]);
    auto cell_dofs_1
        = std::span(dofmaps_list[1].data() + dofmaps_bounds[1][c],
                    dofmaps_bounds[1][c + 1] - dofmaps_bounds[1][c]);
    pattern.insert(cell_dofs_0, cell_dofs_1);
  }
}
//-----------------------------------------------------------------------------
void sparsitybuild::interior_facets(
    la::SparsityPattern& pattern, std::span<const std::int32_t> facets,
    std::array<std::span<const std::int32_t>, 2> dofmaps_list,
    std::array<std::span<const std::size_t>, 2> dofmaps_bounds)
{
  std::array<std::vector<std::int32_t>, 2> macro_dofs;
  for (std::size_t index = 0; index < facets.size(); index += 2)
  {
    int cell_0 = facets[index];
    int cell_1 = facets[index + 1];
    for (std::size_t i = 0; i < 2; ++i)
    {
      auto cell_dofs_0 = std::span(
          dofmaps_list[i].data() + dofmaps_bounds[i][cell_0],
          dofmaps_bounds[i][cell_0 + 1] - dofmaps_bounds[i][cell_0]);
      auto cell_dofs_1 = std::span(
          dofmaps_list[i].data() + dofmaps_bounds[i][cell_1],
          dofmaps_bounds[i][cell_1 + 1] - dofmaps_bounds[i][cell_1]);
      macro_dofs[i].resize(cell_dofs_0.size() + cell_dofs_1.size());
      std::copy(cell_dofs_0.begin(), cell_dofs_0.end(), macro_dofs[i].begin());
      std::copy(cell_dofs_1.begin(), cell_dofs_1.end(),
                std::next(macro_dofs[i].begin(), cell_dofs_0.size()));
    }
    pattern.insert(macro_dofs[0], macro_dofs[1]);
  }
}

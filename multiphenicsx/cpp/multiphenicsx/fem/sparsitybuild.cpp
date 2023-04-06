// Copyright (C) 2016-2023 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Topology.h>
#include <multiphenicsx/fem/sparsitybuild.h>

using namespace dolfinx;
namespace sparsitybuild = multiphenicsx::fem::sparsitybuild;

//-----------------------------------------------------------------------------
void sparsitybuild::cells(
    la::SparsityPattern& pattern, std::span<const std::int32_t> cells,
    std::array<const graph::AdjacencyList<std::int32_t>*, 2> dofmaps)
{
  for (auto c : cells)
    pattern.insert(dofmaps[0]->links(c), dofmaps[1]->links(c));
}
//-----------------------------------------------------------------------------
void sparsitybuild::interior_facets(
    la::SparsityPattern& pattern, std::span<const std::int32_t> facets,
    std::array<const graph::AdjacencyList<std::int32_t>*, 2> dofmaps)
{
  std::array<std::vector<std::int32_t>, 2> macro_dofs;
  for (std::size_t index = 0; index < facets.size(); index += 2)
  {
    int cell_0 = facets[index];
    int cell_1 = facets[index + 1];
    for (std::size_t i = 0; i < 2; ++i)
    {
      auto cell_dofs_0 = dofmaps[i]->links(cell_0);
      auto cell_dofs_1 = dofmaps[i]->links(cell_1);
      macro_dofs[i].resize(cell_dofs_0.size() + cell_dofs_1.size());
      std::copy(cell_dofs_0.begin(), cell_dofs_0.end(), macro_dofs[i].begin());
      std::copy(cell_dofs_1.begin(), cell_dofs_1.end(),
                std::next(macro_dofs[i].begin(), cell_dofs_0.size()));
    }
    pattern.insert(macro_dofs[0], macro_dofs[1]);
  }
}

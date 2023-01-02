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
    la::SparsityPattern& pattern, const mesh::Topology& topology,
    std::array<const graph::AdjacencyList<std::int32_t>*, 2> dofmaps)
{
  const int D = topology.dim();
  auto cells = topology.connectivity(D, 0);
  assert(cells);
  for (int c = 0; c < cells->num_nodes(); ++c)
  {
    const auto cell_dofs0 = dofmaps[0]->links(c);
    const auto cell_dofs1 = dofmaps[1]->links(c);
    if (cell_dofs0.size() > 0 && cell_dofs1.size() > 0)
      pattern.insert(cell_dofs0, cell_dofs1);
  }
}
//-----------------------------------------------------------------------------
void sparsitybuild::interior_facets(
    la::SparsityPattern& pattern, const mesh::Topology& topology,
    std::array<const graph::AdjacencyList<std::int32_t>*, 2> dofmaps)
{
  const int D = topology.dim();
  if (!topology.connectivity(D - 1, 0))
    throw std::runtime_error("Topology facets have not been created.");

  auto connectivity = topology.connectivity(D - 1, D);
  if (!connectivity)
    throw std::runtime_error("Facet-cell connectivity has not been computed.");

  // Array to store macro-dofs, if required (for interior facets)
  std::array<std::vector<std::int32_t>, 2> macro_dofs;

  // Loop over owned facets
  auto map = topology.index_map(D - 1);
  assert(map);
  const std::int32_t num_facets = map->size_local();
  for (int f = 0; f < num_facets; ++f)
  {
    // Get cells incident with facet
    auto cells = connectivity->links(f);
    // Proceed to next facet if only ony connection
    if (cells.size() == 1)
      continue;

    assert(cells.size() == 2);
    const int cell0 = cells[0];
    const int cell1 = cells[1];

    // Skip facets associated to cells with no links
    bool need_insert = true;
    for (std::size_t i = 0; i < 2; i++)
    {
      const auto cell_dofs0 = dofmaps[i]->links(cell0);
      const auto cell_dofs1 = dofmaps[i]->links(cell1);
      if (cell_dofs0.size() == 0 && cell_dofs1.size() == 0)
      {
        need_insert = false;
        break;
      }
    }
    if (!need_insert)
      continue;

    // Tabulate dofs for each dimension on macro element
    for (std::size_t i = 0; i < 2; i++)
    {
      const auto cell_dofs0 = dofmaps[i]->links(cell0);
      const auto cell_dofs1 = dofmaps[i]->links(cell1);
      macro_dofs[i].resize(cell_dofs0.size() + cell_dofs1.size());
      std::copy(cell_dofs0.begin(), cell_dofs0.end(), macro_dofs[i].begin());
      std::copy(cell_dofs1.begin(), cell_dofs1.end(),
                std::next(macro_dofs[i].begin(), cell_dofs0.size()));
    }

    // Insert into sparsity pattern
    pattern.insert(macro_dofs[0], macro_dofs[1]);
  }
}
//-----------------------------------------------------------------------------
void sparsitybuild::exterior_facets(
    la::SparsityPattern& pattern, const mesh::Topology& topology,
    std::array<const graph::AdjacencyList<std::int32_t>*, 2> dofmaps)
{
  const int D = topology.dim();
  if (!topology.connectivity(D - 1, 0))
    throw std::runtime_error("Topology facets have not been created.");

  auto connectivity = topology.connectivity(D - 1, D);
  if (!connectivity)
    throw std::runtime_error("Facet-cell connectivity has not been computed.");

  // Loop over owned facets
  auto map = topology.index_map(D - 1);
  assert(map);
  const std::int32_t num_facets = map->size_local();
  for (int f = 0; f < num_facets; ++f)
  {
    // Proceed to next facet if we have an interior facet
    if (connectivity->num_links(f) == 2)
      continue;

    auto cells = connectivity->links(f);
    assert(cells.size() == 1);
    const auto c = cells[0];
    const auto cell_dofs0 = dofmaps[0]->links(c);
    const auto cell_dofs1 = dofmaps[1]->links(c);
    if (cell_dofs0.size() > 0 && cell_dofs1.size() > 0)
      pattern.insert(cell_dofs0, cell_dofs1);
  }
}
//-----------------------------------------------------------------------------

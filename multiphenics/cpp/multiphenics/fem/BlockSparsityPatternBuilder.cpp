// Copyright (C) 2016-2020 by the multiphenics authors
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

#include <dolfinx/mesh/MeshIterator.h>
#include <multiphenics/fem/BlockSparsityPatternBuilder.h>

using namespace multiphenics;
using namespace multiphenics::fem;

using dolfinx::la::SparsityPattern;
using dolfinx::mesh::Topology;

//-----------------------------------------------------------------------------
void BlockSparsityPatternBuilder::cells(
    SparsityPattern& pattern, const Topology& topology,
    const std::array<const BlockDofMap*, 2> block_dofmaps)
{
  assert(block_dofmaps[0]);
  assert(block_dofmaps[1]);
  const int D = topology.dim();
  auto cells = topology.connectivity(D, 0);
  assert(cells);
  for (int c = 0; c < cells->num_nodes(); ++c)
    pattern.insert_local(block_dofmaps[0]->cell_dofs(c), block_dofmaps[1]->cell_dofs(c));
}
//-----------------------------------------------------------------------------
void BlockSparsityPatternBuilder::interior_facets(
    SparsityPattern& pattern, const Topology& topology,
    const std::array<const BlockDofMap*, 2> block_dofmaps)
{
  assert(block_dofmaps[0]);
  assert(block_dofmaps[1]);

  const int D = topology.dim();
  if (!topology.connectivity(D - 1, 0))
    throw std::runtime_error("Topology facets have not been created.");
    
  auto connectivity = topology.connectivity(D - 1, D);
  if (!connectivity)
    throw std::runtime_error("Facet-cell connectivity has not been computed.");

  // Array to store macro-dofs, if required (for interior facets)
  std::array<Eigen::Array<PetscInt, Eigen::Dynamic, 1>, 2> macro_dofs;
  for (int f = 0; f < connectivity->num_nodes(); ++f)
  {
    // Continue if facet is exterior facet
    if (topology.size_global({D - 1, D}, f) == 1)
      continue;

    // FIXME: sort out ghosting

    // Get cells incident with facet
    auto cells = connectivity->links(f);
    assert(cells.rows() == 2);
    const int cell0 = cells[0];
    const int cell1 = cells[1];

    // Tabulate dofs for each dimension on macro element
    for (std::size_t i = 0; i < 2; i++)
    {
      auto cell_dofs0 = block_dofmaps[i]->cell_dofs(cell0);
      auto cell_dofs1 = block_dofmaps[i]->cell_dofs(cell1);
      macro_dofs[i].resize(cell_dofs0.size() + cell_dofs1.size());
      std::copy(cell_dofs0.data(), cell_dofs0.data() + cell_dofs0.size(),
                macro_dofs[i].data());
      std::copy(cell_dofs1.data(), cell_dofs1.data() + cell_dofs1.size(),
                macro_dofs[i].data() + cell_dofs0.size());
    }

    pattern.insert_local(macro_dofs[0], macro_dofs[1]);
  }
}
//-----------------------------------------------------------------------------
void BlockSparsityPatternBuilder::exterior_facets(
    SparsityPattern& pattern, const Topology& topology,
    const std::array<const BlockDofMap*, 2> block_dofmaps)
{
  const int D = topology.dim();
  if (!topology.connectivity(D - 1, 0))
    throw std::runtime_error("Topology facets have not been created.");
    
  auto connectivity = topology.connectivity(D - 1, D);
  if (!connectivity)
    throw std::runtime_error("Facet-cell connectivity has not been computed.");
    
  for (int f = 0; f < connectivity->num_nodes(); ++f)
  {
    // Skip interior facets
    if (topology.size_global({D - 1, D}, f) > 1)
      continue;

    // FIXME: sort out ghosting

    assert(connectivity->num_links(f) == 1);
    auto cells = connectivity->links(f);
    const int cell = cells[0];

    pattern.insert_local(block_dofmaps[0]->cell_dofs(cell),
                         block_dofmaps[1]->cell_dofs(cell));
  }
}
//-----------------------------------------------------------------------------

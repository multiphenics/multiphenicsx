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

#include <dolfin/mesh/MeshIterator.h>
#include <multiphenics/fem/BlockSparsityPatternBuilder.h>

using namespace multiphenics;
using namespace multiphenics::fem;

using dolfin::la::SparsityPattern;
using dolfin::mesh::Connectivity;
using dolfin::mesh::Mesh;
using dolfin::mesh::MeshEntity;
using dolfin::mesh::MeshRange;

//-----------------------------------------------------------------------------
void BlockSparsityPatternBuilder::cells(
    SparsityPattern& pattern, const Mesh& mesh,
    const std::array<const BlockDofMap*, 2> block_dofmaps)
{
  assert(block_dofmaps[0]);
  assert(block_dofmaps[1]);
  const int D = mesh.topology().dim();
  for (auto& cell : MeshRange(mesh, D))
  {
    pattern.insert_local(block_dofmaps[0]->cell_dofs(cell.index()),
                         block_dofmaps[1]->cell_dofs(cell.index()));
  }
}
//-----------------------------------------------------------------------------
void BlockSparsityPatternBuilder::interior_facets(
    SparsityPattern& pattern, const Mesh& mesh,
    const std::array<const BlockDofMap*, 2> block_dofmaps)
{
  assert(block_dofmaps[0]);
  assert(block_dofmaps[1]);

  const std::size_t D = mesh.topology().dim();
  mesh.create_entities(D - 1);
  mesh.create_connectivity(D - 1, D);

  // Array to store macro-dofs, if required (for interior facets)
  std::array<Eigen::Array<PetscInt, Eigen::Dynamic, 1>, 2> macro_dofs;
  std::shared_ptr<const Connectivity> connectivity
      = mesh.topology().connectivity(D - 1, D);
  if (!connectivity)
    throw std::runtime_error("Facet-cell connectivity has not been computed.");

  for (auto& facet : MeshRange(mesh, D - 1))
  {
    // Continue if facet is exterior facet
    if (connectivity->size_global(facet.index()) == 1)
      continue;

    // FIXME: sort out ghosting

    // Get cells incident with facet
    assert(connectivity->size(facet.index()) == 2);
    const MeshEntity cell0(mesh, D, facet.entities(D)[0]);
    const MeshEntity cell1(mesh, D, facet.entities(D)[1]);

    // Tabulate dofs for each dimension on macro element
    for (std::size_t i = 0; i < 2; i++)
    {
      const auto cell_dofs0 = block_dofmaps[i]->cell_dofs(cell0.index());
      const auto cell_dofs1 = block_dofmaps[i]->cell_dofs(cell1.index());
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
    SparsityPattern& pattern, const Mesh& mesh,
    const std::array<const BlockDofMap*, 2> block_dofmaps)
{
  const std::size_t D = mesh.topology().dim();
  mesh.create_entities(D - 1);
  mesh.create_connectivity(D - 1, D);

  std::shared_ptr<const Connectivity> connectivity
      = mesh.topology().connectivity(D - 1, D);
  if (!connectivity)
    throw std::runtime_error("Facet-cell connectivity has not been computed.");
    
  for (auto& facet : MeshRange(mesh, D - 1))
  {
    // Skip interior facets
    if (connectivity->size_global(facet.index()) > 1)
      continue;

    // FIXME: sort out ghosting

    assert(connectivity->size(facet.index()) == 1);
    MeshEntity cell(mesh, D, facet.entities(D)[0]);
    pattern.insert_local(block_dofmaps[0]->cell_dofs(cell.index()),
                         block_dofmaps[1]->cell_dofs(cell.index()));
  }
}
//-----------------------------------------------------------------------------

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

#ifndef __BLOCK_SPARSITY_PATTERN_BUILDER_H
#define __BLOCK_SPARSITY_PATTERN_BUILDER_H

#include <dolfin/la/SparsityPattern.h>
#include <dolfin/mesh/Mesh.h>
#include <multiphenics/fem/BlockDofMap.h>

namespace multiphenics
{
  namespace fem
  {
    /// This class provides functions to compute the sparsity pattern
    /// based on DOF maps. This class is a copy of the corresponding DOLFIN one,
    /// which unfortunately we cannot reuse because DofMap::cell_dofs is not virtual and thus,
    /// even if BlockDofMap were to inherit from DofMap, the overridden implementation of cell_dofs
    /// would not be called.
    /// TODO Check again if future interface changes to DofMap allow to remove this copy.
    class BlockSparsityPatternBuilder
    {
    public:
      /// Iterate over cells and insert entries into sparsity pattern
      static void cells(dolfin::la::SparsityPattern& pattern, const dolfin::mesh::Mesh& mesh,
                        const std::array<const BlockDofMap*, 2> block_dofmaps);

      /// Iterate over interior facets and insert entries into sparsity pattern
      static void interior_facets(dolfin::la::SparsityPattern& pattern,
                                  const dolfin::mesh::Mesh& mesh,
                                  const std::array<const BlockDofMap*, 2> block_dofmaps);

      /// Iterate over exterior facets and insert entries into sparsity pattern
      static void exterior_facets(dolfin::la::SparsityPattern& pattern,
                                  const dolfin::mesh::Mesh& mesh,
                                  const std::array<const BlockDofMap*, 2> block_dofmaps);
    };
  }
}

#endif

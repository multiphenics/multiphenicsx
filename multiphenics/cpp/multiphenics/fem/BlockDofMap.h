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

#ifndef __BLOCK_DOF_MAP_H
#define __BLOCK_DOF_MAP_H

#include <dolfinx/fem/DofMap.h>

namespace multiphenics
{
  namespace fem
  {
    /// This class handles the mapping of degrees of freedom for block
    /// function spaces, also considering possible restrictions to
    /// subdomains

    class BlockDofMap
    {
    public:

      /// Constructor
      BlockDofMap(std::vector<std::shared_ptr<const dolfinx::fem::DofMap>> dofmaps,
                  std::vector<Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>> restrictions);

    protected:

      /// Helper functions for constructor
      void _map_owned_dofs(std::vector<std::shared_ptr<const dolfinx::fem::DofMap>> dofmaps,
                           std::vector<Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>> restrictions);
      void _map_ghost_dofs(std::vector<std::shared_ptr<const dolfinx::fem::DofMap>> dofmaps,
                           std::vector<Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>> restrictions);
      void _precompute_cell_dofs(std::vector<std::shared_ptr<const dolfinx::fem::DofMap>> dofmaps);
      void _precompute_views(std::vector<std::shared_ptr<const dolfinx::fem::DofMap>> dofmaps);

    public:

      /// Copy constructor
      BlockDofMap(const BlockDofMap& dofmap) = delete;

      /// Create a view for a component, *considering* restrictions. This is supposed to be used only in this class,
      /// as precomputed views are already available through the view() method.
      BlockDofMap(const BlockDofMap& block_dofmap, std::size_t i);

      /// Move constructor
      BlockDofMap(BlockDofMap&& dofmap) = default;

      /// Destructor
      virtual ~BlockDofMap() = default;

      /// Copy assignment
      BlockDofMap& operator=(const BlockDofMap& dofmap) = delete;

      /// Move assignment
      BlockDofMap& operator=(BlockDofMap&& dofmap) = default;

      /// Returns a view of the i-th block
      const BlockDofMap & view(std::size_t b) const;

      /// Local-to-global mapping of dofs on a cell
      /// @param[in] cell_index The cell index.
      /// @return  Local-global map for cell (used process-local global
      /// index)
      Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
      cell_dofs(int cell_index) const;

      // Constructor arguments
      std::vector<std::shared_ptr<const dolfinx::fem::DofMap>> dofmaps() const;
      std::vector<Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>> restrictions() const;

      // Index Map from local to global, with offsets for preceding blocks
      std::shared_ptr<dolfinx::common::IndexMap> index_map;

      // Index Maps from local to global, without offsets from preceding blocks
      std::vector<std::shared_ptr<dolfinx::common::IndexMap>> sub_index_map;

      /// Return informal string representation (pretty-print)
      /// @param[in] verbose Flag to turn on additional output.
      /// @return An informal representation of the function space.
      std::string str(bool verbose) const;

      /// Accessors
      const std::map<std::int32_t, std::int32_t> & original_to_block(std::size_t b) const;
      const std::map<std::int32_t, std::int32_t> & block_to_original(std::size_t b) const;
      const std::map<std::int32_t, std::int32_t> & original_to_sub_block(std::size_t b) const;
      const std::map<std::int32_t, std::int32_t> & sub_block_to_original(std::size_t b) const;

    protected:
      // Constructor arguments
      std::vector<std::shared_ptr<const dolfinx::fem::DofMap>> _dofmaps;
      std::vector<Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>> _restrictions;

      // Cell-local-to-dof map
      std::map<int, std::vector<std::int32_t>> _cell_dofs;
      std::vector<std::int32_t> _empty_vector;

      // Map from original dofs to block dofs, for each component
      std::vector<std::map<std::int32_t, std::int32_t>> _original_to_block;

      // Map from block dofs to original dofs, for each component
      std::vector<std::map<std::int32_t, std::int32_t>> _block_to_original;

      // Map from original dofs to block sub dofs, for each component
      std::vector<std::map<std::int32_t, std::int32_t>> _original_to_sub_block;

      // Map from block sub dofs to original dofs, for each component
      std::vector<std::map<std::int32_t, std::int32_t>> _sub_block_to_original;

      // Precomputed views
      std::vector<std::shared_ptr<BlockDofMap>> _views;
    };
  }
}

#endif

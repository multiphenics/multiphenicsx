// Copyright (C) 2016-2023 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <memory>
#include <map>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>

namespace multiphenicsx
{

namespace fem
{

/// Restriction of a DofMap to a list of active degrees of freedom

class DofMapRestriction
{
public:
  /// Create a DofMapRestriction from a DofMap and a sorted list of active degrees of freedom
  DofMapRestriction(std::shared_ptr<const dolfinx::fem::DofMap> dofmap,
                    const std::vector<std::int32_t>& restriction,
                    bool legacy = false);

  // Copy constructor
  DofMapRestriction(const DofMapRestriction& dofmap_restriction) = delete;

  /// Move constructor
  DofMapRestriction(DofMapRestriction&& dofmap_restriction) = default;

  /// Destructor
  virtual ~DofMapRestriction() = default;

  /// Copy assignment
  DofMapRestriction& operator=(const DofMapRestriction& dofmap_restriction) = delete;

  /// Move assignment
  DofMapRestriction& operator=(DofMapRestriction&& dofmap_restriction) = default;

  /// Local-to-global mapping of dofs on a cell
  /// @param[in] cell_index The cell index.
  /// @return  Local-global map for cell (used process-local global
  /// index)
  std::span<const std::int32_t> cell_dofs(std::int32_t cell_index) const
  {
    return std::span<const std::int32_t>(
        _dof_array.data() + _cell_bounds[cell_index], _cell_bounds[cell_index + 1] - _cell_bounds[cell_index]);
  }

  /// Accessor to DofMap provided to constructor
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap() const
  {
    return _dofmap;
  }

  /// Return map from unrestricted dofs to restricted dofs
  const std::map<std::int32_t, std::int32_t>& unrestricted_to_restricted() const
  {
    return _unrestricted_to_restricted;
  }

  /// Map from restricted dofs to unrestricted dofs
  const std::map<std::int32_t, std::int32_t>& restricted_to_unrestricted() const
  {
    return _restricted_to_unrestricted;
  }

  /// Get dofmap data after restriction has been carried out
  /// @return The adjacency list with dof indices for each cell
  std::pair<std::span<const std::int32_t>, std::span<const std::size_t>> map() const
  {
    return std::make_pair(std::span<const std::int32_t>(_dof_array), std::span<const std::size_t>(_cell_bounds));
  }

  /// Object containing information about dof distribution across
  /// processes
  std::shared_ptr<const dolfinx::common::IndexMap> index_map;

  /// Block size associated to index_map.
  int index_map_bs() const
  {
    return _dofmap->index_map_bs();
  }

private:
  /// Helper function for constructor: map owned unrestricted dofs into restricted ones
  void _map_owned_dofs(std::shared_ptr<const dolfinx::fem::DofMap> dofmap,
                       const std::vector<std::int32_t>& restriction);

  /// Helper function for constructor: map ghost unrestricted dofs into restricted ones
  void _map_ghost_dofs(std::shared_ptr<const dolfinx::fem::DofMap> dofmap,
                       const std::vector<std::int32_t>& restriction);

  /// Helper function for constructor: compute cell dofs arrays
  void _compute_cell_dofs(std::shared_ptr<const dolfinx::fem::DofMap> dofmap);

  /// DofMap provided to constructor
  std::shared_ptr<const dolfinx::fem::DofMap> _dofmap;

  // Map from unrestricted dofs to restricted dofs
  std::map<std::int32_t, std::int32_t> _unrestricted_to_restricted;

  // Map from restricted dofs to unrestricted dofs
  std::map<std::int32_t, std::int32_t> _restricted_to_unrestricted;

  // Cell-local-to-dof map after restriction has been carried out
  std::vector<std::int32_t> _dof_array;
  std::vector<std::size_t> _cell_bounds;
};
} // namespace fem
} // namespace multiphenicsx

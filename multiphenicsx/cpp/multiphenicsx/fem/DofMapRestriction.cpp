// Copyright (C) 2016-2023 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <multiphenicsx/fem/DofMapRestriction.h>

using namespace dolfinx;
using dolfinx::fem::DofMap;
using multiphenicsx::fem::DofMapRestriction;

//-----------------------------------------------------------------------------
DofMapRestriction::DofMapRestriction(
    std::shared_ptr<const DofMap> dofmap,
    const std::vector<std::int32_t>& restriction)
    : _dofmap(dofmap)
{
  // Discard ghost dofs from the list of provided dofs
  auto dofmap_owned_size = dofmap->index_map->size_local();
  auto restriction_end_owned
      = std::ranges::find_if(restriction, [dofmap_owned_size](std::int32_t d)
                             { return d >= dofmap_owned_size; });
  std::span<const std::int32_t> restriction_owned(restriction.begin(),
                                                  restriction_end_owned);
  // Compute index map
  auto [index_submap, ghost_submap]
      = dofmap->index_map->create_submap(restriction_owned);
  assert(index_submap.size_local()
         == restriction_end_owned - restriction.begin());
  assert(static_cast<int>(ghost_submap.size())
         == restriction.end() - restriction_end_owned);
  for (std::int32_t d = 0; d < index_submap.size_local(); ++d)
  {
    assert(_unrestricted_to_restricted.count(restriction[d]) == 0);
    _unrestricted_to_restricted[restriction[d]] = d;
    assert(_restricted_to_unrestricted.count(d) == 0);
    _restricted_to_unrestricted[d] = restriction[d];
  }
  for (std::size_t d = 0; d < ghost_submap.size(); ++d)
  {
    assert(
        _unrestricted_to_restricted.count(dofmap_owned_size + ghost_submap[d])
        == 0);
    _unrestricted_to_restricted[dofmap_owned_size + ghost_submap[d]]
        = index_submap.size_local() + d;
    assert(_restricted_to_unrestricted.count(index_submap.size_local() + d)
           == 0);
    _restricted_to_unrestricted[index_submap.size_local() + d]
        = dofmap_owned_size + ghost_submap[d];
  }
  // Assign index map to public member
  index_map
      = std::make_shared<dolfinx::common::IndexMap>(std::move(index_submap));

  // Compute cell dofs arrays
  _compute_cell_dofs(dofmap);
}
//-----------------------------------------------------------------------------
void DofMapRestriction::_compute_cell_dofs(std::shared_ptr<const DofMap> dofmap)
{
  // Fill in cell dofs first into a temporary std::map
  std::map<int, std::vector<std::int32_t>> restricted_cell_dofs;
  std::size_t restricted_cell_dofs_total_size = 0;
  auto unrestricted_cell_dofs = dofmap->map();
  const int num_cells = unrestricted_cell_dofs.extent(0);
  for (int c = 0; c < num_cells; ++c)
  {
    const auto unrestricted_cell_dofs_c = std::experimental::submdspan(
        unrestricted_cell_dofs, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    std::vector<std::int32_t> restricted_cell_dofs_c;
    restricted_cell_dofs_c.reserve(
        unrestricted_cell_dofs_c.size()); // conservative allocation
    for (std::uint32_t d = 0; d < unrestricted_cell_dofs_c.size(); ++d)
    {
      const auto unrestricted_dof = unrestricted_cell_dofs_c[d];
      if (_unrestricted_to_restricted.count(unrestricted_dof) > 0)
      {
        restricted_cell_dofs_c.push_back(
            _unrestricted_to_restricted[unrestricted_dof]);
      }
    }
    if (restricted_cell_dofs_c.size() > 0)
    {
      restricted_cell_dofs[c].insert(restricted_cell_dofs[c].end(),
                                     restricted_cell_dofs_c.begin(),
                                     restricted_cell_dofs_c.end());
      restricted_cell_dofs_total_size += restricted_cell_dofs_c.size();
    }
  }

  // Flatten std::map into the std::vector dof_array, and store start/end
  // indices associated to each cell in cell_bounds
  _dof_array.reserve(restricted_cell_dofs_total_size);
  _cell_bounds.reserve(num_cells + 1);
  std::size_t current_cell_bound = 0;
  _cell_bounds.push_back(current_cell_bound);
  for (int c = 0; c < num_cells; ++c)
  {
    if (restricted_cell_dofs.count(c) > 0)
    {
      const auto restricted_cell_dofs_c = restricted_cell_dofs.at(c);
      assert(current_cell_bound + restricted_cell_dofs_c.size()
             <= restricted_cell_dofs_total_size);
      _dof_array.insert(_dof_array.end(), restricted_cell_dofs_c.begin(),
                        restricted_cell_dofs_c.end());
      current_cell_bound += restricted_cell_dofs_c.size();
    }
    _cell_bounds.push_back(current_cell_bound);
  }
}
//-----------------------------------------------------------------------------

// Copyright (C) 2016-2026 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <dolfinx/fem/Form.h>
#include <dolfinx/la/SparsityPattern.h>
#include <multiphenicsx/fem/sparsitybuild.h>

namespace multiphenicsx
{

namespace fem
{

/// @brief Create a sparsity pattern for a given form.
/// @note The pattern is not finalised, i.e. the caller is responsible
/// for calling SparsityPattern::assemble.
/// @param[in] a A bilinear form
/// @param[in] index_maps A pair of index maps. Row index map is given by
/// index_maps[0], column index map is given by index_maps[1].
/// @param[in] index_maps_bs A pair of int, representing the block size of
/// index_maps.
/// @param[in] dofmaps_list An array of spans containing the dofmaps list. Row
/// dofmap is given by dofmaps[0], while column dofmap is given by dofmaps[1].
/// @param[in] dofmaps_bounds An array of spans containing the dofmaps cell
/// bounds.
/// @return The corresponding sparsity pattern
template <typename T, std::floating_point U>
dolfinx::la::SparsityPattern create_sparsity_pattern(
    const dolfinx::fem::Form<T, U>& a,
    std::array<std::reference_wrapper<const dolfinx::common::IndexMap>, 2>
        index_maps,
    const std::array<int, 2> index_maps_bs,
    std::array<std::span<const std::int32_t>, 2> dofmaps_list,
    std::array<std::span<const std::size_t>, 2> dofmaps_bounds)
{
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear.");
  }

  // Get mesh
  std::shared_ptr mesh = a.mesh();
  assert(mesh);

  const std::set<dolfinx::fem::IntegralType> integral_types
      = a.integral_types();
  if (integral_types.find(dolfinx::fem::IntegralType::interior_facet)
          != integral_types.end()
      or integral_types.find(dolfinx::fem::IntegralType::exterior_facet)
             != integral_types.end())
  {
    // FIXME: cleanup these calls? Some of the happen internally again.
    int tdim = mesh->topology()->dim();
    mesh->topology_mutable()->create_entities(tdim - 1);
    mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
  }

  dolfinx::common::Timer t0("Build sparsity");

  // Create and build sparsity pattern
  const std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2>
      index_maps_shared_ptr{
          {std::shared_ptr<const dolfinx::common::IndexMap>(
               &index_maps[0].get(), [](const dolfinx::common::IndexMap*) {}),
           std::shared_ptr<const dolfinx::common::IndexMap>(
               &index_maps[1].get(), [](const dolfinx::common::IndexMap*) {})}};
  dolfinx::la::SparsityPattern pattern(mesh->comm(), index_maps_shared_ptr,
                                       index_maps_bs);
  for (auto integral_type : integral_types)
  {
    switch (integral_type)
    {
    case dolfinx::fem::IntegralType::cell:
      for (int i = 0; i < a.num_integrals(integral_type, 0); ++i)
      {
        multiphenicsx::fem::sparsitybuild::cells(
            pattern, a.domain_arg(integral_type, 0, i, 0), dofmaps_list,
            dofmaps_bounds);
      }
      break;
    case dolfinx::fem::IntegralType::interior_facet:
      for (int i = 0; i < a.num_integrals(integral_type, 0); ++i)
      {
        std::span<const std::int32_t> facets
            = a.domain_arg(integral_type, 0, i, 0);
        std::vector<std::int32_t> f;
        f.reserve(facets.size() / 2);
        for (std::size_t i = 0; i < facets.size(); i += 4)
          f.insert(f.end(), {facets[i], facets[i + 2]});
        multiphenicsx::fem::sparsitybuild::interior_facets(
            pattern, f, dofmaps_list, dofmaps_bounds);
      }
      break;
    case dolfinx::fem::IntegralType::exterior_facet:
      for (int i = 0; i < a.num_integrals(integral_type, 0); ++i)
      {
        std::span<const std::int32_t> facets
            = a.domain_arg(integral_type, 0, i, 0);
        std::vector<std::int32_t> cells;
        cells.reserve(facets.size() / 2);
        for (std::size_t i = 0; i < facets.size(); i += 2)
          cells.push_back(facets[i]);
        multiphenicsx::fem::sparsitybuild::cells(pattern, cells, dofmaps_list,
                                                 dofmaps_bounds);
      }
      break;
    default:
      throw std::runtime_error("Unsupported integral type");
    }
  }

  t0.stop();

  return pattern;
}

} // namespace fem
} // namespace multiphenicsx

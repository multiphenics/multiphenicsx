// Copyright (C) 2016-2023 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Topology.h>

namespace multiphenicsx
{

namespace fem
{

/// Functions to build sparsity patterns from degree-of-freedom maps

namespace sparsitybuild
{

/// Iterate over cells and insert entries into sparsity pattern
void cells(dolfinx::la::SparsityPattern& pattern,
           std::span<const std::int32_t> cells,
           std::array<const dolfinx::graph::AdjacencyList<std::int32_t>*, 2> dofmaps);

/// Iterate over interior facets and insert entries into sparsity pattern
void interior_facets(dolfinx::la::SparsityPattern& pattern,
                     std::span<const std::int32_t> facets,
                     std::array<const dolfinx::graph::AdjacencyList<std::int32_t>*, 2> dofmaps);

} // namespace sparsitybuild
} // namespace fem
} // namespace multiphenicsx

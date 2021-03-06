// Copyright (C) 2016-2022 by the multiphenicsx authors
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
           const dolfinx::mesh::Topology& topology,
           std::array<const dolfinx::graph::AdjacencyList<std::int32_t>*, 2> dofmaps);

/// Iterate over interior facets and insert entries into sparsity pattern
void interior_facets(dolfinx::la::SparsityPattern& pattern,
                     const dolfinx::mesh::Topology& topology,
                     std::array<const dolfinx::graph::AdjacencyList<std::int32_t>*, 2> dofmaps);

/// Iterate over exterior facets and insert entries into sparsity pattern
void exterior_facets(dolfinx::la::SparsityPattern& pattern,
                     const dolfinx::mesh::Topology& topology,
                     std::array<const dolfinx::graph::AdjacencyList<std::int32_t>*, 2> dofmaps);

} // namespace sparsitybuild
} // namespace fem
} // namespace multiphenicsx

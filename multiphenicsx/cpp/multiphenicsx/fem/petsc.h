// Copyright (C) 2016-2022 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <memory>
#include <petscmat.h>
#include <petscvec.h>
#include <set>
#include <vector>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>

namespace multiphenicsx
{

namespace fem
{

using dolfinx::fem::IntegralType;

/// Helper functions for assembly into PETSc data structures
namespace petsc
{

/// Create a matrix.
/// @param[in] mesh The mesh
/// @param[in] index_maps A pair of index maps. Row index map is given by index_maps[0], column index map is given
///                       by index_maps[1].
/// @param[in] index_maps_bs A pair of int, representing the block size of index_maps.
/// @param[in] integral_types Required integral types
/// @param[in] dofmaps A pair of AdjacencyList containing the dofmaps. Row dofmap is given by dofmaps[0], while
///                    column dofmap is given by dofmaps[1].
/// @param[in] matrix_type The PETSc matrix type to create
/// @return A sparse matrix with a layout and sparsity that matches the one required by the provided index maps
/// integral types and dofmaps. The caller is responsible for destroying the Mat object.
Mat create_matrix(
    const dolfinx::mesh::Mesh& mesh,
    std::array<std::reference_wrapper<const dolfinx::common::IndexMap>, 2> index_maps,
    const std::array<int, 2> index_maps_bs,
    const std::set<dolfinx::fem::IntegralType>& integral_types,
    std::array<const dolfinx::graph::AdjacencyList<std::int32_t>*, 2> dofmaps,
    const std::string& matrix_type = std::string());

/// Initialise monolithic matrix for an array for bilinear forms. Matrix
/// is not zeroed.
/// @param[in] mesh The mesh
/// @param[in] index_maps A pair of vectors of index maps. Index maps for block (i, j) will be
///                       constructed from (index_maps[0][i], index_maps[1][j]).
/// @param[in] index_maps_bs A pair of vectors of int, representing the block size of the
///                          corresponding entry in index_maps.
/// @param[in] integral_types A matrix of required integral types. Required integral types
///                                 for block (i, j) are deduced from integral_types[i, j].
/// @param[in] dofmaps A pair of vectors of AdjacencyList containing the list dofmaps for each block.
///                    The dofmap pair for block (i, j) will be constructed from
///                    (dofmaps[0][i], dofmaps[1][j]).
/// @param[in] matrix_type The type of PETSc Mat. If empty the PETSc default is
///                 used.
/// @return A sparse matrix with a layout and sparsity that matches the one required by the provided index maps
/// integral types and dofmaps. The caller is responsible for destroying the Mat object.
Mat create_matrix_block(
    const dolfinx::mesh::Mesh& mesh,
    std::array<std::vector<std::reference_wrapper<const dolfinx::common::IndexMap>>, 2> index_maps,
    const std::array<std::vector<int>, 2> index_maps_bs,
    const std::vector<std::vector<std::set<dolfinx::fem::IntegralType>>>& integral_types,
    const std::array<std::vector<const dolfinx::graph::AdjacencyList<std::int32_t>*>, 2>& dofmaps,
    const std::string& matrix_type = std::string());

/// Create nested (MatNest) matrix. Matrix is not zeroed.
/// @param[in] mesh The mesh
/// @param[in] index_maps A pair of vectors of index maps. Index maps for block (i, j) will be
///                       constructed from (index_maps[0][i], index_maps[1][j]).
/// @param[in] index_maps_bs A pair of vectors of int, representing the block size of the
///                          corresponding entry in index_maps.
/// @param[in] integral_types A matrix of required integral types. Required integral types
///                           for block (i, j) are deduced from integral_types[i, j].
/// @param[in] dofmaps A pair of vectors of AdjacencyList containing the list dofmaps for each block.
///                    The dofmap pair for block (i, j) will be constructed from
///                    (dofmaps[0][i], dofmaps[1][j]).
/// @param[in] matrix_types A matrix of PETSc Mat types.
/// @return A sparse matrix with a layout and sparsity that matches the one required by the provided index maps
/// integral types and dofmaps. The caller is responsible for destroying the Mat object.
Mat create_matrix_nest(
    const dolfinx::mesh::Mesh& mesh,
    std::array<std::vector<std::reference_wrapper<const dolfinx::common::IndexMap>>, 2> index_maps,
    const std::array<std::vector<int>, 2> index_maps_bs,
    const std::vector<std::vector<std::set<dolfinx::fem::IntegralType>>>& integral_types,
    const std::array<std::vector<const dolfinx::graph::AdjacencyList<std::int32_t>*>, 2>& dofmaps,
    const std::vector<std::vector<std::string>>& matrix_types);

} // namespace petsc
} // namespace fem
} // namespace multiphenicsx

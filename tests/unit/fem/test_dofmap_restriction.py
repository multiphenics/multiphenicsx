# Copyright (C) 2016-2023 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for multiphenicsx.fem.dofmap_restriction module."""

import typing

import dolfinx.fem
import dolfinx.mesh
import mpi4py.MPI
import numpy as np
import pytest

import common  # noqa
import multiphenicsx.fem


@pytest.fixture
def mesh() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    return dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, 4, 4)


def get_subdomains() -> typing.Tuple[common.SubdomainType, ...]:
    """Generate subdomain parametrization."""
    return (
        # Cells restrictions
        common.CellsAll(),
        common.CellsSubDomain(0.5, 0.5),
        # Facets restrictions
        common.FacetsAll(),
        common.FacetsSubDomain(on_boundary=True),
        common.FacetsSubDomain(X=1.0),
        common.FacetsSubDomain(Y=0.0),
        common.FacetsSubDomain(X=0.75),
        common.FacetsSubDomain(Y=0.25)
    )


def get_function_spaces() -> typing.Tuple[common.FunctionSpaceGeneratorType, ...]:
    """Generate function space parametrization."""
    return (
        lambda mesh: dolfinx.fem.functionspace(mesh, ("Lagrange", 1)),
        lambda mesh: dolfinx.fem.functionspace(mesh, ("Lagrange", 2)),
        lambda mesh: dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, ))),
        lambda mesh: dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim, ))),
        lambda mesh: dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, mesh.geometry.dim))),
        lambda mesh: dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim, mesh.geometry.dim))),
        lambda mesh: common.TaylorHoodFunctionSpace(mesh, ("Lagrange", 1)),
        # lambda mesh: common.TaylorHoodFunctionSpace(mesh, ("Lagrange", 2))
    )


def assert_dofmap_restriction_is_subset_of_dofmap(
    mesh: dolfinx.mesh.Mesh, dofmap: dolfinx.fem.DofMap, dofmap_restriction: multiphenicsx.fem.DofMapRestriction
) -> None:
    """Run checks on DofMapRestriction in the case where restriction is a strict subset of the available dofs."""
    # Get local dimensions
    unrestricted_local_dimension = (dofmap.index_map.local_range[1]
                                    - dofmap.index_map.local_range[0])
    restricted_local_dimension = (dofmap_restriction.index_map.local_range[1]
                                  - dofmap_restriction.index_map.local_range[0])
    assert unrestricted_local_dimension >= restricted_local_dimension
    # Get map from restricted to unrestricted dofs
    restricted_to_unrestricted = dofmap_restriction.restricted_to_unrestricted
    active_unrestricted_dofs = restricted_to_unrestricted.values()
    # Run checks on each cell
    cells_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = cells_map.size_local + cells_map.num_ghosts
    for c in range(num_cells):
        unrestricted_cell_dofs = dofmap.cell_dofs(c)
        restricted_cell_dofs = dofmap_restriction.cell_dofs(c)
        # By filtering out inactive dofs, cell_dofs returned by dofmap
        # should be the same as the cell_dofs returned by dofmap_restriction postprocessed
        # by the transformation through restricted_to_unrestricted
        assert np.array_equal([d for d in unrestricted_cell_dofs if d in active_unrestricted_dofs],
                              [restricted_to_unrestricted[d] for d in restricted_cell_dofs])


def assert_dofmap_restriction_is_same_as_dofmap(
    mesh: dolfinx.mesh.Mesh, dofmap: dolfinx.fem.DofMap, dofmap_restriction: multiphenicsx.fem.DofMapRestriction
) -> None:
    """Run stricter checks on DofMapRestriction in the case where restriction actually contains all available dofs."""
    # Assert that local dimensions are the same
    unrestricted_local_dimension = (dofmap.index_map.local_range[1]
                                    - dofmap.index_map.local_range[0])
    restricted_local_dimension = (dofmap_restriction.index_map.local_range[1]
                                  - dofmap_restriction.index_map.local_range[0])
    assert unrestricted_local_dimension == restricted_local_dimension
    # Assert that global dimensions are the same
    unrestricted_global_dimension = dofmap.index_map.size_global
    restricted_global_dimension = dofmap_restriction.index_map.size_global
    assert unrestricted_global_dimension == restricted_global_dimension
    # Assert that ghosts are the same, apart from ordering
    unrestricted_ghosts = dofmap.index_map.ghosts
    restricted_ghosts = dofmap_restriction.index_map.ghosts
    assert np.array_equal(np.sort(unrestricted_ghosts), np.sort(restricted_ghosts))
    # Assert that the map from restricted to unrestricted is the identity for owned ghosts
    # (but may not be the identity for ghosts, due to different ordering)
    restricted_to_unrestricted = dofmap_restriction.restricted_to_unrestricted
    assert np.array_equal([d for d in range(restricted_local_dimension)],
                          [restricted_to_unrestricted[d] for d in range(restricted_local_dimension)])
    # Assert that the map from unrestricted to restricted is the identity for owned ghosts
    # (but may not be the identity for ghosts, due to different ordering)
    unrestricted_to_restricted = dofmap_restriction.unrestricted_to_restricted
    assert np.array_equal([d for d in range(unrestricted_local_dimension)],
                          [unrestricted_to_restricted[d] for d in range(unrestricted_local_dimension)])
    # Run checks on each cell
    cells_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = cells_map.size_local + cells_map.num_ghosts
    for c in range(num_cells):
        unrestricted_cell_dofs = dofmap.cell_dofs(c)
        restricted_cell_dofs = dofmap_restriction.cell_dofs(c)
        # cell_dofs returned by dofmap should be the same as cell_dofs
        # returned by dofmap_restriction, apart from ghosts who may have
        # different numbering
        assert np.array_equal([d for d in unrestricted_cell_dofs],
                              [restricted_to_unrestricted[d] for d in restricted_cell_dofs])
        # Global numbering should also be the same in dofmap and dofmap_restriction
        assert np.array_equal(dofmap.index_map.local_to_global(unrestricted_cell_dofs),
                              dofmap_restriction.index_map.local_to_global(restricted_cell_dofs))


@pytest.mark.parametrize("FunctionSpace", get_function_spaces())
def test_dofmap_restriction_is_same_as_dofmap(
    mesh: dolfinx.mesh.Mesh, FunctionSpace: common.FunctionSpaceGeneratorType
) -> None:
    """Test for DofMapRestriction in the case where restriction actually contains all available dofs."""
    V = FunctionSpace(mesh)
    active_dofs = common.ActiveDofs(V, subdomain=None)
    dofmap_restriction = multiphenicsx.fem.DofMapRestriction(V.dofmap, active_dofs)
    assert_dofmap_restriction_is_same_as_dofmap(mesh, V.dofmap, dofmap_restriction)
    # Comparison between legacy and new implementation
    dofmap_restriction_legacy = multiphenicsx.fem.DofMapRestriction(V.dofmap, active_dofs, False)
    dofmap_restriction_new = multiphenicsx.fem.DofMapRestriction(V.dofmap, active_dofs, True)
    assert dofmap_restriction_legacy.index_map.size_local == dofmap_restriction_new.index_map.size_local
    assert np.array_equal(
        [dofmap_restriction_legacy.restricted_to_unrestricted[d]
         for d in range(dofmap_restriction_legacy.index_map.size_local)],
        [dofmap_restriction_new.restricted_to_unrestricted[d]
         for d in range(dofmap_restriction_new.index_map.size_local)])
    assert np.array_equal(
        [dofmap_restriction_legacy.unrestricted_to_restricted[d]
         for d in range(V.dofmap.index_map.size_local)],
        [dofmap_restriction_new.unrestricted_to_restricted[d]
         for d in range(V.dofmap.index_map.size_local)])
    masked_map_0_legacy = np.ma.masked_greater_equal(  # type: ignore[no-untyped-call]
        dofmap_restriction_legacy.map()[0], dofmap_restriction_legacy.index_map.size_local)
    masked_map_0_new = np.ma.masked_greater_equal(  # type: ignore[no-untyped-call]
        dofmap_restriction_new.map()[0], dofmap_restriction_new.index_map.size_local)
    assert np.array_equal(masked_map_0_legacy.mask, masked_map_0_new.mask)
    assert np.ma.allequal(masked_map_0_legacy, masked_map_0_new)  # type: ignore[no-untyped-call]
    assert np.array_equal(dofmap_restriction_legacy.map()[1], dofmap_restriction_new.map()[1])


@pytest.mark.parametrize("subdomain", get_subdomains())
@pytest.mark.parametrize("FunctionSpace", get_function_spaces())
def test_dofmap_restriction_is_subset_of_dofmap(
    mesh: dolfinx.mesh.Mesh, subdomain: common.SubdomainType, FunctionSpace: common.FunctionSpaceGeneratorType
) -> None:
    """Test for DofMapRestriction in the case where restriction is a strict subset of the available dofs."""
    V = FunctionSpace(mesh)
    active_dofs = common.ActiveDofs(V, subdomain)
    dofmap_restriction = multiphenicsx.fem.DofMapRestriction(V.dofmap, active_dofs)
    assert_dofmap_restriction_is_subset_of_dofmap(mesh, V.dofmap, dofmap_restriction)
    # Comparison between legacy and new implementation
    dofmap_restriction_legacy = multiphenicsx.fem.DofMapRestriction(V.dofmap, active_dofs, False)
    dofmap_restriction_new = multiphenicsx.fem.DofMapRestriction(V.dofmap, active_dofs, True)
    assert dofmap_restriction_legacy.index_map.size_local == dofmap_restriction_new.index_map.size_local
    assert np.array_equal(
        [dofmap_restriction_legacy.restricted_to_unrestricted[d]
         for d in range(dofmap_restriction_legacy.index_map.size_local)],
        [dofmap_restriction_new.restricted_to_unrestricted[d]
         for d in range(dofmap_restriction_new.index_map.size_local)])
    assert np.array_equal(
        [dofmap_restriction_legacy.unrestricted_to_restricted[d]
         for d in active_dofs if d < dofmap_restriction_legacy.index_map.size_local],
        [dofmap_restriction_new.unrestricted_to_restricted[d]
         for d in active_dofs if d < dofmap_restriction_new.index_map.size_local])
    masked_map_0_legacy = np.ma.masked_greater_equal(  # type: ignore[no-untyped-call]
        dofmap_restriction_legacy.map()[0], dofmap_restriction_legacy.index_map.size_local)
    masked_map_0_new = np.ma.masked_greater_equal(  # type: ignore[no-untyped-call]
        dofmap_restriction_new.map()[0], dofmap_restriction_new.index_map.size_local)
    assert np.array_equal(masked_map_0_legacy.mask, masked_map_0_new.mask)
    assert np.ma.allequal(masked_map_0_legacy, masked_map_0_new)  # type: ignore[no-untyped-call]
    assert np.array_equal(dofmap_restriction_legacy.map()[1], dofmap_restriction_new.map()[1])

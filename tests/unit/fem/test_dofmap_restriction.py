# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for multiphenicsx.fem.dofmap_restriction module."""

import typing

import dolfinx.fem
import dolfinx.mesh
import mpi4py
import numpy as np
import pytest

import common  # noqa
import multiphenicsx.fem


@pytest.fixture
def mesh() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    return dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, 4, 4)


def get_subdomains() -> typing.Tuple[typing.Callable]:
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


def get_function_spaces() -> typing.Tuple[typing.Callable]:
    """Generate function space parametrization."""
    return (
        lambda mesh: dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1)),
        lambda mesh: dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 2)),
        lambda mesh: dolfinx.fem.VectorFunctionSpace(mesh, ("Lagrange", 1)),
        lambda mesh: dolfinx.fem.VectorFunctionSpace(mesh, ("Lagrange", 2)),
        lambda mesh: dolfinx.fem.TensorFunctionSpace(mesh, ("Lagrange", 1)),
        lambda mesh: dolfinx.fem.TensorFunctionSpace(mesh, ("Lagrange", 2)),
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
    # Assert that ghosts are the same
    unrestricted_ghosts = dofmap.index_map.ghosts
    restricted_ghosts = dofmap_restriction.index_map.ghosts
    assert np.array_equal(unrestricted_ghosts, restricted_ghosts)
    # Run checks on each cell
    cells_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = cells_map.size_local + cells_map.num_ghosts
    unrestricted_global_indices = dofmap.index_map.global_indices()
    restricted_global_indices = dofmap_restriction.index_map.global_indices()
    for c in range(num_cells):
        unrestricted_cell_dofs = dofmap.cell_dofs(c)
        restricted_cell_dofs = dofmap_restriction.cell_dofs(c)
        # cell_dofs returned by dofmap should be the same as cell_dofs
        # returned by dofmap_restriction
        assert np.array_equal([d for d in unrestricted_cell_dofs],
                              [d for d in restricted_cell_dofs])
        # Global numbering should also be the same in dofmap and dofmap_restriction
        assert np.array_equal([unrestricted_global_indices[d] for d in unrestricted_cell_dofs],
                              [restricted_global_indices[d] for d in restricted_cell_dofs])


@pytest.mark.parametrize("FunctionSpace", get_function_spaces())
def test_dofmap_restriction_is_same_as_dofmap(
    mesh: dolfinx.mesh.Mesh, FunctionSpace: typing.Callable
) -> None:
    """Test for DofMapRestriction in the case where restriction actually contains all available dofs."""
    V = FunctionSpace(mesh)
    active_dofs = common.ActiveDofs(V, subdomain=None)
    dofmap_restriction = multiphenicsx.fem.DofMapRestriction(V.dofmap, active_dofs)
    assert_dofmap_restriction_is_same_as_dofmap(mesh, V.dofmap, dofmap_restriction)


@pytest.mark.parametrize("subdomain", get_subdomains())
@pytest.mark.parametrize("FunctionSpace", get_function_spaces())
def test_dofmap_restriction_is_subset_of_dofmap(
    mesh: dolfinx.mesh.Mesh, subdomain: typing.Callable, FunctionSpace: typing.Callable
) -> None:
    """Test for DofMapRestriction in the case where restriction is a strict subset of the available dofs."""
    V = FunctionSpace(mesh)
    active_dofs = common.ActiveDofs(V, subdomain)
    dofmap_restriction = multiphenicsx.fem.DofMapRestriction(V.dofmap, active_dofs)
    assert_dofmap_restriction_is_subset_of_dofmap(mesh, V.dofmap, dofmap_restriction)

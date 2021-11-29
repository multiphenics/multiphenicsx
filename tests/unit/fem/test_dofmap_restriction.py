# Copyright (C) 2016-2021 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pytest
from mpi4py import MPI

from dolfinx.fem import FunctionSpace, locate_dofs_topological, TensorFunctionSpace, VectorFunctionSpace
from dolfinx.generation import UnitSquareMesh
from dolfinx.mesh import locate_entities
from ufl import FiniteElement, MixedElement, VectorElement

from multiphenicsx.fem import DofMapRestriction


# Mesh
@pytest.fixture
def mesh():
    return UnitSquareMesh(MPI.COMM_WORLD, 4, 4)


# Auxiliary generation of a mixed function space
def TaylorHoodFunctionSpace(mesh, family_degree):
    (family, degree) = family_degree
    V_element = VectorElement(family, mesh.ufl_cell(), degree + 1)
    Q_element = FiniteElement(family, mesh.ufl_cell(), degree)
    taylor_hood_element = MixedElement(V_element, Q_element)
    return FunctionSpace(mesh, taylor_hood_element)


# Function space parametrization
def get_function_spaces():
    return (
        lambda mesh: FunctionSpace(mesh, ("Lagrange", 1)),
        lambda mesh: FunctionSpace(mesh, ("Lagrange", 2)),
        lambda mesh: VectorFunctionSpace(mesh, ("Lagrange", 1)),
        lambda mesh: VectorFunctionSpace(mesh, ("Lagrange", 2)),
        lambda mesh: TensorFunctionSpace(mesh, ("Lagrange", 1)),
        lambda mesh: TensorFunctionSpace(mesh, ("Lagrange", 2)),
        lambda mesh: TaylorHoodFunctionSpace(mesh, ("Lagrange", 1)),
        # lambda mesh: TaylorHoodFunctionSpace(mesh, ("Lagrange", 2))
    )


# Definition of some representative subdomains
def CellsAll():
    def cells_all(x):
        return np.full(x.shape[1], True)
    cells_all.codimension = 0
    return cells_all


def CellsSubDomain(X, Y):
    def cells_subdomain(x):
        return np.logical_and(x[0] <= X, x[1] <= Y)
    cells_subdomain.codimension = 0
    return cells_subdomain


def FacetsAll():
    def facets_all(x):
        return np.full(x.shape[1], True)
    facets_all.codimension = 1
    return facets_all


def FacetsSubDomain(X=None, Y=None, on_boundary=False):
    eps = np.finfo(float).eps
    assert ((X is not None and Y is None and on_boundary is False)
            or (X is None and Y is not None and on_boundary is False)
            or (X is None and Y is None and on_boundary is True))
    if X is not None:
        def facets_subdomain(x):
            return np.logical_and(x[0] >= X - eps, x[0] <= X + eps)
    elif Y is not None:
        def facets_subdomain(x):
            return np.logical_and(x[1] >= Y - eps, x[1] <= Y + eps)
    elif on_boundary is True:
        def facets_subdomain(x):
            return np.logical_or(
                np.logical_or(x[0] <= eps, x[0] >= 1. - eps),
                np.logical_or(x[1] <= eps, x[1] >= 1. - eps)
            )
    facets_subdomain.codimension = 1
    return facets_subdomain


# Auxiliary function for definition of a list of active dofs
def ActiveDofs(V, subdomain):
    if subdomain is not None:
        entities_dim = V.mesh.topology.dim - subdomain.codimension
        entities = locate_entities(V.mesh, entities_dim, subdomain)
        return locate_dofs_topological(V, entities_dim, entities)
    else:
        return np.arange(0, V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts)


# Subdomain parametrization
def get_subdomains():
    return (
        # Cells restrictions
        CellsAll(),
        CellsSubDomain(0.5, 0.5),
        # Facets restrictions
        FacetsAll(),
        FacetsSubDomain(on_boundary=True),
        FacetsSubDomain(X=1.0),
        FacetsSubDomain(Y=0.0),
        FacetsSubDomain(X=0.75),
        FacetsSubDomain(Y=0.25)
    )


# Run checks on DofMapRestriction in the case where restriction is a strict subset of the available dofs
def assert_dofmap_restriction_is_subset_of_dofmap(mesh, dofmap, dofmap_restriction):
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


# Run stricter checks on DofMapRestriction in the case where restriction actually contains all available dofs
def assert_dofmap_restriction_is_same_as_dofmap(mesh, dofmap, dofmap_restriction):
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


# Test for DofMapRestriction in the case where restriction actually contains all available dofs
@pytest.mark.parametrize("FunctionSpace", get_function_spaces())
def test_dofmap_restriction_is_same_as_dofmap(mesh, FunctionSpace):
    V = FunctionSpace(mesh)
    active_dofs = ActiveDofs(V, subdomain=None)
    dofmap_restriction = DofMapRestriction(V.dofmap, active_dofs)
    assert_dofmap_restriction_is_same_as_dofmap(mesh, V.dofmap, dofmap_restriction)


# Test for DofMapRestriction in the case where restriction is a strict subset of the available dofs
@pytest.mark.parametrize("subdomain", get_subdomains())
@pytest.mark.parametrize("FunctionSpace", get_function_spaces())
def test_dofmap_restriction_is_subset_of_dofmap(mesh, subdomain, FunctionSpace):
    V = FunctionSpace(mesh)
    active_dofs = ActiveDofs(V, subdomain)
    dofmap_restriction = DofMapRestriction(V.dofmap, active_dofs)
    assert_dofmap_restriction_is_subset_of_dofmap(mesh, V.dofmap, dofmap_restriction)

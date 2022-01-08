# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Common functions used across multiphenicsx.fem test files."""

import typing

import dolfinx.fem
import dolfinx.mesh
import numpy as np
import ufl


def ActiveDofs(V: dolfinx.fem.FunctionSpace, subdomain: np.typing.NDArray[bool]) -> np.typing.NDArray[np.int64]:
    """Define a list of active dofs."""
    if subdomain is not None:
        entities_dim = V.mesh.topology.dim - subdomain.codimension
        entities = dolfinx.mesh.locate_entities(V.mesh, entities_dim, subdomain)
        return dolfinx.fem.locate_dofs_topological(V, entities_dim, entities)
    else:
        return np.arange(0, V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts)


def CellsAll() -> typing.Callable:
    """Define a subdomain of codimension 0 marking all cells in the mesh."""
    def cells_all(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[bool]:
        return np.full(x.shape[1], True)
    cells_all.codimension = 0
    return cells_all


def CellsSubDomain(X: np.float64, Y: np.float64) -> typing.Callable:
    """Define a subdomain of codimension 0 marking a subset of the cells in the mesh."""
    def cells_subdomain(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[bool]:
        return np.logical_and(x[0] <= X, x[1] <= Y)
    cells_subdomain.codimension = 0
    return cells_subdomain


def FacetsAll() -> typing.Callable:
    """Define a subdomain of codimension 1 marking all facets in the mesh."""
    def facets_all(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[bool]:
        return np.full(x.shape[1], True)
    facets_all.codimension = 1
    return facets_all


def FacetsSubDomain(
    X: typing.Optional[np.float64] = None, Y: typing.Optional[np.float64] = None,
    on_boundary: typing.Optional[bool] = False
) -> typing.Callable:
    """Define a subdomain of codimension 1 marking a subset of the facets in the mesh."""
    eps = np.finfo(float).eps
    assert ((X is not None and Y is None and on_boundary is False)
            or (X is None and Y is not None and on_boundary is False)
            or (X is None and Y is None and on_boundary is True))
    if X is not None:
        def facets_subdomain(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[bool]:
            return np.logical_and(x[0] >= X - eps, x[0] <= X + eps)
    elif Y is not None:
        def facets_subdomain(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[bool]:
            return np.logical_and(x[1] >= Y - eps, x[1] <= Y + eps)
    elif on_boundary is True:
        def facets_subdomain(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[bool]:
            return np.logical_or(
                np.logical_or(x[0] <= eps, x[0] >= 1. - eps),
                np.logical_or(x[1] <= eps, x[1] >= 1. - eps)
            )
    facets_subdomain.codimension = 1
    return facets_subdomain


def TaylorHoodFunctionSpace(mesh: dolfinx.mesh.Mesh, family_degree: int) -> dolfinx.fem.FunctionSpace:
    """Define a mixed function space."""
    (family, degree) = family_degree
    V_element = ufl.VectorElement(family, mesh.ufl_cell(), degree + 1)
    Q_element = ufl.FiniteElement(family, mesh.ufl_cell(), degree)
    taylor_hood_element = ufl.MixedElement(V_element, Q_element)
    return dolfinx.fem.FunctionSpace(mesh, taylor_hood_element)

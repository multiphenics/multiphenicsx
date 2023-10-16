# Copyright (C) 2016-2023 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Common functions used across multiphenicsx.fem test files."""

import typing

import basix.ufl
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import numpy.typing

SubdomainType = typing.Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.bool_]]
FunctionSpaceGeneratorType = typing.Callable[[dolfinx.mesh.Mesh], dolfinx.fem.FunctionSpaceBase]


def ActiveDofs(
    V: dolfinx.fem.FunctionSpaceBase, subdomain: typing.Optional[SubdomainType]
) -> np.typing.NDArray[np.int32]:
    """Define a list of active dofs."""
    if subdomain is not None:
        entities_dim = V.mesh.topology.dim - subdomain.codimension  # type: ignore[attr-defined]
        entities = dolfinx.mesh.locate_entities(V.mesh, entities_dim, subdomain)
        return dolfinx.fem.locate_dofs_topological(V, entities_dim, entities)
    else:
        return np.arange(0, V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts)


def CellsAll() -> SubdomainType:
    """Define a subdomain of codimension 0 marking all cells in the mesh."""
    def cells_all(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.bool_]:
        return np.full(x.shape[1], True)
    cells_all.codimension = 0  # type: ignore[attr-defined]
    return cells_all


def CellsSubDomain(X: float, Y: float) -> SubdomainType:
    """Define a subdomain of codimension 0 marking a subset of the cells in the mesh."""
    def cells_subdomain(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.bool_]:
        return np.logical_and(x[0] <= X, x[1] <= Y)  # type: ignore[no-any-return]
    cells_subdomain.codimension = 0  # type: ignore[attr-defined]
    return cells_subdomain


def FacetsAll() -> SubdomainType:
    """Define a subdomain of codimension 1 marking all facets in the mesh."""
    def facets_all(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.bool_]:
        return np.full(x.shape[1], True)
    facets_all.codimension = 1  # type: ignore[attr-defined]
    return facets_all


def FacetsSubDomain(
    X: typing.Optional[float] = None, Y: typing.Optional[float] = None,
    on_boundary: bool = False
) -> SubdomainType:
    """Define a subdomain of codimension 1 marking a subset of the facets in the mesh."""
    eps = np.finfo(float).eps
    assert ((X is not None and Y is None and on_boundary is False)
            or (X is None and Y is not None and on_boundary is False)
            or (X is None and Y is None and on_boundary is True))
    if X is not None:
        def facets_subdomain(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.bool_]:
            return np.logical_and(x[0] >= X - eps, x[0] <= X + eps)  # type: ignore[no-any-return]
    elif Y is not None:
        def facets_subdomain(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.bool_]:
            return np.logical_and(x[1] >= Y - eps, x[1] <= Y + eps)  # type: ignore[no-any-return]
    elif on_boundary is True:
        def facets_subdomain(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.bool_]:
            return np.logical_or(  # type: ignore[no-any-return]
                np.logical_or(x[0] <= eps, x[0] >= 1. - eps),
                np.logical_or(x[1] <= eps, x[1] >= 1. - eps)
            )
    facets_subdomain.codimension = 1  # type: ignore[attr-defined]
    return facets_subdomain


def TaylorHoodFunctionSpace(
    mesh: dolfinx.mesh.Mesh, family_degree: typing.Tuple[str, int]
) -> dolfinx.fem.FunctionSpaceBase:
    """Define a mixed function space."""
    (family, degree) = family_degree
    V_element = basix.ufl.element(family, mesh.basix_cell(), degree + 1, shape=(mesh.geometry.dim, ))
    Q_element = basix.ufl.element(family, mesh.basix_cell(), degree)
    taylor_hood_element = basix.ufl.mixed_element([V_element, Q_element])
    return dolfinx.fem.functionspace(mesh, taylor_hood_element)

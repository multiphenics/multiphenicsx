# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for multiphenicsx.io.plotting module."""

import dolfinx.mesh
import mpi4py
import numpy as np
import petsc4py
import pytest

import multiphenicsx.io


@pytest.fixture
def mesh_1d() -> dolfinx.mesh.Mesh:
    """Generate a unit interval mesh for use in tests in this file."""
    comm = mpi4py.MPI.COMM_WORLD
    return dolfinx.mesh.create_unit_interval(comm, 4 * comm.size)


@pytest.fixture
def mesh_2d() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    comm = mpi4py.MPI.COMM_WORLD
    return dolfinx.mesh.create_unit_square(comm, 2 * comm.size, 2 * comm.size)


def test_plot_mesh_1d(mesh_1d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_mesh executes without errors (1D case)."""
    multiphenicsx.io.plot_mesh(mesh_1d)


def test_plot_mesh_2d(mesh_2d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_mesh executes without errors (2D case)."""
    multiphenicsx.io.plot_mesh(mesh_2d)


def test_plot_mesh_entities_2d(mesh_2d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_mesh_entities executes without errors (2D mesh, 2D entities)."""
    cell_entities = dolfinx.mesh.locate_entities(
        mesh_2d, mesh_2d.topology.dim, lambda x: np.full((x.shape[1], ), True))
    multiphenicsx.io.plot_mesh_entities(mesh_2d, mesh_2d.topology.dim, cell_entities)


def test_plot_mesh_entities_boundary_2d(mesh_2d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_mesh_entities executes without errors (2D mesh, 1D entities)."""
    boundary_entities = dolfinx.mesh.locate_entities_boundary(
        mesh_2d, mesh_2d.topology.dim - 1, lambda x: np.full((x.shape[1], ), True))
    multiphenicsx.io.plot_mesh_entities(mesh_2d, mesh_2d.topology.dim - 1, boundary_entities)


def test_plot_mesh_tags_2d(mesh_2d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_mesh_tags executes without errors (2D mesh, 2D tags)."""
    cell_entities = dolfinx.mesh.locate_entities(
        mesh_2d, mesh_2d.topology.dim, lambda x: np.full((x.shape[1], ), True))
    cell_tags = dolfinx.mesh.MeshTags(
        mesh_2d, mesh_2d.topology.dim, cell_entities, np.ones_like(cell_entities))
    multiphenicsx.io.plot_mesh_tags(cell_tags)


def test_plot_mesh_tags_boundary_2d(mesh_2d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_mesh_tags executes without errors (2D mesh, 1D tags)."""
    boundary_entities = dolfinx.mesh.locate_entities_boundary(
        mesh_2d, mesh_2d.topology.dim - 1, lambda x: np.full((x.shape[1], ), True))
    boundary_tags = dolfinx.mesh.MeshTags(
        mesh_2d, mesh_2d.topology.dim - 1, boundary_entities, np.ones_like(boundary_entities))
    multiphenicsx.io.plot_mesh_tags(boundary_tags)


def test_plot_scalar_field_1d(mesh_1d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_scalar_field executes without errors (1D case)."""
    V = dolfinx.fem.FunctionSpace(mesh_1d, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    with u.vector.localForm() as u_local:
        u_local[:] = np.arange(u_local.size)
    if not np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating):
        multiphenicsx.io.plot_scalar_field(u, "u")
    else:
        multiphenicsx.io.plot_scalar_field(u, "u", part="real")
        multiphenicsx.io.plot_scalar_field(u, "u", part="imag")


def test_plot_scalar_field_2d(mesh_2d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_scalar_field executes without errors (2D case)."""
    V = dolfinx.fem.FunctionSpace(mesh_2d, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    with u.vector.localForm() as u_local:
        u_local[:] = np.arange(u_local.size)
    if not np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating):
        multiphenicsx.io.plot_scalar_field(u, "u")
    else:
        multiphenicsx.io.plot_scalar_field(u, "u", part="real")
        multiphenicsx.io.plot_scalar_field(u, "u", part="imag")
    multiphenicsx.io.plot_scalar_field(u, "u", warp_factor=1.0)


def test_plot_vector_field_2d(mesh_2d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_vector_field executes without errors (2D case)."""
    V = dolfinx.fem.VectorFunctionSpace(mesh_2d, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    with u.vector.localForm() as u_local:
        u_local[:] = np.arange(u_local.size)
    if not np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating):
        multiphenicsx.io.plot_vector_field(u, "u")
    else:
        multiphenicsx.io.plot_vector_field(u, "u", part="real")
        multiphenicsx.io.plot_vector_field(u, "u", part="imag")
    multiphenicsx.io.plot_vector_field(u, "u", glyph_factor=1.0)
    multiphenicsx.io.plot_vector_field(u, "u", warp_factor=1.0)

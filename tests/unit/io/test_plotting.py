# Copyright (C) 2016-2023 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for multiphenicsx.io.plotting module."""

import os
import typing

import dolfinx.fem
import dolfinx.mesh
import mpi4py.MPI
import nbvalx.tempfile
import numpy as np
import petsc4py.PETSc
import pytest
import ufl

import multiphenicsx.io

# Use go and pyvista from multiphenicsx.io.plotting rather than the actual libraries
# to handle the case of missing dependencies
go_or_mock = multiphenicsx.io.plotting.go
pyvista_or_mock = multiphenicsx.io.plotting.pyvista

ExpressionGeneratorType = typing.Callable[
    [dolfinx.fem.Function, dolfinx.fem.FunctionSpace],
    typing.Union[dolfinx.fem.Function, typing.Tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]]]


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


def write_plotly_image(
    comm: mpi4py.MPI.Intracomm, fig: go_or_mock.Figure, directory: str, filename: str  # type: ignore[name-defined]
) -> None:
    """Write a plotly figure to file."""
    pytest.importorskip("kaleido")
    fig.write_image(os.path.join(directory, filename + "-" + str(comm.rank) + ".png"))


def write_pyvista_image(
    comm: mpi4py.MPI.Intracomm, viewer: pyvista_or_mock.trame.jupyter.Widget,
    directory: str, filename: str
) -> None:
    """Write a pyvista figure to file."""
    # Currently untested. See #3887 in pyvista repo.
    pass


def test_plot_mesh_1d(mesh_1d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_mesh executes without errors (1D case)."""
    pytest.importorskip("plotly")
    with nbvalx.tempfile.TemporaryDirectory(mesh_1d.comm) as tempdir:
        write_plotly_image(mesh_1d.comm, multiphenicsx.io.plot_mesh(mesh_1d), tempdir, "mesh")


def test_plot_mesh_2d(mesh_2d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_mesh executes without errors (2D case)."""
    pytest.importorskip("pyvista")
    with nbvalx.tempfile.TemporaryDirectory(mesh_2d.comm) as tempdir:
        write_pyvista_image(mesh_2d.comm, multiphenicsx.io.plot_mesh(mesh_2d), tempdir, "mesh")


def test_plot_mesh_entities_2d(mesh_2d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_mesh_entities executes without errors (2D mesh, 2D entities)."""
    pytest.importorskip("pyvista")
    cell_entities = dolfinx.mesh.locate_entities(
        mesh_2d, mesh_2d.topology.dim, lambda x: np.full((x.shape[1], ), True))
    with nbvalx.tempfile.TemporaryDirectory(mesh_2d.comm) as tempdir:
        write_pyvista_image(
            mesh_2d.comm, multiphenicsx.io.plot_mesh_entities(mesh_2d, mesh_2d.topology.dim, cell_entities),
            tempdir, "mesh_entities")


def test_plot_mesh_entities_boundary_2d(mesh_2d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_mesh_entities executes without errors (2D mesh, 1D entities)."""
    pytest.importorskip("pyvista")
    boundary_entities = dolfinx.mesh.locate_entities_boundary(
        mesh_2d, mesh_2d.topology.dim - 1, lambda x: np.full((x.shape[1], ), True))
    with nbvalx.tempfile.TemporaryDirectory(mesh_2d.comm) as tempdir:
        write_pyvista_image(
            mesh_2d.comm, multiphenicsx.io.plot_mesh_entities(mesh_2d, mesh_2d.topology.dim - 1, boundary_entities),
            tempdir, "mesh_entities")


@pytest.mark.parametrize("fill_value", [0, 1])
def test_plot_mesh_tags_2d(mesh_2d: dolfinx.mesh.Mesh, fill_value: int) -> None:
    """Check that plot_mesh_tags executes without errors (2D mesh, 2D tags)."""
    pytest.importorskip("pyvista")
    cell_entities = dolfinx.mesh.locate_entities(
        mesh_2d, mesh_2d.topology.dim, lambda x: np.full((x.shape[1], ), True))
    cell_tags = dolfinx.mesh.meshtags(
        mesh_2d, mesh_2d.topology.dim, cell_entities,
        np.full(cell_entities.shape, fill_value=fill_value, dtype=np.int32))
    with nbvalx.tempfile.TemporaryDirectory(mesh_2d.comm) as tempdir:
        write_pyvista_image(mesh_2d.comm, multiphenicsx.io.plot_mesh_tags(cell_tags), tempdir, "mesh_tags")


@pytest.mark.parametrize("fill_value", [0, 1])
def test_plot_mesh_tags_boundary_2d(mesh_2d: dolfinx.mesh.Mesh, fill_value: int) -> None:
    """Check that plot_mesh_tags executes without errors (2D mesh, 1D tags)."""
    pytest.importorskip("pyvista")
    boundary_entities = dolfinx.mesh.locate_entities_boundary(
        mesh_2d, mesh_2d.topology.dim - 1, lambda x: np.full((x.shape[1], ), True))
    boundary_tags = dolfinx.mesh.meshtags(
        mesh_2d, mesh_2d.topology.dim - 1, boundary_entities,
        np.full(boundary_entities.shape, fill_value=fill_value, dtype=np.int32))
    if fill_value == 0:
        with pytest.raises(AssertionError) as excinfo:
            multiphenicsx.io.plot_mesh_tags(boundary_tags)
        assert str(excinfo.value) == "Zero is used as a placeholder for non-provided entities"
    elif fill_value == 1:
        with nbvalx.tempfile.TemporaryDirectory(mesh_2d.comm) as tempdir:
            write_pyvista_image(mesh_2d.comm, multiphenicsx.io.plot_mesh_tags(boundary_tags), tempdir, "mesh_tags")


@pytest.mark.parametrize("expression_generator", [lambda u, V: u, lambda u, V: (2.0 * u, V)])
def test_plot_scalar_field_1d(  # type: ignore[no-any-unimported]
    mesh_1d: dolfinx.mesh.Mesh, expression_generator: ExpressionGeneratorType
) -> None:
    """Check that plot_scalar_field executes without errors (1D case)."""
    pytest.importorskip("plotly")
    V = dolfinx.fem.FunctionSpace(mesh_1d, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    with u.vector.localForm() as u_local:
        u_local[:] = np.arange(u_local.size)
    with nbvalx.tempfile.TemporaryDirectory(mesh_1d.comm) as tempdir:
        if not np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating):
            write_plotly_image(
                mesh_1d.comm, multiphenicsx.io.plot_scalar_field(expression_generator(u, V), "u"), tempdir, "u")
        else:
            write_plotly_image(
                mesh_1d.comm, multiphenicsx.io.plot_scalar_field(expression_generator(u, V), "u", part="real"),
                tempdir, "real_u")
            write_plotly_image(
                mesh_1d.comm, multiphenicsx.io.plot_scalar_field(expression_generator(u, V), "u", part="imag"),
                tempdir, "imag_u")


@pytest.mark.parametrize("expression_generator", [lambda u, V: u, lambda u, V: (2.0 * u, V)])
def test_plot_scalar_field_2d(  # type: ignore[no-any-unimported]
    mesh_2d: dolfinx.mesh.Mesh, expression_generator: ExpressionGeneratorType
) -> None:
    """Check that plot_scalar_field executes without errors (2D case)."""
    pytest.importorskip("pyvista")
    V = dolfinx.fem.FunctionSpace(mesh_2d, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    with u.vector.localForm() as u_local:
        u_local[:] = np.arange(u_local.size)
    with nbvalx.tempfile.TemporaryDirectory(mesh_2d.comm) as tempdir:
        if not np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating):
            write_pyvista_image(
                mesh_2d.comm, multiphenicsx.io.plot_scalar_field(expression_generator(u, V), "u"), tempdir, "u")
        else:
            write_pyvista_image(
                mesh_2d.comm, multiphenicsx.io.plot_scalar_field(expression_generator(u, V), "u", part="real"),
                tempdir, "real_u")
            write_pyvista_image(
                mesh_2d.comm, multiphenicsx.io.plot_scalar_field(expression_generator(u, V), "u", part="imag"),
                tempdir, "imag_u")
        write_pyvista_image(
            mesh_2d.comm, multiphenicsx.io.plot_scalar_field(expression_generator(u, V), "u", warp_factor=1.0),
            tempdir, "glyph_u")


@pytest.mark.parametrize("expression_generator", [lambda u, V: u, lambda u, V: (2.0 * u, V)])
def test_plot_vector_field_2d(  # type: ignore[no-any-unimported]
    mesh_2d: dolfinx.mesh.Mesh, expression_generator: ExpressionGeneratorType
) -> None:
    """Check that plot_vector_field executes without errors (2D case)."""
    pytest.importorskip("pyvista")
    V = dolfinx.fem.VectorFunctionSpace(mesh_2d, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    with u.vector.localForm() as u_local:
        u_local[:] = np.arange(u_local.size)
    with nbvalx.tempfile.TemporaryDirectory(mesh_2d.comm) as tempdir:
        if not np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating):
            write_pyvista_image(
                mesh_2d.comm, multiphenicsx.io.plot_vector_field(expression_generator(u, V), "u"), tempdir, "u")
        else:
            write_pyvista_image(
                mesh_2d.comm, multiphenicsx.io.plot_vector_field(expression_generator(u, V), "u", part="real"),
                tempdir, "real_u")
            write_pyvista_image(
                mesh_2d.comm, multiphenicsx.io.plot_vector_field(expression_generator(u, V), "u", part="imag"),
                tempdir, "imag_u")
        write_pyvista_image(
            mesh_2d.comm, multiphenicsx.io.plot_vector_field(expression_generator(u, V), "u", glyph_factor=1.0),
            tempdir, "glyph_u")
        write_pyvista_image(
            mesh_2d.comm, multiphenicsx.io.plot_vector_field(expression_generator(u, V), "u", warp_factor=1.0),
            tempdir, "warp_u")

# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for multiphenicsx.io.plotting module."""

import os

import dolfinx.mesh
import mpi4py
import nbvalx.tempfile
import numpy as np
import petsc4py
import pytest

import multiphenicsx.io

# Use go and itkwidgets from multiphenicsx.io.plotting rather than the actual libraries
# to handle the case of missing dependencies
go_or_mock = multiphenicsx.io.plotting.go
itkwidgets_or_mock = multiphenicsx.io.plotting.itkwidgets


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


def write_plotly_image(comm: mpi4py.MPI.Intracomm, fig: go_or_mock.Figure, directory: str, filename: str) -> None:
    """Write a plotly figure to file."""
    pytest.importorskip("kaleido")
    fig.write_image(os.path.join(directory, filename + "-" + str(comm.rank) + ".png"))


def write_itkwidgets_image(
    comm: mpi4py.MPI.Intracomm, viewer: itkwidgets_or_mock.Viewer, directory: str, filename: str
) -> None:
    """Write a itkwidgets figure to file."""
    # Currently untested. itkwidgets suggests to use ipywebrtc, but it does not seem to be working
    # when called from plain python files (error on no data to export) rather than jupyter notebooks
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
        write_itkwidgets_image(mesh_2d.comm, multiphenicsx.io.plot_mesh(mesh_2d), tempdir, "mesh")


def test_plot_mesh_entities_2d(mesh_2d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_mesh_entities executes without errors (2D mesh, 2D entities)."""
    pytest.importorskip("pyvista")
    cell_entities = dolfinx.mesh.locate_entities(
        mesh_2d, mesh_2d.topology.dim, lambda x: np.full((x.shape[1], ), True))
    with nbvalx.tempfile.TemporaryDirectory(mesh_2d.comm) as tempdir:
        write_itkwidgets_image(
            mesh_2d.comm, multiphenicsx.io.plot_mesh_entities(mesh_2d, mesh_2d.topology.dim, cell_entities),
            tempdir, "mesh_entities")


def test_plot_mesh_entities_boundary_2d(mesh_2d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_mesh_entities executes without errors (2D mesh, 1D entities)."""
    pytest.importorskip("pyvista")
    boundary_entities = dolfinx.mesh.locate_entities_boundary(
        mesh_2d, mesh_2d.topology.dim - 1, lambda x: np.full((x.shape[1], ), True))
    with nbvalx.tempfile.TemporaryDirectory(mesh_2d.comm) as tempdir:
        write_itkwidgets_image(
            mesh_2d.comm, multiphenicsx.io.plot_mesh_entities(mesh_2d, mesh_2d.topology.dim - 1, boundary_entities),
            tempdir, "mesh_entities")


def test_plot_mesh_tags_2d(mesh_2d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_mesh_tags executes without errors (2D mesh, 2D tags)."""
    pytest.importorskip("pyvista")
    cell_entities = dolfinx.mesh.locate_entities(
        mesh_2d, mesh_2d.topology.dim, lambda x: np.full((x.shape[1], ), True))
    cell_tags = dolfinx.mesh.MeshTags(
        mesh_2d, mesh_2d.topology.dim, cell_entities, np.ones_like(cell_entities))
    with nbvalx.tempfile.TemporaryDirectory(mesh_2d.comm) as tempdir:
        write_itkwidgets_image(mesh_2d.comm, multiphenicsx.io.plot_mesh_tags(cell_tags), tempdir, "mesh_tags")


def test_plot_mesh_tags_boundary_2d(mesh_2d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_mesh_tags executes without errors (2D mesh, 1D tags)."""
    pytest.importorskip("pyvista")
    boundary_entities = dolfinx.mesh.locate_entities_boundary(
        mesh_2d, mesh_2d.topology.dim - 1, lambda x: np.full((x.shape[1], ), True))
    boundary_tags = dolfinx.mesh.MeshTags(
        mesh_2d, mesh_2d.topology.dim - 1, boundary_entities, np.ones_like(boundary_entities))
    with nbvalx.tempfile.TemporaryDirectory(mesh_2d.comm) as tempdir:
        write_itkwidgets_image(mesh_2d.comm, multiphenicsx.io.plot_mesh_tags(boundary_tags), tempdir, "mesh_tags")


def test_plot_scalar_field_1d(mesh_1d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_scalar_field executes without errors (1D case)."""
    pytest.importorskip("plotly")
    V = dolfinx.fem.FunctionSpace(mesh_1d, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    with u.vector.localForm() as u_local:
        u_local[:] = np.arange(u_local.size)
    with nbvalx.tempfile.TemporaryDirectory(mesh_1d.comm) as tempdir:
        if not np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating):
            write_plotly_image(mesh_1d.comm, multiphenicsx.io.plot_scalar_field(u, "u"), tempdir, "u")
        else:
            write_plotly_image(
                mesh_1d.comm, multiphenicsx.io.plot_scalar_field(u, "u", part="real"), tempdir, "real_u")
            write_plotly_image(
                mesh_1d.comm, multiphenicsx.io.plot_scalar_field(u, "u", part="imag"), tempdir, "imag_u")


def test_plot_scalar_field_2d(mesh_2d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_scalar_field executes without errors (2D case)."""
    pytest.importorskip("pyvista")
    V = dolfinx.fem.FunctionSpace(mesh_2d, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    with u.vector.localForm() as u_local:
        u_local[:] = np.arange(u_local.size)
    with nbvalx.tempfile.TemporaryDirectory(mesh_2d.comm) as tempdir:
        if not np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating):
            write_itkwidgets_image(mesh_2d.comm, multiphenicsx.io.plot_scalar_field(u, "u"), tempdir, "u")
        else:
            write_itkwidgets_image(
                mesh_2d.comm, multiphenicsx.io.plot_scalar_field(u, "u", part="real"), tempdir, "real_u")
            write_itkwidgets_image(
                mesh_2d.comm, multiphenicsx.io.plot_scalar_field(u, "u", part="imag"), tempdir, "imag_u")
        write_itkwidgets_image(
            mesh_2d.comm, multiphenicsx.io.plot_scalar_field(u, "u", warp_factor=1.0), tempdir, "glyph_u")


def test_plot_vector_field_2d(mesh_2d: dolfinx.mesh.Mesh) -> None:
    """Check that plot_vector_field executes without errors (2D case)."""
    pytest.importorskip("pyvista")
    V = dolfinx.fem.VectorFunctionSpace(mesh_2d, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    with u.vector.localForm() as u_local:
        u_local[:] = np.arange(u_local.size)
    with nbvalx.tempfile.TemporaryDirectory(mesh_2d.comm) as tempdir:
        if not np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating):
            write_itkwidgets_image(mesh_2d.comm, multiphenicsx.io.plot_vector_field(u, "u"), tempdir, "u")
        else:
            write_itkwidgets_image(
                mesh_2d.comm, multiphenicsx.io.plot_vector_field(u, "u", part="real"), tempdir, "real_u")
            write_itkwidgets_image(
                mesh_2d.comm, multiphenicsx.io.plot_vector_field(u, "u", part="imag"), tempdir, "imag_u")
        write_itkwidgets_image(
            mesh_2d.comm, multiphenicsx.io.plot_vector_field(u, "u", glyph_factor=1.0), tempdir, "glyph_u")
        write_itkwidgets_image(
            mesh_2d.comm, multiphenicsx.io.plot_vector_field(u, "u", warp_factor=1.0), tempdir, "warp_u")

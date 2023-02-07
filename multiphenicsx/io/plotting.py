# Copyright (C) 2016-2023 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Utilities for plotting dolfinx objects with plotly and pyvista."""

import types
import typing

import dolfinx.fem
import dolfinx.mesh
import dolfinx.plot
import numpy as np
import numpy.typing
import petsc4py.PETSc
import ufl

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = types.ModuleType("go", "Mock plotly.graph_objects module")
    go.Figure = object

try:
    import pyvista
except ImportError:  # pragma: no cover
    pyvista = types.ModuleType("pyvista", "Mock pyvista module")
    pyvista.UnstructuredGrid = object  # type: ignore[misc, assignment]
    pyvista.trame = types.ModuleType("pyvista", "Mock pyvista.trame module")
    pyvista.trame.jupyter = types.ModuleType("pyvista", "Mock pyvista.trame.jupyter module")
    pyvista.trame.jupyter.Widget = object  # type: ignore[misc, assignment]
else:
    import pyvista.trame
    import pyvista.trame.jupyter


def _dolfinx_to_pyvista_mesh(mesh: dolfinx.mesh.Mesh, dim: typing.Optional[int] = None) -> pyvista.UnstructuredGrid:
    if dim is None:
        dim = mesh.topology.dim
    mesh.topology.create_connectivity(dim, dim)
    num_cells = mesh.topology.index_map(dim).size_local + mesh.topology.index_map(dim).num_ghosts
    cell_entities = np.arange(num_cells, dtype=np.int32)
    pyvista_cells, cell_types, coordinates = dolfinx.plot.create_vtk_mesh(
        mesh, dim, cell_entities)
    return pyvista.UnstructuredGrid(pyvista_cells, cell_types, coordinates)


def plot_mesh(mesh: dolfinx.mesh.Mesh) -> typing.Union[  # type: ignore[no-any-unimported]
        go.Figure, pyvista.trame.jupyter.Widget]:
    """
    Plot a dolfinx.mesh.Mesh with plotly (in 1D) or pyvista (in 2D or 3D).

    Parameters
    ----------
    mesh
        Mesh to be plotted.

    Returns
    -------
    :
        A plotly.graph_objects.Figure (in 1D) or pyvista.trame.jupyter.Widget (in 2D or 3D)
        representing a plot of the mesh.
    """
    if mesh.topology.dim == 1:
        return _plot_mesh_plotly(mesh)
    else:
        return _plot_mesh_pyvista(mesh)


def _plot_mesh_plotly(mesh: dolfinx.mesh.Mesh) -> go.Figure:  # type: ignore[no-any-unimported]
    fig = go.Figure(data=go.Scatter(
        x=mesh.geometry.x[:, 0], y=np.zeros(mesh.geometry.x.shape[0]),
        line=dict(color="blue", width=2, dash="solid"),
        marker=dict(color="blue", size=10),
        mode="lines+markers"))
    fig.update_xaxes(title_text="x")
    return fig


def _plot_mesh_pyvista(mesh: dolfinx.mesh.Mesh) -> pyvista.trame.jupyter.Widget:
    grid = _dolfinx_to_pyvista_mesh(mesh)
    plotter = pyvista.Plotter(notebook=True)  # type: ignore[no-untyped-call]
    plotter.add_mesh(grid)  # type: ignore[no-untyped-call]
    return plotter.show(jupyter_backend="client", return_viewer=True)  # type: ignore[no-any-return, no-untyped-call]


def plot_mesh_entities(
    mesh: dolfinx.mesh.Mesh, dim: int, entities: np.typing.NDArray[np.int32]
) -> pyvista.trame.jupyter.Widget:
    """
    Plot dolfinx.mesh.Mesh with pyvista, highlighting the provided `dim`-dimensional entities.

    Parameters
    ----------
    mesh
        Mesh to be plotted. Current implementation is limited to 2D or 3D meshes.
    dim
        Dimension of the entities
    entities
        Array containing the IDs of the entities to be highlighted.

    Returns
    -------
    :
        An pyvista.trame.jupyter.Widget representing a plot of the mesh entities.
    """
    assert mesh.topology.dim > 1
    return _plot_mesh_entities_pyvista(mesh, dim, entities, np.ones_like(entities))


def plot_mesh_tags(mesh_tags: dolfinx.mesh.MeshTags) -> pyvista.trame.jupyter.Widget:
    """
    Plot dolfinx.mesh.MeshTags with pyvista.

    Parameters
    ----------
    mesh
        MeshTags to be plotted. Current implementation is limited to 2D or 3D underlying meshes.

    Returns
    -------
    :
        An pyvista.trame.jupyter.Widget representing a plot of the dolfinx.mesh.MeshTags object.
    """
    mesh = mesh_tags.mesh
    assert mesh.topology.dim > 1
    return _plot_mesh_entities_pyvista(mesh, mesh_tags.dim, mesh_tags.indices, mesh_tags.values)


def _plot_mesh_entities_pyvista(
    mesh: dolfinx.mesh.Mesh, dim: int, indices: np.typing.NDArray[np.int32], values: np.typing.NDArray[np.int32]
) -> pyvista.trame.jupyter.Widget:
    num_cells = mesh.topology.index_map(dim).size_local + mesh.topology.index_map(dim).num_ghosts
    all_values = np.zeros(num_cells)
    if values.shape[0] != num_cells:
        assert np.all(values != 0), "Zero is used as a placeholder for non-provided entities"
    for (index, value) in zip(indices, values):
        all_values[index] = value

    if dim == mesh.topology.dim:
        name = "Subdomains"
    elif dim == mesh.topology.dim - 1:
        name = "Boundaries"
    grid = _dolfinx_to_pyvista_mesh(mesh, dim)
    grid.cell_data[name] = all_values
    grid.set_active_scalars(name)
    plotter = pyvista.Plotter(notebook=True)  # type: ignore[no-untyped-call]
    plotter.add_mesh(grid)  # type: ignore[no-untyped-call]
    return plotter.show(jupyter_backend="client", return_viewer=True)  # type: ignore[no-any-return, no-untyped-call]


def plot_scalar_field(  # type: ignore[no-any-unimported]
    scalar_field: typing.Union[dolfinx.fem.Function, typing.Tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]],
    name: str, warp_factor: float = 0.0, part: str = "real"
) -> typing.Union[go.Figure, pyvista.trame.jupyter.Widget]:
    """
    Plot a scalar field with plotly (in 1D) or pyvista (in 2D or 3D).

    Parameters
    ----------
    scalar_field
        Expression to be plotted, which contains a scalar field.
        If the expression is provided as a dolfinx Function, such function will be plotted.
        If the expression is provided as a tuple containing UFL expression and a dolfinx FunctionSpace,
        the UFL expression will first be interpolated on the function space and then plotted.
    name
        Name of the quantity stored in the scalar field.
    warp_factor
        This argument is ignored for a field on a 1D mesh.
        For a 2D mesh: if provided then the factor is used to produce a warped representation
        the field; if not provided then the scalar field will be plotted on the mesh.
    part
        Part of the solution (real or imag) to be plotted. By default, the real part is plotted.
        The argument is ignored when plotting a real field.

    Returns
    -------
    :
        A plotly.graph_objects.Figure (in 1D) or pyvista.trame.jupyter.Widget (in 2D or 3D)
        representing a plot of the scalar field.
    """
    scalar_field = _interpolate_if_ufl_expression(scalar_field)
    mesh = scalar_field.function_space.mesh
    if mesh.topology.dim == 1:
        return _plot_scalar_field_plotly(scalar_field, name, part)
    else:
        return _plot_scalar_field_pyvista(scalar_field, name, warp_factor, part)


def _plot_scalar_field_plotly(  # type: ignore[no-any-unimported]
    scalar_field: dolfinx.fem.Function, name: str, part: str
) -> go.Figure:
    values = scalar_field.x.array
    values, name = _extract_part(values, name, part)
    coordinates = scalar_field.function_space.tabulate_dof_coordinates()
    coordinates = coordinates[:, 0]
    argsort = coordinates.argsort()
    coordinates = coordinates[argsort]
    values = values[argsort]
    fig = go.Figure(data=go.Scatter(
        x=coordinates, y=values,
        line=dict(color="blue", width=2, dash="solid"),
        marker=dict(color="blue", size=10),
        mode="lines+markers"))
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text=name)
    return fig


def _plot_scalar_field_pyvista(
    scalar_field: dolfinx.fem.Function, name: str, warp_factor: float, part: str
) -> pyvista.trame.jupyter.Widget:
    values = scalar_field.x.array
    values, name = _extract_part(values, name, part)
    pyvista_cells, cell_types, coordinates = dolfinx.plot.create_vtk_mesh(scalar_field.function_space)
    grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, coordinates)
    grid.point_data[name] = values
    grid.set_active_scalars(name)
    plotter = pyvista.Plotter(notebook=True)  # type: ignore[no-untyped-call]
    if warp_factor != 0.0:
        assert warp_factor > 0.0
        warped = grid.warp_by_scalar(factor=warp_factor)  # type: ignore[no-untyped-call]
        plotter.add_mesh(warped)  # type: ignore[no-untyped-call]
    else:
        plotter.add_mesh(grid)  # type: ignore[no-untyped-call]
    return plotter.show(jupyter_backend="client", return_viewer=True)  # type: ignore[no-any-return, no-untyped-call]


def plot_vector_field(  # type: ignore[no-any-unimported]
    vector_field: typing.Union[dolfinx.fem.Function, typing.Tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]],
    name: str, glyph_factor: float = 0.0, warp_factor: float = 0.0,
    part: str = "real"
) -> pyvista.trame.jupyter.Widget:
    """
    Plot a vector field with pyvista.

    Parameters
    ----------
    vector_field
        Expression to be plotted, which contains a vector field.
        If the expression is provided as a dolfinx Function, such function will be plotted.
        If the expression is provided as a tuple containing UFL expression and a dolfinx FunctionSpace,
        the UFL expression will first be interpolated on the function space and then plotted.
    name
        Name of the quantity stored in the vector field.
    glyph_factor
        If provided, the vector field is represented using a gylph, scaled by this factor.
    warp_factor
        If provided then the factor is used to produce a warped representation
        the field; if not provided then the magnitude of the vector field will be plotted on the mesh.
        Only used when `glyph_factor` is not provided.
    part
        Part of the solution (real or imag) to be plotted. By default, the real part is plotted.
        The argument is ignored when plotting a real field.

    Returns
    -------
    :
        An pyvista.trame.jupyter.Widget representing a plot of the vector field.
    """
    vector_field = _interpolate_if_ufl_expression(vector_field)
    mesh = vector_field.function_space.mesh
    assert mesh.topology.dim > 1
    return _plot_vector_field_pyvista(vector_field, name, glyph_factor, warp_factor, part)


def _plot_vector_field_pyvista(
    vector_field: dolfinx.fem.Function, name: str, glyph_factor: float,
    warp_factor: float, part: str
) -> pyvista.trame.jupyter.Widget:
    pyvista_cells, cell_types, coordinates = dolfinx.plot.create_vtk_mesh(vector_field.function_space)
    values = vector_field.x.array.reshape(coordinates.shape[0], vector_field.function_space.dofmap.index_map_bs)
    values, name = _extract_part(values, name, part)
    if values.shape[1] == 2:
        values = np.insert(values, values.shape[1], 0.0, axis=1)
    grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, coordinates)
    grid.point_data[name] = values
    plotter = pyvista.Plotter(notebook=True)  # type: ignore[no-untyped-call]
    if glyph_factor == 0.0:
        grid.set_active_vectors(name)
        if warp_factor == 0.0:
            plotter.add_mesh(grid)  # type: ignore[no-untyped-call]
        else:
            assert warp_factor > 0.0
            warped = grid.warp_by_vector(factor=warp_factor)  # type: ignore[no-untyped-call]
            plotter.add_mesh(warped)  # type: ignore[no-untyped-call]
    else:
        assert glyph_factor > 0.0
        assert warp_factor == 0.0
        glyphs = grid.glyph(orient=name, factor=glyph_factor)  # type: ignore[no-untyped-call]
        plotter.add_mesh(glyphs)  # type: ignore[no-untyped-call]
        mesh = vector_field.function_space.mesh
        grid_background = _dolfinx_to_pyvista_mesh(mesh, mesh.topology.dim - 1)
        plotter.add_mesh(grid_background)  # type: ignore[no-untyped-call]
    return plotter.show(jupyter_backend="client", return_viewer=True)  # type: ignore[no-any-return, no-untyped-call]


def _interpolate_if_ufl_expression(  # type: ignore[no-any-unimported]
    field: typing.Union[dolfinx.fem.Function, typing.Tuple[ufl.core.expr.Expr, dolfinx.fem.FunctionSpace]]
) -> dolfinx.fem.Function:
    if isinstance(field, tuple):
        expression, function_space = field
        interpolated_field = dolfinx.fem.Function(function_space)
        interpolated_field.interpolate(
            dolfinx.fem.Expression(expression, function_space.element.interpolation_points()))
        return interpolated_field
    else:
        assert isinstance(field, dolfinx.fem.Function)
        return field


def _extract_part(  # type: ignore[no-any-unimported]
    values: np.typing.NDArray[petsc4py.PETSc.ScalarType], name: str, part: str
) -> typing.Tuple[np.typing.NDArray[np.float64], str]:
    if np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating):  # pragma: no cover
        if part == "real":
            values = values.real
            name = "real(" + name + ")"
        elif part == "imag":
            values = values.imag
            name = "imag(" + name + ")"
    return values, name

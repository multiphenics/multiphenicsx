# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Utilities to convert a gmsh model to dolfinx."""


import typing

import dolfinx.cpp
import dolfinx.io
import dolfinx.mesh
import mpi4py
import numpy as np

try:
    import gmsh
except ImportError:  # pragma: no cover
    pass


def gmsh_to_fenicsx(model: gmsh.model, gdim: int) -> typing.Tuple[
        dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags]:
    """
    Given a GMSH model, create a DOLFINx mesh and MeshTags.

    Parameters
    ----------
    model : gmsh.model
        The GMSH model.
    gdim: int
        Geometrical dimension of problem.

    Notes
    -----
    Adapted from [1]_.

    References
    ----------
    .. [1] J.S. Dokken, http://jsdokken.com/converted_files/tutorial_gmsh.html
    """
    if mpi4py.MPI.COMM_WORLD.rank == 0:
        # Get mesh geometry
        x = dolfinx.io.extract_gmsh_geometry(model)

        # Get mesh topology for each element
        topologies = dolfinx.io.extract_gmsh_topology_and_markers(model)

        # Get information about each cell type from the msh files
        num_cell_types = len(topologies.keys())
        cell_information = {}
        cell_dimensions = np.zeros(num_cell_types, dtype=np.int32)
        for i, element in enumerate(topologies.keys()):
            properties = model.mesh.getElementProperties(element)
            name, dim, order, num_nodes, coords, _ = properties
            cell_information[i] = {"id": element, "dim": dim,
                                   "num_nodes": num_nodes}
            cell_dimensions[i] = dim

        # Sort elements by ascending dimension
        perm_sort = np.argsort(cell_dimensions)

        # Get cell type data and geometric dimension
        cell_id = cell_information[perm_sort[-1]]["id"]
        tdim = cell_information[perm_sort[-1]]["dim"]
        num_nodes = cell_information[perm_sort[-1]]["num_nodes"]
        cells = np.asarray(topologies[cell_id]["topology"], dtype=np.int64)
        cell_values = np.asarray(topologies[cell_id]["cell_data"], dtype=np.int32)
        cell_id, num_nodes = mpi4py.MPI.COMM_WORLD.bcast([cell_id, num_nodes], root=0)

        # Look up facet data
        assert tdim - 1 in cell_dimensions
        num_facet_nodes = mpi4py.MPI.COMM_WORLD.bcast(
            cell_information[perm_sort[-2]]["num_nodes"], root=0)
        gmsh_facet_id = cell_information[perm_sort[-2]]["id"]
        marked_facets = np.asarray(topologies[gmsh_facet_id]["topology"], dtype=np.int64)
        facet_values = np.asarray(topologies[gmsh_facet_id]["cell_data"], dtype=np.int32)
    else:
        cell_id, num_nodes = mpi4py.MPI.COMM_WORLD.bcast([None, None], root=0)
        cells, x = np.empty([0, num_nodes], dtype=np.int32), np.empty([0, gdim])
        cell_values = np.empty((0,), dtype=np.int32)
        num_facet_nodes = mpi4py.MPI.COMM_WORLD.bcast(None, root=0)
        marked_facets = np.empty((0, num_facet_nodes), dtype=np.int64)
        facet_values = np.empty((0,), dtype=np.int32)

    # Create distributed mesh
    ufl_domain = dolfinx.io.ufl_mesh_from_gmsh(cell_id, gdim)
    gmsh_cell_perm = dolfinx.cpp.io.perm_gmsh(
        dolfinx.cpp.mesh.to_type(str(ufl_domain.ufl_cell())), num_nodes)
    cells = cells[:, gmsh_cell_perm]
    mesh = dolfinx.mesh.create_mesh(mpi4py.MPI.COMM_WORLD, cells, x[:, :gdim], ufl_domain)

    # Create MeshTags for cells
    entities, values = dolfinx.cpp.io.distribute_entity_data(
        mesh, mesh.topology.dim, cells, cell_values)
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    adj = dolfinx.cpp.graph.AdjacencyList_int32(entities)
    ct = dolfinx.mesh.create_meshtags(
        mesh, mesh.topology.dim, adj, np.int32(values))
    ct.name = "subdomains"

    # Create MeshTags for facets
    facet_type = dolfinx.cpp.mesh.cell_entity_type(
        dolfinx.cpp.mesh.to_type(str(ufl_domain.ufl_cell())), mesh.topology.dim - 1, 0)
    gmsh_facet_perm = dolfinx.cpp.io.perm_gmsh(facet_type, num_facet_nodes)
    marked_facets = marked_facets[:, gmsh_facet_perm]
    entities, values = dolfinx.cpp.io.distribute_entity_data(
        mesh, mesh.topology.dim - 1, marked_facets, facet_values)
    mesh.topology.create_connectivity(
        mesh.topology.dim - 1, mesh.topology.dim)
    adj = dolfinx.cpp.graph.AdjacencyList_int32(entities)
    ft = dolfinx.mesh.create_meshtags(
        mesh, mesh.topology.dim - 1, adj, np.int32(values))
    ft.name = "boundaries"

    return mesh, ct, ft

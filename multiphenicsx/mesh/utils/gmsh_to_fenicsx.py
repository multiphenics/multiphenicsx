# Copyright (C) 2016-2021 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
from mpi4py import MPI
from dolfinx.io import extract_gmsh_geometry, extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh
from dolfinx.cpp.io import distribute_entity_data, perm_gmsh
from dolfinx.cpp.mesh import cell_entity_type, to_type
from dolfinx.cpp.graph import AdjacencyList_int32
from dolfinx.mesh import create_mesh, create_meshtags


def gmsh_to_fenicsx(model, gdim):
    """
    Given a GMSH model, create a DOLFINx mesh and MeshTags.

    Parameters
    ----------
    model : gmsh.model
        The GMSH model.
    gdim: int
        Geometrical dimension of problem.

    Author
    ----------
    J. S. Dokken, http://jsdokken.com/converted_files/tutorial_gmsh.html
    """

    assert MPI.COMM_WORLD.size == 1, "This function has been simplified to the case of serial computations"

    # Get mesh geometry
    x = extract_gmsh_geometry(model)

    # Get mesh topology for each element
    topologies = extract_gmsh_topology_and_markers(model)

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

    # Look up facet data
    assert tdim - 1 in cell_dimensions
    num_facet_nodes = cell_information[perm_sort[-2]]["num_nodes"]
    gmsh_facet_id = cell_information[perm_sort[-2]]["id"]
    marked_facets = np.asarray(topologies[gmsh_facet_id]["topology"], dtype=np.int64)
    facet_values = np.asarray(topologies[gmsh_facet_id]["cell_data"], dtype=np.int32)

    # Create distributed mesh
    ufl_domain = ufl_mesh_from_gmsh(cell_id, gdim)
    gmsh_cell_perm = perm_gmsh(to_type(str(ufl_domain.ufl_cell())), num_nodes)
    cells = cells[:, gmsh_cell_perm]
    mesh = create_mesh(MPI.COMM_WORLD, cells, x[:, :gdim], ufl_domain)

    # Create MeshTags for cells
    entities, values = distribute_entity_data(
        mesh, mesh.topology.dim, cells, cell_values)
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    adj = AdjacencyList_int32(entities)
    ct = create_meshtags(mesh, mesh.topology.dim,
                         adj, np.int32(values))
    ct.name = "subdomains"

    # Create MeshTags for facets
    facet_type = cell_entity_type(to_type(str(ufl_domain.ufl_cell())),
                                  mesh.topology.dim - 1, 0)
    gmsh_facet_perm = perm_gmsh(facet_type, num_facet_nodes)
    marked_facets = marked_facets[:, gmsh_facet_perm]

    entities, values = distribute_entity_data(
        mesh, mesh.topology.dim - 1, marked_facets, facet_values)
    mesh.topology.create_connectivity(
        mesh.topology.dim - 1, mesh.topology.dim)
    adj = AdjacencyList_int32(entities)
    ft = create_meshtags(mesh, mesh.topology.dim - 1,
                         adj, np.int32(values))
    ft.name = "boundaries"

    return mesh, ct, ft

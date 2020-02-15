# Copyright (C) 2016-2020 by the multiphenics authors
#
# This file is part of multiphenics.
#
# multiphenics is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

from multiphenics.mesh import MeshRestriction

# Helper function to generate subdomain restriction based on a subdomain id
def generate_subdomain_restriction(mesh, subdomains, subdomain_id, restriction=None):
    D = mesh.topology.dim
    # Initialize empty restriction, if not provided as input
    if restriction is None:
        restriction = MeshRestriction(mesh)
    # Mark restriction mesh functions based on subdomain id
    [mesh.create_connectivity(D, d) for d in range(D)]
    connectivity = [mesh.topology.connectivity(D, d) for d in range(D)]
    for c in range(mesh.num_cells()):
        if subdomains.values[c] == subdomain_id:
            restriction[D].values[c] = 1
            for d in range(D):
                for e in connectivity[d].links(c):
                    restriction[d].values[e] = 1
    # Return
    return restriction

# Helper function to generate boundary restriction based on a boundary id
def generate_boundary_restriction(mesh, boundaries, boundary_id, restriction=None):
    D = mesh.topology.dim
    # Initialize empty restriction, if not provided as input
    if restriction is None:
        restriction = MeshRestriction(mesh)
    # Mark restriction mesh functions based on boundary id
    [mesh.create_connectivity(D - 1, d) for d in range(D - 1)]
    connectivity = [mesh.topology.connectivity(D - 1, d) for d in range(D - 1)]
    for f in range(mesh.num_entities(D - 1)):
        if boundaries.values[f] == boundary_id:
            restriction[D - 1].values[f] = 1
            for d in range(D - 1):
                for e in connectivity[d].links(f):
                    restriction[d].values[e] = 1
    # Return
    return restriction

# Helper function to generate interface restriction based on a pair of neighboring subdomain ids
def generate_interface_restriction(mesh, subdomains, subdomain_ids, restriction=None):
    assert isinstance(subdomain_ids, set)
    assert len(subdomain_ids) == 2
    D = mesh.topology.dim
    # Initialize empty restriction, if not provided as input
    if restriction is None:
        restriction = MeshRestriction(mesh)
    # Mark restriction mesh functions based on subdomain ids (except the mesh function corresponding to dimension D, as it is trivially false)
    [mesh.create_connectivity(D - 1, d) for d in range(D + 1)]
    connectivity = [mesh.topology.connectivity(D - 1, d) for d in range(D + 1)]
    for f in range(mesh.num_entities(D - 1)):
        subdomains_ids_f = set(subdomains.values[c] for c in connectivity[D].links(f))
        assert len(subdomains_ids_f) in (1, 2)
        if subdomains_ids_f == subdomain_ids:
            restriction[D - 1].values[f] = 1
            for d in range(D - 1):
                for e in connectivity[d].links(f):
                    restriction[d].values[e] = 1
    # Return
    return restriction

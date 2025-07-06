# Copyright (C) 2016-2025 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for multiphenicsx.fem.petsc module, solver classes."""

import typing

import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import mpi4py.MPI
import numpy as np
import petsc4py.PETSc
import pytest
import ufl

import multiphenicsx.fem
import multiphenicsx.fem.petsc

import common  # isort: skip

petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}

@pytest.fixture
def mesh() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    return dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, 4, 4)


def get_subdomains() -> tuple[typing.Optional[common.SubdomainType], ...]:
    """Generate subdomain parametrization for tests on vectors and matrices."""
    return (
        # Unrestricted
        None,
        # Restricted
        common.CellsSubDomain(0.5, 0.75)
    )


def active_measure(  # type: ignore[no-any-unimported]
    mesh: dolfinx.mesh.Mesh, subdomain: typing.Optional[common.SubdomainType]
) -> ufl.Measure:
    """Return a measure which is active on the provided subdomain."""
    if subdomain is not None:
        entities_dim = mesh.topology.dim - subdomain.codimension  # type: ignore[attr-defined]
        entities = dolfinx.mesh.locate_entities(mesh, entities_dim, subdomain)
        mesh.topology.create_connectivity(entities_dim, mesh.topology.dim)
        values = np.full(entities.shape, 1, dtype=np.intc)
        meshtags = dolfinx.mesh.meshtags(mesh, entities_dim, entities, values)
        dx = ufl.Measure("dx", subdomain_data=meshtags)
        return dx(1)
    else:
        return ufl.dx


@pytest.mark.parametrize("subdomain", get_subdomains())
@pytest.mark.parametrize("kind", [None, "mpi"])
def test_plain_linear_solver(
    mesh: dolfinx.mesh.Mesh, subdomain: typing.Optional[common.SubdomainType],
    kind: typing.Optional[str]
) -> None:
    """Test solution of a linear problem with single form with restrictions."""
    # Define function space and trial/test functions
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    # Define restriction
    active_dofs = common.ActiveDofs(V, subdomain)
    restriction = multiphenicsx.fem.DofMapRestriction(V.dofmap, active_dofs)
    active_dx = active_measure(mesh, subdomain)
    # Define forms
    a = ufl.inner(u, v) * active_dx
    x = ufl.SpatialCoordinate(mesh)
    f = x[0] + 3 * x[1]
    L = ufl.inner(f, v) * active_dx
    # Define boundary conditions
    f_expr = dolfinx.fem.Expression(f, V.element.interpolation_points)
    f_bc = dolfinx.fem.Function(V)
    f_bc.interpolate(f_expr)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[1], 1.0))
    dofs_bc = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bcs = [dolfinx.fem.dirichletbc(f_bc, dofs_bc)]
    # Solve
    problem = multiphenicsx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options=petsc_options, kind=kind, restriction=restriction)
    solution, convergence_reason, _ = problem.solve()
    assert convergence_reason > 0
    # Compute error
    error_ufl = dolfinx.fem.form(ufl.inner(solution - f, solution - f) * active_dx)
    error = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(error_ufl), op=mpi4py.MPI.SUM))
    tol = 500 * np.finfo(petsc4py.PETSc.ScalarType).eps
    assert error < tol


@pytest.mark.parametrize("subdomain", get_subdomains())
@pytest.mark.parametrize("kind", ["mpi", "nest", [["aij", None], [None, "baij"]]])
def test_block_nest_linear_solver(
    mesh: dolfinx.mesh.Mesh, subdomain: typing.Optional[common.SubdomainType],
    kind: typing.Optional[typing.Union[str, list[list[str]]]]
) -> None:
    """Test solution of a linear problem with single form with restrictions."""
    # Define function spaces
    V = [dolfinx.fem.functionspace(mesh, ("Lagrange", degree)) for degree in (1, 2)]
    u = [ufl.TrialFunction(Vi) for Vi in V]
    v = [ufl.TestFunction(Vi) for Vi in V]
    # Define restriction
    active_dofs = [common.ActiveDofs(Vi, subdomain) for Vi in V]
    restriction = [
        multiphenicsx.fem.DofMapRestriction(Vi.dofmap, active_dofs_i) for (Vi, active_dofs_i) in zip(V, active_dofs)]
    active_dx = active_measure(mesh, subdomain)
    # Define forms
    a = [[ufl.inner(u[0], v[0]) * active_dx, None], [None, ufl.inner(u[1], v[1]) * active_dx]]
    x = ufl.SpatialCoordinate(mesh)
    f = [x[0] + 3 * x[1], -(x[1] ** 2) + x[0]]
    L = [ufl.inner(f[0], v[0]) * active_dx, ufl.inner(f[1], v[1]) * active_dx]
    # Define boundary conditions
    f_expr = [dolfinx.fem.Expression(fi, Vi.element.interpolation_points) for (fi, Vi) in zip(f, V)]
    f_bc = [dolfinx.fem.Function(Vi) for Vi in V]
    for i in range(2):
        f_bc[i].interpolate(f_expr[i])
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[1], 1.0))
    dofs_bc = [dolfinx.fem.locate_dofs_topological(Vi, mesh.topology.dim - 1, boundary_facets) for Vi in V]
    bcs = [dolfinx.fem.dirichletbc(f_bc_i, dofs_bc_i) for (f_bc_i, dofs_bc_i) in zip(f_bc, dofs_bc)]
    # Solve
    problem = multiphenicsx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options=petsc_options, kind=kind, restriction=restriction)
    solutions, convergence_reason, _ = problem.solve()
    assert convergence_reason > 0
    # Compute error
    for (fi, si) in zip(f, solutions):
        error_i_ufl = dolfinx.fem.form(ufl.inner(si - fi, si - fi) * active_dx)
        error_i = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(error_i_ufl), op=mpi4py.MPI.SUM))
        tol = 500 * np.finfo(petsc4py.PETSc.ScalarType).eps
        assert error_i < tol

# Copyright (C) 2016-2021 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pytest
import scipy.sparse
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import cpp, DirichletBC, Function, FunctionSpace, UnitSquareMesh, VectorFunctionSpace
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import locate_entities
from ufl import dx, inner, MixedElement, TestFunction, TrialFunction

from multiphenicsx.fem import (
    apply_lifting, apply_lifting_nest,
    assemble_matrix, assemble_matrix_block, assemble_matrix_nest,
    assemble_vector, assemble_vector_block, assemble_vector_nest,
    BlockVecSubVectorWrapper, BlockVecSubVectorReadWrapper,
    DofMapRestriction, NestVecSubVectorWrapper, NestVecSubVectorReadWrapper,
    set_bc, set_bc_nest, VecSubVectorWrapper)

from test_dofmap_restriction import ActiveDofs, CellsSubDomain, FacetsSubDomain, TaylorHoodFunctionSpace


# Mesh
@pytest.fixture
def mesh():
    return UnitSquareMesh(MPI.COMM_WORLD, 4, 4)


# Subdomain parametrization
def get_subdomains():
    return (
        # Unrestricted
        None,
        # Restricted
        CellsSubDomain(0.5, 0.75)
    )


def get_subdomains_pairs():
    return (
        # (unrestricted, unrestricted)
        (None, None),
        # (unrestricted, restricted)
        (None, CellsSubDomain(0.5, 0.75)),
        # (restricted, unrestricted)
        (CellsSubDomain(0.5, 0.75), None),
        # (restricted, restricted)
        (CellsSubDomain(0.5, 0.75), CellsSubDomain(0.75, 0.5))
    )


# Function space parametrization
def get_function_spaces():
    return (
        lambda mesh: FunctionSpace(mesh, ("Lagrange", 1)),
        lambda mesh: VectorFunctionSpace(mesh, ("Lagrange", 1)),
        lambda mesh: TaylorHoodFunctionSpace(mesh, ("Lagrange", 1))
    )


def get_function_spaces_pairs():
    for i in get_function_spaces():
        for j in get_function_spaces():
            yield (i, j)


# Generation of function employed in form definition
def get_function(V, preprocess_x=None):
    if preprocess_x is None:
        def preprocess_x(x):
            return x

    shape = V.ufl_element().value_shape()
    if len(shape) == 0:
        def f(x):
            x = preprocess_x(x)
            return 2 * x[0] + 4 * x[1] * x[1]
    elif len(shape) == 1 and shape[0] == 2:
        def f(x):
            x = preprocess_x(x)
            return np.stack([
                2 * x[0] + 4 * x[1] * x[1],
                3 * x[0] + 5 * x[1] * x[1]
            ], axis=0)
    elif len(shape) == 1 and shape[0] == 3:
        def f(x):
            x = preprocess_x(x)
            return np.stack([
                2 * x[0] + 4 * x[1] * x[1],
                3 * x[0] + 5 * x[1] * x[1],
                7 * x[0] + 11 * x[1] * x[1]
            ], axis=0)
    u = Function(V)
    try:
        u.interpolate(f)
    except RuntimeError:
        assert len(shape) == 1

        assert isinstance(V.ufl_element(), MixedElement)
        rows = [np.prod(sub_element.value_shape(), dtype=int) for sub_element in V.ufl_element().sub_elements()]
        rows = np.hstack(([0], np.cumsum(rows)))

        for i in range(len(rows) - 1):
            u.sub(i).interpolate(lambda x: f(x)[rows[i]:rows[i + 1], :])
    return u


# Generation of functions employed in block form definition
def get_function_pair(V1, V2):
    u1 = get_function(V1)
    u2 = get_function(V2, lambda x: np.flip(x, axis=1))
    return [u1, u2]


# Generation of linear forms employed in the vector test case
def get_linear_form(V):
    v = TestFunction(V)
    f = get_function(V)
    return inner(f, v) * dx


# Generation of two-by-one block linear forms employed in the block/nest vector test cases
def get_block_linear_form(V1, V2):
    v1, v2 = TestFunction(V1), TestFunction(V2)
    assert V1.mesh == V2.mesh
    f1, f2 = get_function_pair(V1, V2)
    return [inner(f1, v1) * dx, inner(f2, v2) * dx]


# Generation of bilinear forms employed in the matrix test case
def get_bilinear_form(V):
    u = TrialFunction(V)
    v = TestFunction(V)
    f = get_function(V)
    shape = V.ufl_element().value_shape()
    if len(shape) == 0:
        return f * inner(u, v) * dx
    elif len(shape) == 1:
        return sum(f[i] * inner(u[i], v[i]) for i in range(shape[0])) * dx


# Generation of two-by-two block bilinear forms employed in the block/nest matrix test cases
def get_block_bilinear_form(V1, V2):
    u1, u2 = TrialFunction(V1), TrialFunction(V2)
    v1, v2 = TestFunction(V1), TestFunction(V2)
    assert V1.mesh == V2.mesh
    f1, f2 = get_function_pair(V1, V2)

    def diff(v, index):
        if index >= 0:
            assert index in (0, 1)
            return v.dx(index)
        else:
            assert index == -1
            return v

    block_form = [[None, None], [None, None]]

    # (1, 1) block
    shape_1 = V1.ufl_element().value_shape()
    if len(shape_1) == 0:
        block_form[0][0] = f1 * inner(u1, v1) * dx
    elif len(shape_1) == 1:
        block_form[0][0] = sum(f1[i] * inner(u1[i], v1[i])
                               for i in range(shape_1[0])) * dx

    # (2, 2) block
    shape_2 = V2.ufl_element().value_shape()
    if len(shape_2) == 0:
        block_form[1][1] = f2 * inner(u2, v2) * dx
    elif len(shape_2) == 1:
        block_form[1][1] = sum(f2[i] * inner(u2[i], v2[i])
                               for i in range(shape_2[0])) * dx

    # (1, 2) and (2, 1) blocks
    if len(shape_1) == 0:
        if len(shape_2) == 0:
            block_form[0][1] = f1 * inner(u2, v1) * dx
            block_form[1][0] = f2 * inner(u1, v2) * dx
        elif len(shape_2) == 1:
            block_form[0][1] = sum(f1 * inner(u2[i], diff(v1, i - 1))
                                   for i in range(shape_2[0])) * dx
            block_form[1][0] = sum(f2[i] * inner(diff(u1, i - 1), v2[i])
                                   for i in range(shape_2[0])) * dx
    elif len(shape_1) == 1:
        if len(shape_2) == 0:
            block_form[0][1] = sum(f1[i] * inner(diff(u2, i - 1), v1[i])
                                   for i in range(shape_1[0])) * dx
            block_form[1][0] = sum(f2 * inner(u1[i], diff(v2, i - 1))
                                   for i in range(shape_1[0])) * dx
        elif len(shape_2) == 1:
            block_form[0][1] = sum(f1[i % shape_1[0]] * inner(u2[i % shape_2[0]], v1[i % shape_1[0]])
                                   for i in range(max(shape_1[0], shape_2[0]))) * dx
            block_form[1][0] = sum(f2[i % shape_2[0]] * inner(u1[i % shape_1[0]], v2[i % shape_2[0]])
                                   for i in range(max(shape_1[0], shape_2[0]))) * dx
    return block_form


# Auxiliary function for boundary dofs localization
def locate_boundary_dofs(V, collapsed_V=None):
    entities_dim = V.mesh.topology.dim - 1
    entities = locate_entities(V.mesh, entities_dim, FacetsSubDomain(on_boundary=True))
    if collapsed_V is None:
        return locate_dofs_topological(V, entities_dim, entities)
    else:
        return locate_dofs_topological((V, collapsed_V), entities_dim, entities)


# Generation of boundary conditions employed in the non-block/nest test cases
def get_boundary_conditions(offset=0):

    def _get_boundary_conditions(V):
        num_sub_elements = V.ufl_element().num_sub_elements()
        if num_sub_elements == 0:
            bc1_fun = Function(V)
            with bc1_fun.vector.localForm() as local_form:
                local_form.set(1. + offset)
            bdofs = locate_boundary_dofs(V)
            return [DirichletBC(bc1_fun, bdofs)]
        else:
            bc1 = list()
            for i in range(num_sub_elements):
                bc1_fun = Function(V.sub(i).collapse())
                with bc1_fun.vector.localForm() as local_form:
                    local_form.set(i + 1. + offset)
                bdofs = locate_boundary_dofs(V.sub(i), bc1_fun.function_space)
                bc1.append(DirichletBC(bc1_fun, bdofs, V.sub(i)))
            return bc1

    return (lambda _: [],
            _get_boundary_conditions)


# Generation of boundary conditions employed in the block/nest test cases
def get_boundary_conditions_pairs():
    return (lambda _, __: [[], []],
            lambda V1, _: [get_boundary_conditions(offset=0)[1](V1), []],
            lambda _, V2: [[], get_boundary_conditions(offset=10)[1](V2)],
            lambda V1, V2: [get_boundary_conditions(offset=0)[1](V1),
                            get_boundary_conditions(offset=10)[1](V2)])


# Helper function to allgather global map from restricted dofs to unrestricted dofs
def get_global_restricted_to_unrestricted(dofmap_restriction, comm):
    assert isinstance(dofmap_restriction, (DofMapRestriction, list))
    if isinstance(dofmap_restriction, DofMapRestriction):  # case of standard matrix/vector
        dofmap = dofmap_restriction.dofmap
        bs = dofmap.index_map_bs
        assert bs == dofmap_restriction.index_map_bs
        restricted_to_unrestricted = dict()
        for (restricted, unrestricted) in dofmap_restriction.restricted_to_unrestricted.items():
            for s in range(bs):
                restricted_to_unrestricted[bs * restricted + s] = bs * unrestricted + s
        all_restricted_to_unrestricted = comm.allgather(restricted_to_unrestricted)
        all_unrestricted_local_ranges = comm.allgather([lr * bs for lr in dofmap.index_map.local_range])
        all_restricted_local_ranges = comm.allgather([lr * bs for lr in dofmap_restriction.index_map.local_range])
        unrestricted_base_index = [lr[0] for lr in all_unrestricted_local_ranges]
        restricted_base_index = [lr[0] for lr in all_restricted_local_ranges]
        global_restricted_to_unrestricted = dict()
        for r in range(comm.Get_size()):
            for (restricted, unrestricted) in all_restricted_to_unrestricted[r].items():
                if unrestricted < all_unrestricted_local_ranges[r][1] - all_unrestricted_local_ranges[r][0]:
                    assert restricted + restricted_base_index[r] not in global_restricted_to_unrestricted
                    global_restricted_to_unrestricted[
                        restricted + restricted_base_index[r]] = unrestricted + unrestricted_base_index[r]
        return global_restricted_to_unrestricted
    elif isinstance(dofmap_restriction, list):  # case of block/nest matrix/vector
        assert all(isinstance(dofmap_restriction_, DofMapRestriction) for dofmap_restriction_ in dofmap_restriction)
        assert len(dofmap_restriction) == 2, "The code below is hardcoded for two blocks"
        dofmaps = [dofmap_restriction_.dofmap for dofmap_restriction_ in dofmap_restriction]
        bs = [dofmap.index_map_bs for dofmap in dofmaps]
        for (dofmap_restriction_, bs_) in zip(dofmap_restriction, bs):
            assert dofmap_restriction_.index_map_bs == bs_
        restricted_to_unrestricted = list()
        for (dofmap_restriction_, bs_) in zip(dofmap_restriction, bs):
            restricted_to_unrestricted_ = dict()
            for (restricted, unrestricted) in dofmap_restriction_.restricted_to_unrestricted.items():
                for s in range(bs_):
                    restricted_to_unrestricted_[bs_ * restricted + s] = bs_ * unrestricted + s
            restricted_to_unrestricted.append(restricted_to_unrestricted_)
        all_restricted_to_unrestricted = [comm.allgather(restricted_to_unrestricted_)
                                          for restricted_to_unrestricted_ in restricted_to_unrestricted]
        all_unrestricted_local_ranges = [comm.allgather([lr * dofmap.index_map_bs
                                                        for lr in dofmap.index_map.local_range])
                                         for dofmap in dofmaps]
        all_restricted_local_ranges = [comm.allgather([lr * dofmap_restriction_.index_map_bs
                                                      for lr in dofmap_restriction_.index_map.local_range])
                                       for dofmap_restriction_ in dofmap_restriction]
        unrestricted_base_index = [[None] * comm.Get_size() for _ in range(2)]
        restricted_base_index = [[None] * comm.Get_size() for _ in range(2)]
        for r in range(comm.Get_size()):
            if r > 0:
                unrestricted_base_index[0][r] = (unrestricted_base_index[1][r - 1]
                                                 + (all_unrestricted_local_ranges[1][r - 1][1]
                                                 - all_unrestricted_local_ranges[1][r - 1][0]))
            else:
                unrestricted_base_index[0][0] = 0
            unrestricted_base_index[1][r] = (unrestricted_base_index[0][r]
                                             + (all_unrestricted_local_ranges[0][r][1]
                                             - all_unrestricted_local_ranges[0][r][0]))
            if r > 0:
                restricted_base_index[0][r] = (restricted_base_index[1][r - 1]
                                               + (all_restricted_local_ranges[1][r - 1][1]
                                               - all_restricted_local_ranges[1][r - 1][0]))
            else:
                restricted_base_index[0][0] = 0
            restricted_base_index[1][r] = (restricted_base_index[0][r]
                                           + (all_restricted_local_ranges[0][r][1]
                                           - all_restricted_local_ranges[0][r][0]))
        global_restricted_to_unrestricted = dict()
        for b in range(2):
            for r in range(comm.Get_size()):
                for (restricted, unrestricted) in all_restricted_to_unrestricted[b][r].items():
                    if unrestricted < all_unrestricted_local_ranges[b][r][1] - all_unrestricted_local_ranges[b][r][0]:
                        assert restricted + restricted_base_index[b][r] not in global_restricted_to_unrestricted
                        global_restricted_to_unrestricted[
                            restricted + restricted_base_index[b][r]] = unrestricted + unrestricted_base_index[b][r]
        return global_restricted_to_unrestricted
    else:
        raise RuntimeError("Invalid argument provided to gather_global_restriction_map")


# Helper function to convert distributed PETSc Vec to a dense allgather-ed numpy array
def to_numpy_vector(vec):
    local_np_vec = vec.getArray()
    comm = vec.getComm().tompi4py()
    return np.hstack(comm.allgather(local_np_vec))


# Helper function to convert distributed PETSc Mat to a dense allgather-ed numpy matrix
def to_numpy_matrix(mat):
    ai, aj, av = mat.getValuesCSR()
    local_np_mat = scipy.sparse.csr_matrix((av, aj, ai), shape=(mat.getLocalSize()[0], mat.getSize()[1])).toarray()
    comm = mat.getComm().tompi4py()
    return np.vstack(comm.allgather(local_np_mat))


# Assert equality between the vector resulting from the assembly of a linear form with non-empty
# restriction argument and the vector resulting from the assembly of the same linear form with empty
# restriction argument accompanied by a postprocessing which manually removes discarded rows
def assert_vector_equal(unrestricted_vector, restricted_vector, dofmap_restriction):
    # Gather global representation of provided vectors
    unrestricted_vector_global = to_numpy_vector(unrestricted_vector)
    restricted_vector_global = to_numpy_vector(restricted_vector)
    # Gather global map from restricted dofs to unrestricted dofs
    restricted_to_unrestricted_global = get_global_restricted_to_unrestricted(
        dofmap_restriction, restricted_vector.comm.tompi4py())
    assert restricted_vector_global.shape[0] == len(restricted_to_unrestricted_global)
    # Manually discard rows from global representation of unrestricted vector
    restricted_vector_global_expected = restricted_vector_global * 0.
    for (restricted_i, unrestricted_i) in restricted_to_unrestricted_global.items():
        restricted_vector_global_expected[restricted_i] = unrestricted_vector_global[unrestricted_i]
    assert np.allclose(restricted_vector_global, restricted_vector_global_expected)


# Assert equality between the matrix resulting from the assembly of a bilinear form with non-empty
# restriction argument and the matrix resulting from the assembly of the same bilinear form with empty
# restriction argument accompanied by a postprocessing which manually removes discarded rows/cols
def assert_matrix_equal(unrestricted_matrix, restricted_matrix, dofmap_restriction):
    # Gather global representation of provided matrices
    unrestricted_matrix_global = to_numpy_matrix(unrestricted_matrix)
    restricted_matrix_global = to_numpy_matrix(restricted_matrix)
    # Gather global map from restricted dofs to unrestricted dofs
    assert isinstance(dofmap_restriction, tuple)
    assert len(dofmap_restriction) == 2
    restricted_to_unrestricted_global = [
        get_global_restricted_to_unrestricted(dofmap_restriction[0], restricted_matrix.comm.tompi4py()),
        get_global_restricted_to_unrestricted(dofmap_restriction[1], restricted_matrix.comm.tompi4py())
    ]
    assert restricted_matrix_global.shape[0] == len(restricted_to_unrestricted_global[0])
    assert restricted_matrix_global.shape[1] == len(restricted_to_unrestricted_global[1])
    # Manually discard rows and cols from global representation of unrestricted matrix
    restricted_matrix_global_expected = restricted_matrix_global * 0.
    for (restricted_i, unrestricted_i) in restricted_to_unrestricted_global[0].items():
        for (restricted_j, unrestricted_j) in restricted_to_unrestricted_global[1].items():
            restricted_matrix_global_expected[restricted_i, restricted_j] = unrestricted_matrix_global[
                unrestricted_i, unrestricted_j]
    assert np.allclose(restricted_matrix_global, restricted_matrix_global_expected)


# Test assembly of a linear form with restrictions
@pytest.mark.parametrize("subdomain", get_subdomains())
@pytest.mark.parametrize("FunctionSpace", get_function_spaces())
@pytest.mark.parametrize("dirichlet_bcs", get_boundary_conditions())
def test_vector_assembly_with_restriction(mesh, subdomain, FunctionSpace, dirichlet_bcs):
    V = FunctionSpace(mesh)
    active_dofs = ActiveDofs(V, subdomain)
    dofmap_restriction = DofMapRestriction(V.dofmap, active_dofs)
    linear_form = get_linear_form(V)
    bilinear_form = get_bilinear_form(V)
    bcs = dirichlet_bcs(V)
    # Assembly without BCs
    unrestricted_vector = assemble_vector(linear_form)
    unrestricted_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    restricted_vector = assemble_vector(linear_form, restriction=dofmap_restriction)
    restricted_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert_vector_equal(unrestricted_vector, restricted_vector, dofmap_restriction)
    # BC application for linear problems
    unrestricted_vector_linear = assemble_vector(linear_form)
    apply_lifting(unrestricted_vector_linear, [bilinear_form], [bcs])
    unrestricted_vector_linear.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    restricted_vector_linear = assemble_vector(linear_form, restriction=dofmap_restriction)
    apply_lifting(restricted_vector_linear, [bilinear_form], [bcs], restriction=dofmap_restriction)
    restricted_vector_linear.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert_vector_equal(unrestricted_vector_linear, restricted_vector_linear, dofmap_restriction)
    set_bc(unrestricted_vector_linear, bcs)
    set_bc(restricted_vector_linear, bcs, restriction=dofmap_restriction)
    assert_vector_equal(unrestricted_vector_linear, restricted_vector_linear, dofmap_restriction)
    # BC application for nonlinear problems
    unrestricted_solution = cpp.la.create_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
    restricted_solution = cpp.la.create_vector(dofmap_restriction.index_map, dofmap_restriction.index_map_bs)
    with unrestricted_solution.localForm() as unrestricted_solution_local, \
            get_function(V).vector.localForm() as function_local:
        active_dofs_bs = [V.dofmap.index_map_bs * d + s
                          for d in active_dofs.astype(np.int32) for s in range(V.dofmap.index_map_bs)]
        unrestricted_solution_local[active_dofs_bs] = function_local[active_dofs_bs]
        with VecSubVectorWrapper(restricted_solution, V.dofmap, dofmap_restriction) as restricted_solution_wrapper:
            restricted_solution_wrapper[:] = unrestricted_solution_local
    unrestricted_vector_nonlinear = assemble_vector(linear_form)
    apply_lifting(unrestricted_vector_nonlinear, [bilinear_form], [bcs], [unrestricted_solution])
    unrestricted_vector_nonlinear.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    restricted_vector_nonlinear = assemble_vector(linear_form, restriction=dofmap_restriction)
    apply_lifting(restricted_vector_nonlinear, [bilinear_form], [bcs], [restricted_solution],
                  restriction=dofmap_restriction, restriction_x0=[dofmap_restriction])
    restricted_vector_nonlinear.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert_vector_equal(unrestricted_vector_nonlinear, restricted_vector_nonlinear, dofmap_restriction)
    set_bc(unrestricted_vector_nonlinear, bcs, unrestricted_solution)
    set_bc(restricted_vector_nonlinear, bcs, restricted_solution, restriction=dofmap_restriction)
    assert_vector_equal(unrestricted_vector_nonlinear, restricted_vector_nonlinear, dofmap_restriction)


# Test block assembly of a two-by-one block linear form with restrictions
@pytest.mark.parametrize("subdomains", get_subdomains_pairs())
@pytest.mark.parametrize("FunctionSpaces", get_function_spaces_pairs())
@pytest.mark.parametrize("dirichlet_bcs", get_boundary_conditions_pairs())
def test_block_vector_assembly_with_restriction(mesh, subdomains, FunctionSpaces, dirichlet_bcs):
    V = [FunctionSpace(mesh) for FunctionSpace in FunctionSpaces]
    dofmaps = [V_.dofmap for V_ in V]
    active_dofs = [ActiveDofs(V_, subdomain) for (V_, subdomain) in zip(V, subdomains)]
    dofmap_restriction = [DofMapRestriction(V_.dofmap, active_dofs_) for (V_, active_dofs_) in zip(V, active_dofs)]
    block_linear_form = get_block_linear_form(*V)
    block_bilinear_form = get_block_bilinear_form(*V)
    bcs = [bc for bcs in dirichlet_bcs(*V) for bc in bcs]
    # Assembly for linear problems
    unrestricted_vector_linear = assemble_vector_block(block_linear_form, block_bilinear_form, bcs=bcs)
    restricted_vector_linear = assemble_vector_block(block_linear_form, block_bilinear_form, bcs=bcs,
                                                     restriction=dofmap_restriction)
    assert_vector_equal(unrestricted_vector_linear, restricted_vector_linear, dofmap_restriction)
    # Assembly for nonlinear problems
    unrestricted_solution = cpp.fem.create_vector_block([(V_.dofmap.index_map, V_.dofmap.index_map_bs)
                                                         for V_ in V])
    restricted_solution = cpp.fem.create_vector_block([(restriction.index_map, restriction.index_map_bs)
                                                       for restriction in dofmap_restriction])
    with BlockVecSubVectorWrapper(unrestricted_solution, dofmaps) as unrestricted_solution_wrapper:
        for (active_dofs_sub, unrestricted_solution_sub_local, function_sub, dofmap_sub) in zip(
                active_dofs, unrestricted_solution_wrapper, get_function_pair(*V), dofmaps):
            active_dofs_sub_bs = [dofmap_sub.index_map_bs * d + s
                                  for d in active_dofs_sub.astype(np.int32) for s in range(dofmap_sub.index_map_bs)]
            with function_sub.vector.localForm() as function_sub_local:
                unrestricted_solution_sub_local[active_dofs_sub_bs] = function_sub_local[active_dofs_sub_bs]
    with BlockVecSubVectorWrapper(restricted_solution, dofmaps, dofmap_restriction) as restricted_solution_wrapper, \
            BlockVecSubVectorReadWrapper(unrestricted_solution, dofmaps) as unrestricted_solution_wrapper:
        for (restricted_solution_sub, unrestricted_solution_sub) in zip(
                restricted_solution_wrapper, unrestricted_solution_wrapper):
            restricted_solution_sub[:] = unrestricted_solution_sub
    unrestricted_vector_nonlinear = assemble_vector_block(block_linear_form, block_bilinear_form, bcs=bcs,
                                                          x0=unrestricted_solution)
    restricted_vector_nonlinear = assemble_vector_block(block_linear_form, block_bilinear_form, bcs=bcs,
                                                        x0=restricted_solution, restriction=dofmap_restriction,
                                                        restriction_x0=dofmap_restriction)
    assert_vector_equal(unrestricted_vector_nonlinear, restricted_vector_nonlinear, dofmap_restriction)


# Test nest assembly of a two-by-one block linear form with restrictions
@pytest.mark.parametrize("subdomains", get_subdomains_pairs())
@pytest.mark.parametrize("FunctionSpaces", get_function_spaces_pairs())
@pytest.mark.parametrize("dirichlet_bcs", get_boundary_conditions_pairs())
def test_nest_vector_assembly_with_restriction(mesh, subdomains, FunctionSpaces, dirichlet_bcs):
    V = [FunctionSpace(mesh) for FunctionSpace in FunctionSpaces]
    dofmaps = [V_.dofmap for V_ in V]
    active_dofs = [ActiveDofs(V_, subdomain) for (V_, subdomain) in zip(V, subdomains)]
    dofmap_restriction = [DofMapRestriction(V_.dofmap, active_dofs_) for (V_, active_dofs_) in zip(V, active_dofs)]
    block_linear_form = get_block_linear_form(*V)
    block_bilinear_form = get_block_bilinear_form(*V)
    bcs_pair = dirichlet_bcs(*V)
    bcs_flattened = [bc for bcs in bcs_pair for bc in bcs]
    # Assembly without BCs
    unrestricted_vector = assemble_vector_nest(block_linear_form)
    for unrestricted_vector_sub in unrestricted_vector.getNestSubVecs():
        unrestricted_vector_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    restricted_vector = assemble_vector_nest(block_linear_form, restriction=dofmap_restriction)
    for restricted_vector_sub in restricted_vector.getNestSubVecs():
        restricted_vector_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert_vector_equal(unrestricted_vector, restricted_vector, dofmap_restriction)
    # BC application for linear problems
    unrestricted_vector_linear = assemble_vector_nest(block_linear_form)
    apply_lifting_nest(unrestricted_vector_linear, block_bilinear_form, bcs_flattened)
    for unrestricted_vector_sub in unrestricted_vector_linear.getNestSubVecs():
        unrestricted_vector_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    restricted_vector_linear = assemble_vector_nest(block_linear_form, restriction=dofmap_restriction)
    apply_lifting_nest(restricted_vector_linear, block_bilinear_form, bcs_flattened, restriction=dofmap_restriction)
    for restricted_vector_sub in restricted_vector_linear.getNestSubVecs():
        restricted_vector_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert_vector_equal(unrestricted_vector_linear, restricted_vector_linear, dofmap_restriction)
    set_bc_nest(unrestricted_vector_linear, bcs_pair)
    set_bc_nest(restricted_vector_linear, bcs_pair, restriction=dofmap_restriction)
    assert_vector_equal(unrestricted_vector_linear, restricted_vector_linear, dofmap_restriction)
    # BC application for nonlinear problems
    unrestricted_solution = cpp.fem.create_vector_nest([(V_.dofmap.index_map, V_.dofmap.index_map_bs)
                                                        for V_ in V])
    restricted_solution = cpp.fem.create_vector_nest([(restriction.index_map, restriction.index_map_bs)
                                                      for restriction in dofmap_restriction])
    with NestVecSubVectorWrapper(unrestricted_solution, dofmaps) as unrestricted_solution_wrapper:
        for (active_dofs_sub, unrestricted_solution_sub_local, function_sub, dofmap_sub) in zip(
                active_dofs, unrestricted_solution_wrapper, get_function_pair(*V), dofmaps):
            active_dofs_sub_bs = [dofmap_sub.index_map_bs * d + s
                                  for d in active_dofs_sub.astype(np.int32) for s in range(dofmap_sub.index_map_bs)]
            with function_sub.vector.localForm() as function_sub_local:
                unrestricted_solution_sub_local[active_dofs_sub_bs] = function_sub_local[active_dofs_sub_bs]
    with NestVecSubVectorWrapper(restricted_solution, dofmaps, dofmap_restriction) as restricted_solution_wrapper, \
            NestVecSubVectorReadWrapper(unrestricted_solution, dofmaps) as unrestricted_solution_wrapper:
        for (restricted_solution_sub, unrestricted_solution_sub) in zip(
                restricted_solution_wrapper, unrestricted_solution_wrapper):
            restricted_solution_sub[:] = unrestricted_solution_sub
    unrestricted_vector_nonlinear = assemble_vector_nest(block_linear_form)
    apply_lifting_nest(unrestricted_vector_nonlinear, block_bilinear_form, bcs_flattened, unrestricted_solution)
    for unrestricted_vector_sub in unrestricted_vector_linear.getNestSubVecs():
        unrestricted_vector_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    restricted_vector_nonlinear = assemble_vector_nest(block_linear_form, restriction=dofmap_restriction)
    apply_lifting_nest(restricted_vector_nonlinear, block_bilinear_form, bcs_flattened, restricted_solution,
                       restriction=dofmap_restriction, restriction_x0=dofmap_restriction)
    for restricted_vector_sub in restricted_vector_linear.getNestSubVecs():
        restricted_vector_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert_vector_equal(unrestricted_vector_nonlinear, restricted_vector_nonlinear, dofmap_restriction)
    set_bc_nest(unrestricted_vector_nonlinear, bcs_pair, unrestricted_solution)
    set_bc_nest(restricted_vector_nonlinear, bcs_pair, restricted_solution, restriction=dofmap_restriction)
    assert_vector_equal(unrestricted_vector_nonlinear, restricted_vector_nonlinear, dofmap_restriction)


# Test assembly of a bilinear form with restrictions
@pytest.mark.parametrize("subdomain", get_subdomains())
@pytest.mark.parametrize("FunctionSpace", get_function_spaces())
@pytest.mark.parametrize("dirichlet_bcs", get_boundary_conditions())
def test_matrix_assembly_with_restriction(mesh, subdomain, FunctionSpace, dirichlet_bcs):
    V = FunctionSpace(mesh)
    active_dofs = ActiveDofs(V, subdomain)
    dofmap_restriction = DofMapRestriction(V.dofmap, active_dofs)
    form = get_bilinear_form(V)
    bcs = dirichlet_bcs(V)
    unrestricted_matrix = assemble_matrix(form, bcs=bcs)
    unrestricted_matrix.assemble()
    restricted_matrix = assemble_matrix(form, bcs=bcs, restriction=(dofmap_restriction, dofmap_restriction))
    restricted_matrix.assemble()
    assert_matrix_equal(unrestricted_matrix, restricted_matrix, (dofmap_restriction, dofmap_restriction))


# Test block assembly of a two-by-two block bilinear form with restrictions
@pytest.mark.parametrize("subdomains", get_subdomains_pairs())
@pytest.mark.parametrize("FunctionSpaces", get_function_spaces_pairs())
@pytest.mark.parametrize("dirichlet_bcs", get_boundary_conditions_pairs())
def test_block_matrix_assembly_with_restriction(mesh, subdomains, FunctionSpaces, dirichlet_bcs):
    V = [FunctionSpace(mesh) for FunctionSpace in FunctionSpaces]
    active_dofs = [ActiveDofs(V_, subdomain) for (V_, subdomain) in zip(V, subdomains)]
    dofmap_restriction = [DofMapRestriction(V_.dofmap, active_dofs_) for (V_, active_dofs_) in zip(V, active_dofs)]
    block_form = get_block_bilinear_form(*V)
    bcs = [bc for bcs in dirichlet_bcs(*V) for bc in bcs]
    unrestricted_matrix = assemble_matrix_block(block_form, bcs=bcs)
    unrestricted_matrix.assemble()
    restricted_matrix = assemble_matrix_block(block_form, bcs=bcs, restriction=(dofmap_restriction, dofmap_restriction))
    restricted_matrix.assemble()
    assert_matrix_equal(unrestricted_matrix, restricted_matrix, (dofmap_restriction, dofmap_restriction))


# Test nest assembly of a two-by-two block bilinear form with restrictions
@pytest.mark.parametrize("subdomains", get_subdomains_pairs())
@pytest.mark.parametrize("FunctionSpaces", get_function_spaces_pairs())
@pytest.mark.parametrize("dirichlet_bcs", get_boundary_conditions_pairs())
def test_nest_matrix_assembly_with_restriction(mesh, subdomains, FunctionSpaces, dirichlet_bcs):
    V = [FunctionSpace(mesh) for FunctionSpace in FunctionSpaces]
    active_dofs = [ActiveDofs(V_, subdomain) for (V_, subdomain) in zip(V, subdomains)]
    dofmap_restriction = [DofMapRestriction(V_.dofmap, active_dofs_) for (V_, active_dofs_) in zip(V, active_dofs)]
    block_form = get_block_bilinear_form(*V)
    bcs = [bc for bcs in dirichlet_bcs(*V) for bc in bcs]
    unrestricted_matrix = assemble_matrix_nest(block_form, bcs=bcs)
    unrestricted_matrix.assemble()
    restricted_matrix = assemble_matrix_nest(block_form, bcs=bcs, restriction=(dofmap_restriction, dofmap_restriction))
    restricted_matrix.assemble()
    for i in range(2):
        for j in range(2):
            unrestricted_matrix_ij = unrestricted_matrix.getNestSubMatrix(i, j)
            restricted_matrix_ij = restricted_matrix.getNestSubMatrix(i, j)
            assert_matrix_equal(unrestricted_matrix_ij, restricted_matrix_ij,
                                (dofmap_restriction[i], dofmap_restriction[j]))

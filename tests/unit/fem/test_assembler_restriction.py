# Copyright (C) 2016-2025 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for multiphenicsx.fem.petsc module, assembler functions."""

import types
import typing

import basix.ufl
import dolfinx.cpp
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.la
import dolfinx.la.petsc
import dolfinx.mesh
import mpi4py.MPI
import numpy as np
import numpy.typing as npt
import petsc4py.PETSc
import pytest
import scipy.sparse
import ufl

import multiphenicsx.fem
import multiphenicsx.fem.petsc

import common  # isort: skip

PreprocessXType = typing.Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
DirichletBCsGeneratorType = typing.Callable[[dolfinx.fem.FunctionSpace], list[dolfinx.fem.DirichletBC]]
DirichletBCsPairGeneratorType = typing.Callable[
    [dolfinx.fem.FunctionSpace, dolfinx.fem.FunctionSpace], list[list[dolfinx.fem.DirichletBC]]]
ApplySetDirichletBCsNonlinearArgumentsType = typing.Callable[
    [petsc4py.PETSc.Vec, petsc4py.PETSc.Vec, list[multiphenicsx.fem.DofMapRestriction]],  # type: ignore[name-defined]
    tuple[
        petsc4py.PETSc.Vec, typing.Union[list[multiphenicsx.fem.DofMapRestriction], None]  # type: ignore[name-defined]
    ]]
DofMapRestrictionsType = typing.Union[
    multiphenicsx.fem.DofMapRestriction, list[multiphenicsx.fem.DofMapRestriction]]


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


def get_subdomains_pairs() -> tuple[
        tuple[typing.Optional[common.SubdomainType], typing.Optional[common.SubdomainType]], ...]:
    """Generate subdomain parametrization for tests on block/nest vectors and matrices."""
    return (
        # (unrestricted, unrestricted)
        (None, None),
        # (unrestricted, restricted)
        (None, common.CellsSubDomain(0.5, 0.75)),
        # (restricted, unrestricted)
        (common.CellsSubDomain(0.5, 0.75), None),
        # (restricted, restricted)
        (common.CellsSubDomain(0.5, 0.75), common.CellsSubDomain(0.75, 0.5))
    )


def get_function_spaces() -> tuple[common.FunctionSpaceGeneratorType, ...]:
    """Generate function space parametrization for tests on vectors and matrices."""
    return (
        lambda mesh: dolfinx.fem.functionspace(mesh, ("Lagrange", 1)),
        lambda mesh: dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, ))),
        lambda mesh: common.TaylorHoodFunctionSpace(mesh, ("Lagrange", 1))
    )


def get_function_spaces_pairs() -> typing.Iterator[
        tuple[common.FunctionSpaceGeneratorType, common.FunctionSpaceGeneratorType]]:
    """Generate function space parametrization for tests on block/nest vectors and matrices."""
    for i in get_function_spaces():
        for j in get_function_spaces():
            yield (i, j)


def get_function(
    V: dolfinx.fem.FunctionSpace, preprocess_x: typing.Optional[PreprocessXType] = None
) -> dolfinx.fem.Function:
    """Generate function employed in form definition."""
    if preprocess_x is None:
        def preprocess_x(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return x

    shape = V.ufl_element().reference_value_shape
    if len(shape) == 0:
        def f(x: npt.NDArray[np.float64]) -> npt.NDArray[  # type: ignore[name-defined]
                petsc4py.PETSc.ScalarType]:
            x = preprocess_x(x)
            return 2 * x[0] + 4 * x[1] * x[1]  # type: ignore[no-any-return]
    elif len(shape) == 1 and shape[0] == 2:
        def f(x: npt.NDArray[np.float64]) -> npt.NDArray[  # type: ignore[name-defined]
                petsc4py.PETSc.ScalarType]:
            x = preprocess_x(x)
            return np.stack([
                2 * x[0] + 4 * x[1] * x[1],
                3 * x[0] + 5 * x[1] * x[1]
            ], axis=0)
    elif len(shape) == 1 and shape[0] == 3:
        def f(x: npt.NDArray[np.float64]) -> npt.NDArray[  # type: ignore[name-defined]
                petsc4py.PETSc.ScalarType]:
            x = preprocess_x(x)
            return np.stack([
                2 * x[0] + 4 * x[1] * x[1],
                3 * x[0] + 5 * x[1] * x[1],
                7 * x[0] + 11 * x[1] * x[1]
            ], axis=0)
    u = dolfinx.fem.Function(V)
    try:
        u.interpolate(f)
    except RuntimeError:
        assert len(shape) == 1

        assert isinstance(V.ufl_element(), basix.ufl._MixedElement)
        rows = [np.prod(sub_element.reference_value_shape, dtype=int) for sub_element in V.ufl_element().sub_elements]
        rows = np.hstack(([0], np.cumsum(rows))).tolist()

        for i in range(len(rows) - 1):
            u.sub(i).interpolate(lambda x: f(x)[rows[i]:rows[i + 1], :])
    return u


def get_function_pair(
    V1: dolfinx.fem.FunctionSpace, V2: dolfinx.fem.FunctionSpace
) -> list[dolfinx.fem.Function]:
    """Generate functions employed in block form definition."""
    u1 = get_function(V1)
    u2 = get_function(V2, lambda x: np.flip(x, axis=1))
    return [u1, u2]


def get_linear_form(V: dolfinx.fem.FunctionSpace) -> dolfinx.fem.Form:
    """Generate linear forms employed in the vector test case."""
    v = ufl.TestFunction(V)
    f = get_function(V)
    return dolfinx.fem.form(ufl.inner(f, v) * ufl.dx)  # type: ignore[no-any-return]


def get_block_linear_form(V1: dolfinx.fem.FunctionSpace, V2: dolfinx.fem.FunctionSpace) -> dolfinx.fem.Form:
    """Generate two-by-one block linear forms employed in the block/nest vector test cases."""
    v1, v2 = ufl.TestFunction(V1), ufl.TestFunction(V2)
    assert V1.mesh == V2.mesh
    f1, f2 = get_function_pair(V1, V2)
    return dolfinx.fem.form([ufl.inner(f1, v1) * ufl.dx, ufl.inner(f2, v2) * ufl.dx])  # type: ignore[no-any-return]


def get_bilinear_form(V: dolfinx.fem.FunctionSpace) -> dolfinx.fem.Form:
    """Generate bilinear forms employed in the matrix test case."""
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = get_function(V)
    shape = V.ufl_element().reference_value_shape
    if len(shape) == 0:
        form = f * ufl.inner(u, v) * ufl.dx
    elif len(shape) == 1:
        form = sum(f[i] * ufl.inner(u[i], v[i]) for i in range(shape[0])) * ufl.dx
    return dolfinx.fem.form(form)  # type: ignore[no-any-return]


def get_block_bilinear_form(V1: dolfinx.fem.FunctionSpace, V2: dolfinx.fem.FunctionSpace) -> dolfinx.fem.Form:
    """Generate two-by-two block bilinear forms employed in the block/nest matrix test cases."""
    u1, u2 = ufl.TrialFunction(V1), ufl.TrialFunction(V2)
    v1, v2 = ufl.TestFunction(V1), ufl.TestFunction(V2)
    assert V1.mesh == V2.mesh
    f1, f2 = get_function_pair(V1, V2)

    def diff(v: ufl.Argument, index: int) -> typing.Union[  # type: ignore[no-any-unimported]
            ufl.indexed.Indexed, ufl.Argument]:
        if index >= 0:
            assert index in (0, 1)
            return v.dx(index)
        else:
            assert index == -1
            return v

    block_form = [[None, None], [None, None]]

    # (1, 1) block
    shape_1 = V1.ufl_element().reference_value_shape
    if len(shape_1) == 0:
        block_form[0][0] = f1 * ufl.inner(u1, v1) * ufl.dx
    elif len(shape_1) == 1:
        block_form[0][0] = sum(
            f1[i] * ufl.inner(u1[i], v1[i]) for i in range(shape_1[0])) * ufl.dx

    # (2, 2) block
    shape_2 = V2.ufl_element().reference_value_shape
    if len(shape_2) == 0:
        block_form[1][1] = f2 * ufl.inner(u2, v2) * ufl.dx
    elif len(shape_2) == 1:
        block_form[1][1] = sum(
            f2[i] * ufl.inner(u2[i], v2[i]) for i in range(shape_2[0])) * ufl.dx

    # (1, 2) and (2, 1) blocks
    if len(shape_1) == 0:
        if len(shape_2) == 0:
            block_form[0][1] = f1 * ufl.inner(u2, v1) * ufl.dx
            block_form[1][0] = f2 * ufl.inner(u1, v2) * ufl.dx
        elif len(shape_2) == 1:
            block_form[0][1] = sum(
                f1 * ufl.inner(u2[i], diff(v1, i - 1)) for i in range(shape_2[0])) * ufl.dx
            block_form[1][0] = sum(
                f2[i] * ufl.inner(diff(u1, i - 1), v2[i]) for i in range(shape_2[0])) * ufl.dx
    elif len(shape_1) == 1:
        if len(shape_2) == 0:
            block_form[0][1] = sum(
                f1[i] * ufl.inner(diff(u2, i - 1), v1[i]) for i in range(shape_1[0])) * ufl.dx
            block_form[1][0] = sum(
                f2 * ufl.inner(u1[i], diff(v2, i - 1)) for i in range(shape_1[0])) * ufl.dx
        elif len(shape_2) == 1:
            block_form[0][1] = sum(
                f1[i % shape_1[0]] * ufl.inner(u2[i % shape_2[0]], v1[i % shape_1[0]])
                for i in range(max(shape_1[0], shape_2[0]))) * ufl.dx
            block_form[1][0] = sum(
                f2[i % shape_2[0]] * ufl.inner(u1[i % shape_1[0]], v2[i % shape_2[0]])
                for i in range(max(shape_1[0], shape_2[0]))) * ufl.dx
    return dolfinx.fem.form(block_form)  # type: ignore[no-any-return]


def get_vec_types(plain: bool) -> tuple[typing.Optional[str], ...]:
    """Generate vector types to be used in test."""
    if plain:
        return (
            None,  # default, synonim of mpi
            "mpi"
        )
    else:
        return (
            # None,  # not valid
            "mpi",  # block vector
            "nest"  # nest vector
        )

def get_mat_types(plain: bool) -> tuple[typing.Optional[typing.Union[str, list[list[str]]]], ...]:
    """Generate matrix types to be used in test."""
    mat_types = (
        None,
        "seqaij" if mpi4py.MPI.COMM_WORLD.size == 1 else "mpiaij"
    )
    if not plain:
        mat_types = (  # type: ignore[assignment]
            *mat_types,
            "nest",
            [["seqaij" if mpi4py.MPI.COMM_WORLD.size == 1 else "mpiaij" for j in range(2)] for i in range(2)]
        )
    return mat_types


def locate_boundary_dofs(
    V: dolfinx.fem.FunctionSpace, collapsed_V: typing.Optional[dolfinx.fem.FunctionSpace] = None
) -> npt.NDArray[np.int32]:
    """Locate DOFs on the boundary."""
    entities_dim = V.mesh.topology.dim - 1
    entities = dolfinx.mesh.locate_entities(V.mesh, entities_dim, common.FacetsSubDomain(on_boundary=True))
    V.mesh.topology.create_connectivity(entities_dim, V.mesh.topology.dim)
    if collapsed_V is None:
        return dolfinx.fem.locate_dofs_topological(V, entities_dim, entities)
    else:
        return dolfinx.fem.locate_dofs_topological((V, collapsed_V), entities_dim, entities)


def get_boundary_conditions(offset: int = 0) -> tuple[DirichletBCsGeneratorType, ...]:
    """Generate boundary conditions employed in the non-block/nest test cases."""
    def _get_boundary_conditions(V: dolfinx.fem.FunctionSpace) -> list[dolfinx.fem.DirichletBC]:
        num_sub_elements = V.ufl_element().num_sub_elements
        if num_sub_elements == 0:
            bc1_fun = dolfinx.fem.Function(V)
            bc1_vector = bc1_fun.x.petsc_vec
            with bc1_vector.localForm() as local_form:
                local_form.set(1. + offset)
            bc1_vector.destroy()
            bdofs = locate_boundary_dofs(V)
            return [dolfinx.fem.dirichletbc(bc1_fun, bdofs)]
        else:
            bc1 = list()
            for i in range(num_sub_elements):
                Vi = V.sub(i)
                bc1_fun = dolfinx.fem.Function(Vi.collapse()[0])
                bc1_vector = bc1_fun.x.petsc_vec
                with bc1_vector.localForm() as local_form:
                    local_form.set(i + 1. + offset)
                bc1_vector.destroy()
                bdofs = locate_boundary_dofs(Vi, bc1_fun.function_space)
                bc1.append(dolfinx.fem.dirichletbc(bc1_fun, bdofs, Vi))
            return bc1

    return (lambda _: [],
            _get_boundary_conditions)


def get_boundary_conditions_pairs() -> tuple[DirichletBCsPairGeneratorType, ...]:
    """Generate boundary conditions employed in the block/nest test cases."""
    return (lambda _, __: [[], []],
            lambda V1, _: [get_boundary_conditions(offset=0)[1](V1), []],
            lambda _, V2: [[], get_boundary_conditions(offset=10)[1](V2)],
            lambda V1, V2: [get_boundary_conditions(offset=0)[1](V1),
                            get_boundary_conditions(offset=10)[1](V2)])


def get_apply_set_boundary_conditions_nonlinear_arguments() -> tuple[ApplySetDirichletBCsNonlinearArgumentsType, ...]:
    """Generate arguments x0 and restriction_x0 to be passed while applying or setting BCs for nonlinear problems."""
    return (
        lambda unrestricted_solution, restricted_solution, dofmap_restriction: (
            restricted_solution, dofmap_restriction
        ),
        lambda unrestricted_solution, restricted_solution, dofmap_restriction: (
            unrestricted_solution, None
        )
    )

def get_global_restricted_to_unrestricted(
    dofmap_restriction: DofMapRestrictionsType, comm: mpi4py.MPI.Intracomm
) -> dict[np.int32, np.int32]:
    """Allgather global map from restricted dofs to unrestricted dofs."""
    assert isinstance(dofmap_restriction, (multiphenicsx.fem.DofMapRestriction, list))
    if isinstance(dofmap_restriction, multiphenicsx.fem.DofMapRestriction):  # case of standard matrix/vector
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
        assert all(isinstance(
            dofmap_restriction_, multiphenicsx.fem.DofMapRestriction) for dofmap_restriction_ in dofmap_restriction)
        assert len(dofmap_restriction) == 2, "The code below is hardcoded for two blocks"
        dofmaps = [dofmap_restriction_.dofmap for dofmap_restriction_ in dofmap_restriction]
        bs = [dofmap.index_map_bs for dofmap in dofmaps]
        for (dofmap_restriction_, bs_) in zip(dofmap_restriction, bs):
            assert dofmap_restriction_.index_map_bs == bs_
        restricted_to_unrestricted_list = list()
        for (dofmap_restriction_, bs_) in zip(dofmap_restriction, bs):
            restricted_to_unrestricted = dict()
            for (restricted, unrestricted) in dofmap_restriction_.restricted_to_unrestricted.items():
                for s in range(bs_):
                    restricted_to_unrestricted[bs_ * restricted + s] = bs_ * unrestricted + s
            restricted_to_unrestricted_list.append(restricted_to_unrestricted)
        all_restricted_to_unrestricted = [comm.allgather(restricted_to_unrestricted)
                                          for restricted_to_unrestricted in restricted_to_unrestricted_list]
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


def to_numpy_vector(vec: petsc4py.PETSc.Vec) -> npt.NDArray[  # type: ignore[name-defined]
        petsc4py.PETSc.ScalarType]:
    """Convert distributed PETSc Vec to a dense allgather-ed numpy array."""
    local_np_vec = vec.getArray()
    comm = vec.getComm().tompi4py()
    return np.hstack(comm.allgather(local_np_vec))


def to_numpy_matrix(mat: petsc4py.PETSc.Mat) -> npt.NDArray[  # type: ignore[name-defined]
        petsc4py.PETSc.ScalarType]:
    """Convert distributed PETSc Mat to a dense allgather-ed numpy matrix."""
    ai, aj, av = mat.getValuesCSR()
    local_np_mat = scipy.sparse.csr_matrix((av, aj, ai), shape=(mat.getLocalSize()[0], mat.getSize()[1])).toarray()
    comm = mat.getComm().tompi4py()
    return np.vstack(comm.allgather(local_np_mat))


def assert_vector_equal(
    unrestricted_vector: petsc4py.PETSc.Vec, restricted_vector: petsc4py.PETSc.Vec,  # type: ignore[name-defined]
    dofmap_restriction: DofMapRestrictionsType
) -> None:
    """
    Verify assembly results for vector cases.

    Assert equality between the vector resulting from the assembly of a linear form with non-empty
    restriction argument and the vector resulting from the assembly of the same linear form with empty
    restriction argument accompanied by a postprocessing which manually removes discarded rows
    """
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


def assert_matrix_equal(
    unrestricted_matrix: petsc4py.PETSc.Mat, restricted_matrix: petsc4py.PETSc.Mat,  # type: ignore[name-defined]
    dofmap_restriction: tuple[DofMapRestrictionsType, DofMapRestrictionsType]
) -> None:
    """
    Verify assembly results for matrix cases.

    Assert equality between the matrix resulting from the assembly of a bilinear form with non-empty
    restriction argument and the matrix resulting from the assembly of the same bilinear form with empty
    restriction argument accompanied by a postprocessing which manually removes discarded rows/cols
    """
    if unrestricted_matrix.getType() == petsc4py.PETSc.Mat.Type.NEST:  # type: ignore[attr-defined]
        row_is, col_is = unrestricted_matrix.getNestISs()
        for i in range(len(row_is)):
            for j in range(len(col_is)):
                unrestricted_matrix_ij = unrestricted_matrix.getNestSubMatrix(i, j)
                restricted_matrix_ij = restricted_matrix.getNestSubMatrix(i, j)
                assert_matrix_equal(unrestricted_matrix_ij, restricted_matrix_ij,
                                    (dofmap_restriction[0][i], dofmap_restriction[1][j]))
    else:
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


def create_vector_from_dofmap(  # type: ignore[no-any-unimported]
    dofmap: typing.Union[dolfinx.fem.DofMap, dolfinx.cpp.fem.DofMap, multiphenicsx.fem.DofMapRestriction]
) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
    """Create a vector from a DofMap or DofMapRestriction, rather than a form."""
    # TODO remove this function and use upstream PR #3694 when ready
    return dolfinx.la.petsc.create_vector(dofmap.index_map, dofmap.index_map_bs)


def create_vector_from_dofmaps(  # type: ignore[no-any-unimported]
    dofmaps: list[
        typing.Union[dolfinx.fem.DofMap, dolfinx.cpp.fem.DofMap, multiphenicsx.fem.DofMapRestriction]],
    vec_type: str
) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
    """Create a vector from two DofMap or DofMapRestriction, rather than a form."""
    # TODO remove this function and use upstream PR #3694 when ready
    if vec_type == "mpi":
        return dolfinx.cpp.fem.petsc.create_vector_block(
            [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps])
    elif vec_type == "nest":
        return dolfinx.cpp.fem.petsc.create_vector_nest(
            [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps])
    else:
        raise RuntimeError("Invalid vector type")

def attach_attributes_to_solution_block_vector(
    solution: petsc4py.PETSc.Vec, fem_module: types.ModuleType,   # type: ignore[name-defined]
    block_linear_form: dolfinx.fem.Form
) -> None:
    """Attach the expected attributes to manually created block vectors."""
    if fem_module == dolfinx.fem:
        # dolfinx.fem.petsc assembly expects block vector to have an additional _blocks attribute
        dolfinx.fem.petsc._assign_block_data(block_linear_form, solution)  # type: ignore[arg-type]
    else:
        # multiphenicsx.fem.petsc assembly expects block vector to have an additional _dofmaps attribute
        function_spaces: list[dolfinx.fem.FunctionSpace] = dolfinx.fem.extract_function_spaces(
            block_linear_form)  # type: ignore[arg-type, assignment]
        dofmaps = [function_space.dofmap for function_space in function_spaces]
        solution.setAttr("_dofmaps", dofmaps)


@pytest.mark.parametrize("subdomain", get_subdomains())
@pytest.mark.parametrize("FunctionSpace", get_function_spaces())
@pytest.mark.parametrize("dirichlet_bcs", get_boundary_conditions())
@pytest.mark.parametrize(
    "apply_set_dirichlet_bcs_nonlinear_arguments", get_apply_set_boundary_conditions_nonlinear_arguments())
@pytest.mark.parametrize("unrestricted_fem_module", (dolfinx.fem, multiphenicsx.fem))
@pytest.mark.parametrize("restricted_fem_module", (multiphenicsx.fem, ))
@pytest.mark.parametrize("vec_type", get_vec_types(True))
def test_plain_vector_assembly_with_restriction(
    mesh: dolfinx.mesh.Mesh, subdomain: typing.Optional[common.SubdomainType],
    FunctionSpace: common.FunctionSpaceGeneratorType, dirichlet_bcs: DirichletBCsGeneratorType,
    apply_set_dirichlet_bcs_nonlinear_arguments: ApplySetDirichletBCsNonlinearArgumentsType,
    unrestricted_fem_module: types.ModuleType, restricted_fem_module: types.ModuleType,
    vec_type: str
) -> None:
    """Test assembly of a single linear form with restrictions."""
    V = FunctionSpace(mesh)
    active_dofs = common.ActiveDofs(V, subdomain)
    dofmap_restriction = restricted_fem_module.DofMapRestriction(V.dofmap, active_dofs)
    linear_form = get_linear_form(V)
    bilinear_form = get_bilinear_form(V)
    bcs = dirichlet_bcs(V)
    # Assembly without BCs
    unrestricted_vector = unrestricted_fem_module.petsc.assemble_vector(
        linear_form, kind=vec_type)
    dolfinx.la.petsc._ghost_update(
        unrestricted_vector, insert_mode=petsc4py.PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
        scatter_mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    restricted_vector = restricted_fem_module.petsc.assemble_vector(
        linear_form, kind=vec_type, restriction=dofmap_restriction)
    dolfinx.la.petsc._ghost_update(
        restricted_vector, insert_mode=petsc4py.PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
        scatter_mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    assert_vector_equal(unrestricted_vector, restricted_vector, dofmap_restriction)
    unrestricted_vector.destroy()
    restricted_vector.destroy()
    # BC application for linear problems
    unrestricted_vector_linear = unrestricted_fem_module.petsc.assemble_vector(
        linear_form, kind=vec_type)
    unrestricted_fem_module.petsc.apply_lifting(
        unrestricted_vector_linear, [bilinear_form], [bcs])
    dolfinx.la.petsc._ghost_update(
        unrestricted_vector_linear, insert_mode=petsc4py.PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
        scatter_mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    restricted_vector_linear = restricted_fem_module.petsc.assemble_vector(
        linear_form, kind=vec_type, restriction=dofmap_restriction)
    restricted_fem_module.petsc.apply_lifting(
        restricted_vector_linear, [bilinear_form], [bcs], restriction=dofmap_restriction)
    dolfinx.la.petsc._ghost_update(
        restricted_vector_linear, insert_mode=petsc4py.PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
        scatter_mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    assert_vector_equal(unrestricted_vector_linear, restricted_vector_linear, dofmap_restriction)
    unrestricted_fem_module.petsc.set_bc(unrestricted_vector_linear, bcs)
    restricted_fem_module.petsc.set_bc(restricted_vector_linear, bcs, restriction=dofmap_restriction)
    assert_vector_equal(unrestricted_vector_linear, restricted_vector_linear, dofmap_restriction)
    unrestricted_vector_linear.destroy()
    restricted_vector_linear.destroy()
    # BC application for nonlinear problems
    unrestricted_solution = create_vector_from_dofmap(V.dofmap)
    restricted_solution = create_vector_from_dofmap(dofmap_restriction)
    bc_vector = get_function(V).x.petsc_vec
    with unrestricted_solution.localForm() as unrestricted_solution_local, \
            bc_vector.localForm() as function_local:
        active_dofs_bs = [
            V.dofmap.index_map_bs * d + s
            for d in active_dofs.astype(np.int32) for s in range(V.dofmap.index_map_bs)]
        unrestricted_solution_local[active_dofs_bs] = function_local[active_dofs_bs]
        with restricted_fem_module.petsc.VecSubVectorWrapper(
                restricted_solution, V.dofmap, dofmap_restriction) as restricted_solution_wrapper:
            restricted_solution_wrapper[:] = unrestricted_solution_local
    unrestricted_vector_nonlinear = unrestricted_fem_module.petsc.assemble_vector(
        linear_form, kind=vec_type)
    unrestricted_fem_module.petsc.apply_lifting(
        unrestricted_vector_nonlinear, [bilinear_form], [bcs], [unrestricted_solution])
    dolfinx.la.petsc._ghost_update(
        unrestricted_vector_nonlinear, insert_mode=petsc4py.PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
        scatter_mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    restricted_vector_nonlinear = restricted_fem_module.petsc.assemble_vector(
        linear_form, kind=vec_type, restriction=dofmap_restriction)
    x0_arg, restriction_x0_arg = apply_set_dirichlet_bcs_nonlinear_arguments(
        unrestricted_solution, restricted_solution, dofmap_restriction)
    x0_arg_list = [x0_arg]
    if restriction_x0_arg is not None:
        restriction_x0_arg_list = [restriction_x0_arg]
    else:
        restriction_x0_arg_list = None
    restricted_fem_module.petsc.apply_lifting(
        restricted_vector_nonlinear, [bilinear_form], [bcs], x0=x0_arg_list, restriction=dofmap_restriction,
        restriction_x0=restriction_x0_arg_list)
    dolfinx.la.petsc._ghost_update(
        restricted_vector_nonlinear, insert_mode=petsc4py.PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
        scatter_mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    assert_vector_equal(unrestricted_vector_nonlinear, restricted_vector_nonlinear, dofmap_restriction)
    unrestricted_fem_module.petsc.set_bc(unrestricted_vector_nonlinear, bcs, unrestricted_solution)
    restricted_fem_module.petsc.set_bc(
        restricted_vector_nonlinear, bcs, x0=x0_arg, restriction=dofmap_restriction,
        restriction_x0=restriction_x0_arg)
    assert_vector_equal(unrestricted_vector_nonlinear, restricted_vector_nonlinear, dofmap_restriction)
    unrestricted_vector_nonlinear.destroy()
    restricted_vector_nonlinear.destroy()
    unrestricted_solution.destroy()
    restricted_solution.destroy()
    bc_vector.destroy()


@pytest.mark.parametrize("subdomains", get_subdomains_pairs())
@pytest.mark.parametrize("FunctionSpaces", get_function_spaces_pairs())
@pytest.mark.parametrize("dirichlet_bcs", get_boundary_conditions_pairs())
@pytest.mark.parametrize(
    "apply_set_dirichlet_bcs_nonlinear_arguments", get_apply_set_boundary_conditions_nonlinear_arguments())
@pytest.mark.parametrize("unrestricted_fem_module", (dolfinx.fem, multiphenicsx.fem))
@pytest.mark.parametrize("restricted_fem_module", (multiphenicsx.fem, ))
@pytest.mark.parametrize("vec_type", get_vec_types(False))
def test_block_nest_vector_assembly_with_restriction(
    mesh: dolfinx.mesh.Mesh,
    subdomains: tuple[typing.Optional[common.SubdomainType], typing.Optional[common.SubdomainType]],
    FunctionSpaces: tuple[common.FunctionSpaceGeneratorType, common.FunctionSpaceGeneratorType],
    dirichlet_bcs: DirichletBCsPairGeneratorType,
    apply_set_dirichlet_bcs_nonlinear_arguments: ApplySetDirichletBCsNonlinearArgumentsType,
    unrestricted_fem_module: types.ModuleType, restricted_fem_module: types.ModuleType,
    vec_type: str
) -> None:
    """Test block/nest assembly of a two-by-one block linear form with restrictions."""
    V = [FunctionSpace(mesh) for FunctionSpace in FunctionSpaces]
    dofmaps = [V_.dofmap for V_ in V]
    active_dofs = [common.ActiveDofs(V_, subdomain) for (V_, subdomain) in zip(V, subdomains)]
    dofmap_restriction = [
        restricted_fem_module.DofMapRestriction(V_.dofmap, active_dofs_) for (V_, active_dofs_) in zip(V, active_dofs)]
    block_linear_form = get_block_linear_form(*V)
    block_bilinear_form = get_block_bilinear_form(*V)
    bcs = dirichlet_bcs(*V)
    # Assembly without BCs
    unrestricted_vector = unrestricted_fem_module.petsc.assemble_vector(
        block_linear_form, kind=vec_type)
    dolfinx.la.petsc._ghost_update(
        unrestricted_vector, insert_mode=petsc4py.PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
        scatter_mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    restricted_vector = restricted_fem_module.petsc.assemble_vector(
        block_linear_form, kind=vec_type, restriction=dofmap_restriction)
    dolfinx.la.petsc._ghost_update(
        restricted_vector, insert_mode=petsc4py.PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
        scatter_mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    assert_vector_equal(unrestricted_vector, restricted_vector, dofmap_restriction)
    unrestricted_vector.destroy()
    restricted_vector.destroy()
    # BC application for linear problems
    unrestricted_vector_linear = unrestricted_fem_module.petsc.assemble_vector(
        block_linear_form, kind=vec_type)
    unrestricted_fem_module.petsc.apply_lifting(
        unrestricted_vector_linear, block_bilinear_form, bcs)
    dolfinx.la.petsc._ghost_update(
        unrestricted_vector_linear, insert_mode=petsc4py.PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
        scatter_mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    restricted_vector_linear = restricted_fem_module.petsc.assemble_vector(
        block_linear_form, kind=vec_type, restriction=dofmap_restriction)
    restricted_fem_module.petsc.apply_lifting(
        restricted_vector_linear, block_bilinear_form, bcs, restriction=dofmap_restriction)
    dolfinx.la.petsc._ghost_update(
        restricted_vector_linear, insert_mode=petsc4py.PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
        scatter_mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    assert_vector_equal(unrestricted_vector_linear, restricted_vector_linear, dofmap_restriction)
    unrestricted_fem_module.petsc.set_bc(
        unrestricted_vector_linear, bcs)
    restricted_fem_module.petsc.set_bc(
        restricted_vector_linear, bcs, restriction=dofmap_restriction)
    assert_vector_equal(unrestricted_vector_linear, restricted_vector_linear, dofmap_restriction)
    unrestricted_vector_linear.destroy()
    restricted_vector_linear.destroy()
    # Assembly for nonlinear problems
    unrestricted_solution = create_vector_from_dofmaps([V_.dofmap for V_ in V], vec_type)
    restricted_solution = create_vector_from_dofmaps(dofmap_restriction, vec_type)
    if vec_type == "mpi":
        attach_attributes_to_solution_block_vector(unrestricted_solution, unrestricted_fem_module, block_linear_form)
        if unrestricted_fem_module == dolfinx.fem:
            attach_attributes_to_solution_block_vector(unrestricted_solution, restricted_fem_module, block_linear_form)
        attach_attributes_to_solution_block_vector(restricted_solution, restricted_fem_module, block_linear_form)
        BlockNestVecSubVectorWrapper = restricted_fem_module.petsc.BlockVecSubVectorWrapper
    else:
        BlockNestVecSubVectorWrapper = restricted_fem_module.petsc.NestVecSubVectorWrapper
    with BlockNestVecSubVectorWrapper(unrestricted_solution, dofmaps) as unrestricted_solution_wrapper:
        for (active_dofs_sub, unrestricted_solution_sub_local, function_sub, dofmap_sub) in zip(
                active_dofs, unrestricted_solution_wrapper, get_function_pair(*V), dofmaps):
            active_dofs_sub_bs = [
                dofmap_sub.index_map_bs * d + s
                for d in active_dofs_sub.astype(np.int32) for s in range(dofmap_sub.index_map_bs)]
            function_sub_vector = function_sub.x.petsc_vec
            with function_sub_vector.localForm() as function_sub_local:
                unrestricted_solution_sub_local[active_dofs_sub_bs] = function_sub_local[active_dofs_sub_bs]
            function_sub_vector.destroy()
    with BlockNestVecSubVectorWrapper(
            restricted_solution, dofmaps, dofmap_restriction) as restricted_solution_wrapper, \
            BlockNestVecSubVectorWrapper(unrestricted_solution, dofmaps) as unrestricted_solution_wrapper:
        for (restricted_solution_sub, unrestricted_solution_sub) in zip(
                restricted_solution_wrapper, unrestricted_solution_wrapper):
            restricted_solution_sub[:] = unrestricted_solution_sub
    unrestricted_vector_nonlinear = unrestricted_fem_module.petsc.assemble_vector(
        block_linear_form, kind=vec_type)
    unrestricted_fem_module.petsc.apply_lifting(
        unrestricted_vector_nonlinear, block_bilinear_form, bcs, unrestricted_solution)
    dolfinx.la.petsc._ghost_update(
        unrestricted_vector_nonlinear, insert_mode=petsc4py.PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
        scatter_mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    restricted_vector_nonlinear = restricted_fem_module.petsc.assemble_vector(
        block_linear_form, kind=vec_type, restriction=dofmap_restriction)
    x0_arg, restriction_x0_arg = apply_set_dirichlet_bcs_nonlinear_arguments(
        unrestricted_solution, restricted_solution, dofmap_restriction)
    restricted_fem_module.petsc.apply_lifting(
        restricted_vector_nonlinear, block_bilinear_form, bcs, x0=x0_arg, restriction=dofmap_restriction,
        restriction_x0=restriction_x0_arg)
    dolfinx.la.petsc._ghost_update(
        restricted_vector_nonlinear, insert_mode=petsc4py.PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
        scatter_mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    assert_vector_equal(unrestricted_vector_nonlinear, restricted_vector_nonlinear, dofmap_restriction)
    unrestricted_fem_module.petsc.set_bc(
        unrestricted_vector_nonlinear, bcs, unrestricted_solution)
    restricted_fem_module.petsc.set_bc(
        restricted_vector_nonlinear, bcs, x0=x0_arg, restriction=dofmap_restriction,
        restriction_x0=restriction_x0_arg)
    assert_vector_equal(unrestricted_vector_nonlinear, restricted_vector_nonlinear, dofmap_restriction)
    unrestricted_vector_nonlinear.destroy()
    restricted_vector_nonlinear.destroy()
    unrestricted_solution.destroy()
    restricted_solution.destroy()


@pytest.mark.parametrize("subdomain", get_subdomains())
@pytest.mark.parametrize("FunctionSpace", get_function_spaces())
@pytest.mark.parametrize("dirichlet_bcs", get_boundary_conditions())
@pytest.mark.parametrize("unrestricted_fem_module", (dolfinx.fem, multiphenicsx.fem))
@pytest.mark.parametrize("restricted_fem_module", (multiphenicsx.fem, ))
@pytest.mark.parametrize("mat_type", get_mat_types(True))
def test_plain_matrix_assembly_with_restriction(
    mesh: dolfinx.mesh.Mesh, subdomain: typing.Optional[common.SubdomainType],
    FunctionSpace: common.FunctionSpaceGeneratorType, dirichlet_bcs: DirichletBCsGeneratorType,
    unrestricted_fem_module: types.ModuleType, restricted_fem_module: types.ModuleType, mat_type: str
) -> None:
    """Test assembly of a single bilinear form with restrictions."""
    V = FunctionSpace(mesh)
    active_dofs = common.ActiveDofs(V, subdomain)
    dofmap_restriction = restricted_fem_module.DofMapRestriction(V.dofmap, active_dofs)
    form = get_bilinear_form(V)
    bcs = dirichlet_bcs(V)
    unrestricted_matrix = unrestricted_fem_module.petsc.assemble_matrix(form, bcs=bcs, kind=mat_type)
    unrestricted_matrix.assemble()
    restricted_matrix = restricted_fem_module.petsc.assemble_matrix(
        form, bcs=bcs, kind=mat_type, restriction=(dofmap_restriction, dofmap_restriction))
    restricted_matrix.assemble()
    assert_matrix_equal(unrestricted_matrix, restricted_matrix, (dofmap_restriction, dofmap_restriction))
    unrestricted_matrix.destroy()
    restricted_matrix.destroy()


@pytest.mark.parametrize("subdomains", get_subdomains_pairs())
@pytest.mark.parametrize("FunctionSpaces", get_function_spaces_pairs())
@pytest.mark.parametrize("dirichlet_bcs", get_boundary_conditions_pairs())
@pytest.mark.parametrize("unrestricted_fem_module", (dolfinx.fem, multiphenicsx.fem))
@pytest.mark.parametrize("restricted_fem_module", (multiphenicsx.fem, ))
@pytest.mark.parametrize("mat_type", get_mat_types(False))
def test_block_nest_matrix_assembly_with_restriction(
    mesh: dolfinx.mesh.Mesh,
    subdomains: tuple[typing.Optional[common.SubdomainType], typing.Optional[common.SubdomainType]],
    FunctionSpaces: tuple[common.FunctionSpaceGeneratorType, common.FunctionSpaceGeneratorType],
    dirichlet_bcs: DirichletBCsPairGeneratorType, unrestricted_fem_module: types.ModuleType,
    restricted_fem_module: types.ModuleType, mat_type: str
) -> None:
    """Test block/nest assembly of a two-by-two block bilinear form with restrictions."""
    V = [FunctionSpace(mesh) for FunctionSpace in FunctionSpaces]
    active_dofs = [common.ActiveDofs(V_, subdomain) for (V_, subdomain) in zip(V, subdomains)]
    dofmap_restriction = [
        restricted_fem_module.DofMapRestriction(V_.dofmap, active_dofs_) for (V_, active_dofs_) in zip(V, active_dofs)]
    block_form = get_block_bilinear_form(*V)
    bcs = [bc for bcs in dirichlet_bcs(*V) for bc in bcs]
    unrestricted_matrix = unrestricted_fem_module.petsc.assemble_matrix(
        block_form, bcs=bcs, kind=mat_type)
    unrestricted_matrix.assemble()
    restricted_matrix = restricted_fem_module.petsc.assemble_matrix(
        block_form, bcs=bcs, kind=mat_type, restriction=(dofmap_restriction, dofmap_restriction))
    restricted_matrix.assemble()
    assert_matrix_equal(unrestricted_matrix, restricted_matrix, (dofmap_restriction, dofmap_restriction))
    unrestricted_matrix.destroy()
    restricted_matrix.destroy()

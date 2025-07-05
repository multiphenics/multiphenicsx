# Copyright (C) 2016-2025 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Assembly functions for variational forms."""

import collections.abc
import contextlib
import functools
import types
import typing

import dolfinx.cpp as dcpp
import dolfinx.fem
import dolfinx.fem.assemble
import dolfinx.la
import dolfinx.la.petsc
import numpy as np
import numpy.typing as npt
import petsc4py.PETSc

from multiphenicsx.cpp import cpp_library as mcpp

DolfinxRank1FormsType = typing.Union[
    dolfinx.fem.Form, typing.Sequence[dolfinx.fem.Form]
]
DolfinxRank2FormsType = typing.Union[
    dolfinx.fem.Form, typing.Sequence[typing.Sequence[dolfinx.fem.Form]]
]
DolfinxVectorKindType = typing.Optional[str]
DolfinxMatrixKindType = typing.Optional[typing.Union[str, typing.Sequence[typing.Sequence[str]]]]
DolfinxConstantsType_Base = npt.NDArray[petsc4py.PETSc.ScalarType]  # type: ignore[no-any-unimported]
DolfinxConstantsType = typing.Optional[
    typing.Union[DolfinxConstantsType_Base, typing.Sequence[typing.Optional[DolfinxConstantsType_Base]]]
]
DolfinxCoefficientsType_Base = dict[  # type: ignore[no-any-unimported]
    tuple[dolfinx.fem.IntegralType, int],
    npt.NDArray[petsc4py.PETSc.ScalarType]
]
DolfinxCoefficientsType = typing.Optional[
    typing.Union[DolfinxCoefficientsType_Base, typing.Sequence[typing.Optional[DolfinxCoefficientsType_Base]]]
]
MultiphenicsxRank1RestrictionsType = typing.Optional[typing.Union[  # type: ignore[no-any-unimported]
    typing.Optional[mcpp.fem.DofMapRestriction],
    typing.Optional[typing.Sequence[mcpp.fem.DofMapRestriction]]
]]
MultiphenicsxRank2RestrictionsType = typing.Optional[typing.Union[  # type: ignore[no-any-unimported]
    typing.Optional[tuple[mcpp.fem.DofMapRestriction, mcpp.fem.DofMapRestriction]],
    typing.Optional[tuple[
        typing.Sequence[mcpp.fem.DofMapRestriction],
        typing.Sequence[mcpp.fem.DofMapRestriction]
    ]
]]]


def _get_block_function_spaces(block_form: typing.Sequence[typing.Any]) -> list[typing.Any]:
    if isinstance(block_form[0], collections.abc.Sequence):
        return _get_block_function_spaces_rank_2(block_form)
    else:
        return _get_block_function_spaces_rank_1(block_form)


def _get_block_function_spaces_rank_1(
    block_form: typing.Sequence[dolfinx.fem.Form]
) -> list[dolfinx.fem.FunctionSpace]:
    assert all(isinstance(block_form_, dolfinx.fem.Form) for block_form_ in block_form)
    return [form.function_spaces[0] for form in block_form]


def _get_block_function_spaces_rank_2(
    block_form: typing.Sequence[typing.Sequence[dolfinx.fem.Form]]
) -> list[list[dolfinx.fem.FunctionSpace]]:
    assert all(isinstance(block_form_, typing.Sequence) for block_form_ in block_form)
    assert all(
        isinstance(form, dolfinx.fem.Form) or form is None for block_form_ in block_form for form in block_form_)
    a = block_form
    rows = len(a)
    cols = len(a[0])
    assert all(len(a_i) == cols for a_i in a)
    assert all(
        a[i][j] is None or a[i][j].rank == 2 for i in range(rows) for j in range(cols))
    function_spaces_0 = list()
    for i in range(rows):
        function_spaces_0_i = None
        for j in range(cols):
            if a[i][j] is not None:
                function_spaces_0_i = a[i][j].function_spaces[0]
                break
        assert function_spaces_0_i is not None
        function_spaces_0.append(function_spaces_0_i)
    function_spaces_1 = list()
    for j in range(cols):
        function_spaces_1_j = None
        for i in range(rows):
            if a[i][j] is not None:
                function_spaces_1_j = a[i][j].function_spaces[1]
                break
        assert function_spaces_1_j is not None
        function_spaces_1.append(function_spaces_1_j)
    function_spaces = [function_spaces_0, function_spaces_1]
    assert all(a[i][j] is None or a[i][j].function_spaces[0] == function_spaces[0][i]
               for i in range(rows) for j in range(cols))
    assert all(a[i][j] is None or a[i][j].function_spaces[1] == function_spaces[1][j]
               for i in range(rows) for j in range(cols))
    return function_spaces


def _same_dofmap(  # type: ignore[no-any-unimported]
    dofmap1: typing.Union[dolfinx.fem.DofMap, dcpp.fem.DofMap],
    dofmap2: typing.Union[dolfinx.fem.DofMap, dcpp.fem.DofMap]
) -> bool:
    try:
        dofmap1 = dofmap1._cpp_object
    except AttributeError:
        pass

    try:
        dofmap2 = dofmap2._cpp_object
    except AttributeError:
        pass

    return dofmap1 == dofmap2


# -- Vector instantiation ----------------------------------------------------

def create_vector(  # type: ignore[no-any-unimported]
    L: DolfinxRank1FormsType, kind: DolfinxVectorKindType = None,
    restriction: MultiphenicsxRank1RestrictionsType = None
) -> petsc4py.PETSc.Vec:
    """
    Create a PETSc vector that is compatible with a linear form(s) and a restriction.

    Three cases are supported:

    1. For a single linear form ``L``, if ``kind`` is ``None`` or is
       ``PETSc.Vec.Type.MPI``, a ghosted PETSc vector which is
       compatible with ``L`` is created.

    2. If ``L`` is a sequence of linear forms and ``kind`` is ``None``
       or is ``PETSc.Vec.Type.MPI``, a ghosted PETSc vector which is
       compatible with ``L`` is created. The created vector ``b`` is
       initialized such that on each MPI process ``b = [b_0, b_1, ...,
       b_n, b_0g, b_1g, ..., b_ng]``, where ``b_i`` are the entries
       associated with the 'owned' degrees-of-freedom for ``L[i]`` and
       ``b_ig`` are the 'unowned' (ghost) entries for ``L[i]``.

    3. If ``L`` is a sequence of linear forms and ``kind`` is
       ``PETSc.Vec.Type.NEST``, a PETSc nested vector (a 'nest' of
       ghosted PETSc vectors) which is compatible with ``L`` is created.

    Parameters
    ----------
    L
        Linear form or a sequence of linear forms.
    kind
        PETSc vector type (``VecType``) to create.
    restriction
        A dofmap restriction. If not provided, the unrestricted vector will be created.

    Returns
    -------
        A PETSc vector with a layout that is compatible with ``L`` and restriction
        `restriction`. The vector is not initialised to zero.
    """
    if isinstance(L, collections.abc.Sequence):
        function_spaces = _get_block_function_spaces(L)
        dofmaps = [function_space.dofmap for function_space in function_spaces]
        if restriction is None:
            index_maps = [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps]
        else:
            assert isinstance(restriction, collections.abc.Sequence)
            assert len(restriction) == len(dofmaps)
            assert all(
                _same_dofmap(restriction_.dofmap, dofmap) for (restriction_, dofmap) in zip(restriction, dofmaps))
            index_maps = [(restriction_.index_map, restriction_.index_map_bs) for restriction_ in restriction]
        if kind == petsc4py.PETSc.Vec.Type.NEST:
            return dcpp.fem.petsc.create_vector_nest(index_maps)
        elif kind == petsc4py.PETSc.Vec.Type.MPI:
            b = dcpp.fem.petsc.create_vector_block(index_maps)
            b.setAttr("_dofmaps", dofmaps)
            return b
        else:  # pragma: no cover
            raise NotImplementedError(
                "Vector type must be specified for blocked/nested assembly."
                f"Vector type '{kind}' not supported."
                "Did you mean 'nest' or 'mpi'?"
            )
    else:
        dofmap = L.function_spaces[0].dofmap
        if restriction is None:
            index_map = dofmap.index_map
            index_map_bs = dofmap.index_map_bs
        else:
            assert not isinstance(restriction, collections.abc.Sequence)
            assert _same_dofmap(restriction.dofmap, dofmap)
            index_map = restriction.index_map
            index_map_bs = restriction.index_map_bs
        return dolfinx.la.petsc.create_vector(index_map, index_map_bs)


# -- Matrix instantiation ----------------------------------------------------

def create_matrix(  # type: ignore[no-any-unimported]
    a: DolfinxRank2FormsType, kind: DolfinxMatrixKindType = None,
    restriction: MultiphenicsxRank2RestrictionsType = None
) -> petsc4py.PETSc.Mat:
    """
    Create a PETSc matrix that is compatible with the (sequence) of bilinear form(s) and a restriction.

    Three cases are supported:

    1. For a single bilinear form, it creates a compatible PETSc matrix
       of type ``kind``.
    2. For a rectangular array of bilinear forms, if ``kind`` is
       ``PETSc.Mat.Type.NEST`` or ``kind`` is an array of PETSc ``Mat``
       types (with the same shape as ``a``), a matrix of type
       ``PETSc.Mat.Type.NEST`` is created. The matrix is compatible
       with the forms ``a``.
    3. For a rectangular array of bilinear forms, it create a single
       (non-nested) matrix of type ``kind`` that is compatible with the
       array of for forms ``a``. If ``kind`` is ``None``, then the
       matrix is the default type.

       In this case, the matrix is arranged::

             A = [a_00 ... a_0n]
                 [a_10 ... a_1n]
                 [     ...     ]
                 [a_m0 ..  a_mn]

    Parameters
    ----------
    a
        A bilinear form or a nested sequence of bilinear forms.
    kind
        The PETSc matrix type (``MatType``).
    restriction
        A dofmap restriction. If not provided, the unrestricted matrix will be created.

    Returns
    -------
    :
        A PETSc matrix with a layout that is compatible with `a` and restriction `restriction`.
    """
    if isinstance(a, collections.abc.Sequence):
        function_spaces = _get_block_function_spaces(a)
        rows, cols = len(function_spaces[0]), len(function_spaces[1])
        mesh = None
        for j in range(cols):
            for i in range(rows):
                if a[i][j] is not None:
                    mesh = a[i][j].mesh
                    break
        assert mesh is not None
        assert all(a[i][j] is None or a[i][j].mesh == mesh for i in range(rows) for j in range(cols))
        assert all(function_space.mesh == mesh for function_space in function_spaces[0])
        assert all(function_space.mesh == mesh for function_space in function_spaces[1])
        if restriction is None:
            index_maps = (
                [function_spaces[0][i].dofmap.index_map for i in range(rows)],
                [function_spaces[1][j].dofmap.index_map for j in range(cols)])
            index_maps_bs = (
                [function_spaces[0][i].dofmap.index_map_bs for i in range(rows)],
                [function_spaces[1][j].dofmap.index_map_bs for j in range(cols)])
            dofmaps_list = (
                [function_spaces[0][i].dofmap.map() for i in range(rows)],
                [function_spaces[1][j].dofmap.map() for j in range(cols)])
            dofmaps_bounds = (
                [np.arange(dofmaps_list[0][i].shape[0] + 1, dtype=np.uint64) * dofmaps_list[0][i].shape[1]
                 for i in range(rows)],
                [np.arange(dofmaps_list[1][j].shape[0] + 1, dtype=np.uint64) * dofmaps_list[1][j].shape[1]
                 for j in range(cols)])
        else:
            assert len(restriction) == 2
            assert len(restriction[0]) == rows
            assert len(restriction[1]) == cols
            index_maps = (
                [restriction[0][i].index_map for i in range(rows)],
                [restriction[1][j].index_map for j in range(cols)])
            index_maps_bs = (
                [restriction[0][i].index_map_bs for i in range(rows)],
                [restriction[1][j].index_map_bs for j in range(cols)])
            dofmaps_list = (
                [restriction[0][i].map()[0] for i in range(rows)],
                [restriction[1][j].map()[0] for j in range(cols)])
            dofmaps_bounds = (
                [restriction[0][i].map()[1] for i in range(rows)],
                [restriction[1][j].map()[1] for j in range(cols)])
        a_cpp = [[None if form is None else form._cpp_object for form in forms] for forms in a]
        if kind == petsc4py.PETSc.Mat.Type.NEST:  # create nest matrix with default types
            return mcpp.fem.petsc.create_matrix_nest(
                a_cpp, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds, None)
        else:
            if kind is None or isinstance(kind, str):  # create block matrix
                return mcpp.fem.petsc.create_matrix_block(
                    a_cpp, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds, kind)
            else:  # create nest matrix with provided types
                return mcpp.fem.petsc.create_matrix_nest(
                    a_cpp, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds, kind)
    else:
        assert a.rank == 2
        function_spaces = a.function_spaces
        assert all(function_space.mesh == a.mesh for function_space in function_spaces)
        if restriction is None:
            index_maps = [  # type: ignore[assignment]
                function_space.dofmap.index_map for function_space in function_spaces]
            index_maps_bs = [  # type: ignore[assignment]
                function_space.dofmap.index_map_bs for function_space in function_spaces]
            dofmaps_list = [  # type: ignore[assignment]
                function_space.dofmap.map() for function_space in function_spaces]  # type: ignore[attr-defined]
            dofmaps_bounds = [  # type: ignore[assignment]
                np.arange(dofmap_list.shape[0] + 1, dtype=np.uint64) * dofmap_list.shape[1]  # type: ignore
                for dofmap_list in dofmaps_list]
        else:
            assert len(restriction) == 2
            index_maps = [  # type: ignore[assignment]
                restriction_.index_map for restriction_ in restriction]  # type: ignore[union-attr]
            index_maps_bs = [  # type: ignore[assignment]
                restriction_.index_map_bs for restriction_ in restriction]  # type: ignore[union-attr]
            dofmaps_list = [  # type: ignore[assignment]
                restriction_.map()[0] for restriction_ in restriction]  # type: ignore[union-attr]
            dofmaps_bounds = [  # type: ignore[assignment]
                restriction_.map()[1] for restriction_ in restriction]  # type: ignore[union-attr]
        return mcpp.fem.petsc.create_matrix(
            a._cpp_object, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds, kind)


# -- Vector assembly ---------------------------------------------------------

def _VecSubVectorWrapperBase(CppWrapperClass: type) -> type:

    class _VecSubVectorWrapperBase_Class:
        """Wrap a PETSc Vec object."""

        def __init__(  # type: ignore[no-any-unimported]
            self, b: petsc4py.PETSc.Vec, unrestricted_index_set: petsc4py.PETSc.IS,
            restricted_index_set: typing.Optional[petsc4py.PETSc.IS] = None,
            unrestricted_to_restricted: typing.Optional[dict[int, int]] = None,
            unrestricted_to_restricted_bs: typing.Optional[int] = None
        ) -> None:
            if restricted_index_set is None:
                assert unrestricted_to_restricted is None
                assert unrestricted_to_restricted_bs is None
                self._cpp_object = CppWrapperClass(b, unrestricted_index_set)
            else:
                self._cpp_object = CppWrapperClass(
                    b, unrestricted_index_set, restricted_index_set,
                    unrestricted_to_restricted, unrestricted_to_restricted_bs)

        def __enter__(self) -> npt.NDArray[petsc4py.PETSc.ScalarType]:  # type: ignore[no-any-unimported]
            """Return Vec content when entering the context."""
            return self._cpp_object.content  # type: ignore[no-any-return]

        def __exit__(
            self, exception_type: type[BaseException], exception_value: BaseException,
            traceback: types.TracebackType
        ) -> None:
            """Do nothing when leaving the context."""
            pass

    return _VecSubVectorWrapperBase_Class


_VecSubVectorReadWrapper = _VecSubVectorWrapperBase(mcpp.la.petsc.VecSubVectorReadWrapper)


class _VecSubVectorWrapper(_VecSubVectorWrapperBase(mcpp.la.petsc.VecSubVectorWrapper)):  # type: ignore[misc]
    def __exit__(
        self, exception_type: type[BaseException], exception_value: BaseException,
        traceback: types.TracebackType
    ) -> None:
        """Restore the Vec content when leaving the context."""
        self._cpp_object.restore()


def VecSubVectorWrapperBase(_VecSubVectorWrapperClass: type) -> type:
    """Return the base class to wrap VecSubVectorWrapper or VecSubVectorReadWrapper."""

    class VecSubVectorWrapperBase_Class:
        """Wrap a PETSc Vec object."""

        def __init__(  # type: ignore[no-any-unimported]
            self, b: typing.Union[petsc4py.PETSc.Vec, None], dofmap: dcpp.fem.DofMap,
            restriction: typing.Optional[mcpp.fem.DofMapRestriction] = None,
            ghosted: bool = True
        ) -> None:
            if b is None:
                self._wrapper = None
            else:
                if restriction is None:  # pragma: no cover
                    index_map = (dofmap.index_map, dofmap.index_map_bs)
                    index_set = mcpp.la.petsc.create_index_sets(
                        [index_map], [dofmap.index_map_bs], ghosted=ghosted,
                        ghost_block_layout=mcpp.la.petsc.GhostBlockLayout.trailing)[0]
                    self._wrapper = _VecSubVectorWrapperClass(b, index_set)
                    self._unrestricted_index_set = index_set
                    self._restricted_index_set = None
                    self._unrestricted_to_restricted = None
                    self._unrestricted_to_restricted_bs = None
                else:
                    assert _same_dofmap(dofmap, restriction.dofmap)
                    unrestricted_index_map = (dofmap.index_map, dofmap.index_map_bs)
                    unrestricted_index_set = mcpp.la.petsc.create_index_sets(
                        [unrestricted_index_map], [dofmap.index_map_bs], ghosted=ghosted,
                        ghost_block_layout=mcpp.la.petsc.GhostBlockLayout.trailing)[0]
                    restricted_index_map = (restriction.index_map, restriction.index_map_bs)
                    restricted_index_set = mcpp.la.petsc.create_index_sets(
                        [restricted_index_map], [restriction.index_map_bs], ghosted=ghosted,
                        ghost_block_layout=mcpp.la.petsc.GhostBlockLayout.trailing)[0]
                    unrestricted_to_restricted = restriction.unrestricted_to_restricted
                    unrestricted_to_restricted_bs = restriction.index_map_bs
                    self._wrapper = _VecSubVectorWrapperClass(
                        b, unrestricted_index_set, restricted_index_set,
                        unrestricted_to_restricted, unrestricted_to_restricted_bs)
                    self._unrestricted_index_set = unrestricted_index_set
                    self._restricted_index_set = restricted_index_set
                    self._unrestricted_to_restricted = unrestricted_to_restricted
                    self._unrestricted_to_restricted_bs = unrestricted_to_restricted_bs

        def __enter__(self) -> typing.Optional[  # type: ignore[no-any-unimported]
                npt.NDArray[petsc4py.PETSc.ScalarType]]:
            """Return Vec content when entering the context."""
            if self._wrapper is not None:
                return self._wrapper.__enter__()  # type: ignore[no-any-return]
            else:
                return None

        def __exit__(
            self, exception_type: type[BaseException], exception_value: BaseException,
            traceback: types.TracebackType
        ) -> None:
            """Restore the Vec content when leaving the context."""
            if self._wrapper is not None:
                self._wrapper.__exit__(exception_type, exception_value, traceback)
                self._unrestricted_index_set.destroy()
                if self._restricted_index_set is not None:
                    self._restricted_index_set.destroy()

    return VecSubVectorWrapperBase_Class


VecSubVectorReadWrapper = VecSubVectorWrapperBase(_VecSubVectorReadWrapper)


VecSubVectorWrapper = VecSubVectorWrapperBase(_VecSubVectorWrapper)


def BlockVecSubVectorWrapperBase(_VecSubVectorWrapperClass: type) -> type:
    """Return the base class to wrap BlockVecSubVectorWrapper or BlockVecSubVectorReadWrapper."""

    class BlockVecSubVectorWrapperBase_Class:
        """Wrap a PETSc Vec object with multiple blocks."""

        def __init__(  # type: ignore[no-any-unimported]
            self, b: typing.Union[petsc4py.PETSc.Vec, None],
            dofmaps: typing.Sequence[dcpp.fem.DofMap],
            restriction: typing.Optional[typing.Sequence[mcpp.fem.DofMapRestriction]] = None,
            ghosted: bool = True
        ) -> None:
            self._b = b
            self._len = len(dofmaps)
            if b is not None:
                if restriction is None:
                    index_maps = [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps]
                    index_sets = mcpp.la.petsc.create_index_sets(
                        index_maps, [1] * len(index_maps), ghosted=ghosted,
                        ghost_block_layout=mcpp.la.petsc.GhostBlockLayout.trailing)
                    self._unrestricted_index_sets = index_sets
                    self._restricted_index_sets = None
                    self._unrestricted_to_restricted = None
                    self._unrestricted_to_restricted_bs = None
                else:
                    assert len(dofmaps) == len(restriction)
                    assert all([
                        _same_dofmap(dofmap, restriction_.dofmap)
                        for (dofmap, restriction_) in zip(dofmaps, restriction)])
                    unrestricted_index_maps = [
                        (dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps]
                    unrestricted_index_sets = mcpp.la.petsc.create_index_sets(
                        unrestricted_index_maps, [1] * len(unrestricted_index_maps),
                        ghost_block_layout=mcpp.la.petsc.GhostBlockLayout.trailing)
                    restricted_index_maps = [
                        (restriction_.index_map, restriction_.index_map_bs) for restriction_ in restriction]
                    restricted_index_sets = mcpp.la.petsc.create_index_sets(
                        restricted_index_maps, [1] * len(restricted_index_maps),
                        ghosted=ghosted, ghost_block_layout=mcpp.la.petsc.GhostBlockLayout.trailing)
                    unrestricted_to_restricted = [
                        restriction_.unrestricted_to_restricted for restriction_ in restriction]
                    unrestricted_to_restricted_bs = [
                        restriction_.index_map_bs for restriction_ in restriction]
                    self._unrestricted_index_sets = unrestricted_index_sets
                    self._restricted_index_sets = restricted_index_sets
                    self._unrestricted_to_restricted = unrestricted_to_restricted
                    self._unrestricted_to_restricted_bs = unrestricted_to_restricted_bs

        def __iter__(self) -> typing.Optional[  # type: ignore[no-any-unimported, return]
                typing.Iterator[npt.NDArray[petsc4py.PETSc.ScalarType]]]:
            """Iterate over blocks."""
            with contextlib.ExitStack() as wrapper_stack:
                for index in range(self._len):
                    if self._b is None:
                        yield None
                    else:
                        if self._restricted_index_sets is None:
                            assert self._unrestricted_to_restricted is None
                            assert self._unrestricted_to_restricted_bs is None
                            wrapper = _VecSubVectorWrapperClass(
                                self._b, self._unrestricted_index_sets[index])
                        else:
                            assert self._unrestricted_to_restricted is not None
                            assert self._unrestricted_to_restricted_bs is not None
                            wrapper = _VecSubVectorWrapperClass(
                                self._b, self._unrestricted_index_sets[index],
                                self._restricted_index_sets[index], self._unrestricted_to_restricted[index],
                                self._unrestricted_to_restricted_bs[index])
                        yield wrapper_stack.enter_context(wrapper)

        def __enter__(self) -> "BlockVecSubVectorWrapperBase_Class":
            """Return this context."""
            return self

        def __exit__(
            self, exception_type: type[BaseException], exception_value: BaseException,
            traceback: types.TracebackType
        ) -> None:
            """Clean up when leaving the context."""
            if self._b is not None:
                for index_set in self._unrestricted_index_sets:
                    index_set.destroy()
                if self._restricted_index_sets is not None:
                    for index_set in self._restricted_index_sets:
                        index_set.destroy()

    return BlockVecSubVectorWrapperBase_Class


BlockVecSubVectorReadWrapper = BlockVecSubVectorWrapperBase(_VecSubVectorReadWrapper)


BlockVecSubVectorWrapper = BlockVecSubVectorWrapperBase(_VecSubVectorWrapper)


def NestVecSubVectorWrapperBase(VecSubVectorWrapperClass: type) -> type:
    """Return the base class to wrap NestVecSubVectorWrapper or NestVecSubVectorReadWrapper."""

    class NestVecSubVectorWrapperBase_Class:
        """Wrap a PETSc Vec object with nested blocks."""

        def __init__(  # type: ignore[no-any-unimported]
            self, b: typing.Union[petsc4py.PETSc.Vec, typing.Sequence[petsc4py.PETSc.Vec], None],
            dofmaps: typing.Sequence[dcpp.fem.DofMap],
            restriction: typing.Optional[typing.Sequence[mcpp.fem.DofMapRestriction]] = None,
            ghosted: bool = True
        ) -> None:
            if b is not None:
                if isinstance(b, collections.abc.Sequence):
                    self._b = b
                    self._b_destroy = False
                else:
                    self._b = b.getNestSubVecs()
                    self._b_destroy = True
                assert len(self._b) == len(dofmaps)
            else:
                self._b_destroy = False
                self._b = [None] * len(dofmaps)
            self._dofmaps = dofmaps
            self._restriction = restriction
            self._ghosted = ghosted

        def __iter__(self) -> typing.Optional[  # type: ignore[no-any-unimported, return]
                typing.Iterator[npt.NDArray[petsc4py.PETSc.ScalarType]]]:
            """Iterate over blocks."""
            with contextlib.ExitStack() as wrapper_stack:
                for index, b_index in enumerate(self._b):
                    if b_index is None:
                        yield None
                    else:
                        if self._restriction is None:
                            if self._ghosted:
                                yield wrapper_stack.enter_context(b_index.localForm()).array_w
                            else:
                                yield b_index.array_w
                        else:
                            wrapper = VecSubVectorWrapperClass(
                                b_index, self._dofmaps[index], self._restriction[index], ghosted=self._ghosted)
                            yield wrapper_stack.enter_context(wrapper)

        def __enter__(self) -> "NestVecSubVectorWrapperBase_Class":
            """Return this context."""
            return self

        def __exit__(
            self, exception_type: type[BaseException], exception_value: BaseException,
            traceback: types.TracebackType
        ) -> None:
            """Clean up when leaving the context."""
            if self._b_destroy:
                for b_index in self._b:
                    b_index.destroy()

    return NestVecSubVectorWrapperBase_Class


NestVecSubVectorReadWrapper = NestVecSubVectorWrapperBase(VecSubVectorReadWrapper)


NestVecSubVectorWrapper = NestVecSubVectorWrapperBase(VecSubVectorWrapper)


@functools.singledispatch
def assemble_vector(  # type: ignore[no-any-unimported]
    L: DolfinxRank1FormsType, constants: DolfinxConstantsType = None, coeffs: DolfinxCoefficientsType = None,
    kind: DolfinxVectorKindType = None, restriction: MultiphenicsxRank1RestrictionsType = None
) -> petsc4py.PETSc.Vec:
    """
    Assemble linear form(s) into a new PETSc vector.

    Three cases are supported:

    1. If ``L`` is a single linear form, the form is assembled into a
       ghosted PETSc vector.

    2. If ``L`` is a sequence of linear forms and ``kind`` is ``None``
       or is ``PETSc.Vec.Type.MPI``, the forms are assembled into a
       vector ``b`` such that ``b = [b_0, b_1, ..., b_n, b_0g, b_1g,
       ..., b_ng]`` where ``b_i`` are the entries associated with the
       'owned' degrees-of-freedom for ``L[i]`` and ``b_ig`` are the
       'unowned' (ghost) entries for ``L[i]``.

    3. If ``L`` is a sequence of linear forms and ``kind`` is
       ``PETSc.Vec.Type.NEST``, the forms are assembled into a PETSc
       nested vector ``b`` (a nest of ghosted PETSc vectors) such that
       ``L[i]`` is assembled into into the ith nested matrix in ``b``.

    Constant and coefficient data that appear in the forms(s) can be
    packed outside of this function to avoid re-packing by this
    function. The functions :func:`dolfinx.fem.pack_constants` and
    :func:`dolfinx.fem.pack_coefficients` can be used to 'pre-pack' the
    data.

    Parameters
    ----------
    L
        A linear form or sequence of linear forms.
    constants
        Constants that appear in the form.
        For a single form, ``constants.ndim==1``. For multiple forms, the constants for
        form ``L[i]`` are  ``constants[i]``.
        If not provided, any required constants will be computed.
    coeffs
        Coefficients that appear in the form.
        For a single form, ``coeffs.shape=(num_cells, n)``. For multiple forms, the
        coefficients for form ``L[i]`` are  ``coeffs[i]``.
        If not provided, any required coefficients will be computed.
    kind
        PETSc vector type.
    restriction
        A dofmap restriction. If not provided, the unrestricted vector will be assembled.

    Returns
    -------
    :
        The assembled PETSc vector.

    Notes
    -----
    The returned vector is not finalised, i.e. ghost values are not accumulated on the owning processes.
    """
    b = create_vector(L, kind, restriction)
    dolfinx.fem.petsc._zero_vector(b)
    return assemble_vector(b, L, constants, coeffs, restriction)  # type: ignore[arg-type]



@assemble_vector.register
def _(  # type: ignore[no-any-unimported]
    b: petsc4py.PETSc.Vec, L: DolfinxRank1FormsType,
    constants: DolfinxConstantsType = None, coeffs: DolfinxCoefficientsType = None,
    restriction: MultiphenicsxRank1RestrictionsType = None
) -> petsc4py.PETSc.Vec:
    """
    Assemble linear form(s) into a PETSc vector.

    The vector ``b`` must have been initialized with a size/layout that
    is consistent with the linear form. The vector ``b`` is normally
    created by :func:`create_vector`.

    Constants and coefficients that appear in the forms(s) can be passed
    to avoid re-computation of constants and coefficients. The functions
    :func:`dolfinx.fem.assemble.pack_constants` and
    :func:`dolfinx.fem.assemble.pack_coefficients` can be called.

    Parameters
    ----------
    b
        PETSc vector to assemble the contribution of the linear form into.
    L
        A linear form or sequence of linear forms to assemble into ``b``.
    constants
        Constants appearing in the form. For a single form,
        ``constants.ndim==1``. For multiple forms, the constants for
        form ``L[i]`` are  ``constants[i]``.
        If not provided, any required constants will be computed.
    coeffs
        Coefficients appearing in the form. For a single form,
        ``coeffs.shape=(num_cells, n)``. For multiple forms, the
        coefficients for form ``L[i]`` are  ``coeffs[i]``.
        If not provided, any required coefficients will be computed.
    restriction
        A dofmap restriction. If not provided, the unrestricted vector will be assembled.

    Returns
    -------
    :
        The assembled PETSc vector.

    Notes
    -----
    The vector is not zeroed before assembly and it is not finalised, i.e. ghost values are not accumulated
    on the owning processes.
    """
    if b.getType() == petsc4py.PETSc.Vec.Type.NEST:  # nest vector
        assert isinstance(L, collections.abc.Sequence)
        constants = [None] * len(L) if constants is None else constants
        coeffs = [None] * len(L) if coeffs is None else coeffs
        function_spaces = _get_block_function_spaces(L)
        dofmaps = [function_space.dofmap for function_space in function_spaces]
        with NestVecSubVectorWrapper(b, dofmaps, restriction) as nest_b:
            for b_sub, L_sub, constant, coeff in zip(nest_b, L, constants, coeffs):
                dolfinx.fem.assemble.assemble_vector(b_sub, L_sub, constant, coeff)  # type: ignore[arg-type, call-arg]
    elif isinstance(L, collections.abc.Sequence):  # block vector
        constants = [
            None if form is None else dcpp.fem.pack_constants(form._cpp_object)
            for form in L] if constants is None else constants
        coeffs = [
            {} if form is None else dcpp.fem.pack_coefficients(form._cpp_object)
            for form in L] if coeffs is None else coeffs
        function_spaces = _get_block_function_spaces(L)
        dofmaps = [function_space.dofmap for function_space in function_spaces]
        assert all(
            _same_dofmap(b_dofmap, dofmap) for (b_dofmap, dofmap) in zip(b.getAttr("_dofmaps"), dofmaps))
        with BlockVecSubVectorWrapper(b, dofmaps, restriction) as block_b:
            for b_sub, L_sub, constant, coeff in zip(block_b, L, constants, coeffs):
                dcpp.fem.assemble_vector(b_sub, L_sub._cpp_object, constant, coeff)
    else:  # single form
        if restriction is None:
            with b.localForm() as b_local:
                dolfinx.fem.assemble.assemble_vector(  # type: ignore[call-arg]
                    b_local.array_w, L, constants, coeffs)  # type: ignore[arg-type]
        else:
            with VecSubVectorWrapper(b, L.function_spaces[0].dofmap, restriction) as b_sub:
                dolfinx.fem.assemble.assemble_vector(b_sub, L, constants, coeffs)  # type: ignore[arg-type, call-arg]

    return b


# -- Matrix assembly ---------------------------------------------------------


class _MatSubMatrixWrapper:
    """Wrap a PETSc Mat object."""

    def __init__(  # type: ignore[no-any-unimported]
        self, A: petsc4py.PETSc.Mat, unrestricted_index_sets: tuple[petsc4py.PETSc.IS, petsc4py.PETSc.IS],
        restricted_index_sets: typing.Optional[tuple[petsc4py.PETSc.IS, petsc4py.PETSc.IS]] = None,
        unrestricted_to_restricted: typing.Optional[tuple[dict[int, int], dict[int, int]]] = None,
        unrestricted_to_restricted_bs: typing.Optional[tuple[int, int]] = None
    ) -> None:
        if restricted_index_sets is None:
            assert unrestricted_to_restricted is None
            assert unrestricted_to_restricted_bs is None
            self._cpp_object = mcpp.la.petsc.MatSubMatrixWrapper(A, unrestricted_index_sets)
        else:
            self._cpp_object = mcpp.la.petsc.MatSubMatrixWrapper(
                A, unrestricted_index_sets,
                restricted_index_sets,
                unrestricted_to_restricted,
                unrestricted_to_restricted_bs)
        self._cpp_object_mat: typing.Optional[petsc4py.PETSc.Mat] = None  # type: ignore[no-any-unimported]

    def __enter__(self) -> petsc4py.PETSc.Mat:  # type: ignore[no-any-unimported]
        """Return submatrix content."""
        self._cpp_object_mat = self._cpp_object.mat()
        return self._cpp_object_mat

    def __exit__(
        self, exception_type: type[BaseException], exception_value: BaseException,
        traceback: types.TracebackType
    ) -> None:
        """Restore submatrix content."""
        assert self._cpp_object_mat is not None
        self._cpp_object_mat.destroy()
        self._cpp_object.restore()


class MatSubMatrixWrapper:
    """Wrap a PETSc Mat object."""

    def __init__(  # type: ignore[no-any-unimported]
        self, A: petsc4py.PETSc.Mat, dofmaps: tuple[dcpp.fem.DofMap, dcpp.fem.DofMap],
        restriction: typing.Optional[tuple[mcpp.fem.DofMapRestriction, mcpp.fem.DofMapRestriction]] = None
    ) -> None:
        assert len(dofmaps) == 2
        if restriction is None:  # pragma: no cover
            index_maps = (
                (dofmaps[0].index_map, dofmaps[0].index_map_bs),
                (dofmaps[1].index_map, dofmaps[1].index_map_bs))
            index_sets = (
                mcpp.la.petsc.create_index_sets([index_maps[0]], [dofmaps[0].index_map_bs])[0],
                mcpp.la.petsc.create_index_sets([index_maps[1]], [dofmaps[1].index_map_bs])[0])
            self._wrapper = _MatSubMatrixWrapper(A, index_sets)
            self._unrestricted_index_sets = index_sets
            self._restricted_index_sets = None
            self._unrestricted_to_restricted = None
            self._unrestricted_to_restricted_bs = None
        else:
            assert len(restriction) == 2
            assert all([_same_dofmap(dofmaps[i], restriction[i].dofmap) for i in range(2)])
            unrestricted_index_maps = (
                (dofmaps[0].index_map, dofmaps[0].index_map_bs),
                (dofmaps[1].index_map, dofmaps[1].index_map_bs))
            unrestricted_index_sets = (
                mcpp.la.petsc.create_index_sets(
                    [unrestricted_index_maps[0]], [dofmaps[0].index_map_bs])[0],
                mcpp.la.petsc.create_index_sets(
                    [unrestricted_index_maps[1]], [dofmaps[1].index_map_bs])[0])
            restricted_index_maps = (
                (restriction[0].index_map, restriction[0].index_map_bs),
                (restriction[1].index_map, restriction[1].index_map_bs))
            restricted_index_sets = (
                mcpp.la.petsc.create_index_sets(
                    [restricted_index_maps[0]], [restriction[0].index_map_bs])[0],
                mcpp.la.petsc.create_index_sets(
                    [restricted_index_maps[1]], [restriction[1].index_map_bs])[0])
            unrestricted_to_restricted = (
                restriction[0].unrestricted_to_restricted,
                restriction[1].unrestricted_to_restricted)
            unrestricted_to_restricted_bs = (
                restriction[0].index_map_bs,
                restriction[1].index_map_bs)
            self._wrapper = _MatSubMatrixWrapper(
                A, unrestricted_index_sets, restricted_index_sets, unrestricted_to_restricted,
                unrestricted_to_restricted_bs)
            self._unrestricted_index_sets = unrestricted_index_sets
            self._restricted_index_sets = restricted_index_sets
            self._unrestricted_to_restricted = unrestricted_to_restricted
            self._unrestricted_to_restricted_bs = unrestricted_to_restricted_bs

    def __enter__(self) -> petsc4py.PETSc.Mat:  # type: ignore[no-any-unimported]
        """Return submatrix content."""
        return self._wrapper.__enter__()

    def __exit__(
        self, exception_type: type[BaseException], exception_value: BaseException,
        traceback: types.TracebackType
    ) -> None:
        """Restore submatrix content."""
        self._wrapper.__exit__(exception_type, exception_value, traceback)
        self._unrestricted_index_sets[0].destroy()
        self._unrestricted_index_sets[1].destroy()
        if self._restricted_index_sets is not None:
            self._restricted_index_sets[0].destroy()
            self._restricted_index_sets[1].destroy()


class BlockMatSubMatrixWrapper:
    """Wrap a PETSc Mat object with several blocks."""

    def __init__(  # type: ignore[no-any-unimported]
        self, A: petsc4py.PETSc.Mat,
        dofmaps: tuple[typing.Sequence[dcpp.fem.DofMap], typing.Sequence[dcpp.fem.DofMap]],
        restriction: typing.Optional[
            tuple[typing.Sequence[mcpp.fem.DofMapRestriction], typing.Sequence[mcpp.fem.DofMapRestriction]]] = None
    ) -> None:
        self._A = A
        assert len(dofmaps) == 2
        if restriction is None:
            index_maps = (
                [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps[0]],
                [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps[1]])
            index_sets = (
                mcpp.la.petsc.create_index_sets(index_maps[0], [1] * len(index_maps[0])),
                mcpp.la.petsc.create_index_sets(index_maps[1], [1] * len(index_maps[1])))
            self._unrestricted_index_sets = index_sets
            self._restricted_index_sets = None
            self._unrestricted_to_restricted = None
            self._unrestricted_to_restricted_bs = None
        else:
            assert len(restriction) == 2
            for i in range(2):
                assert len(dofmaps[i]) == len(restriction[i])
                assert all(
                    [_same_dofmap(dofmap, restriction_.dofmap)
                     for (dofmap, restriction_) in zip(dofmaps[i], restriction[i])])
            unrestricted_index_maps = (
                [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps[0]],
                [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps[1]])
            unrestricted_index_sets = (
                mcpp.la.petsc.create_index_sets(
                    unrestricted_index_maps[0], [1] * len(unrestricted_index_maps[0])),
                mcpp.la.petsc.create_index_sets(
                    unrestricted_index_maps[1], [1] * len(unrestricted_index_maps[1])))
            restricted_index_maps = (
                [(restriction_.index_map, restriction_.index_map_bs) for restriction_ in restriction[0]],
                [(restriction_.index_map, restriction_.index_map_bs) for restriction_ in restriction[1]])
            restricted_index_sets = (
                mcpp.la.petsc.create_index_sets(
                    restricted_index_maps[0], [1] * len(restricted_index_maps[0])),
                mcpp.la.petsc.create_index_sets(
                    restricted_index_maps[1], [1] * len(restricted_index_maps[1])))
            unrestricted_to_restricted = (
                [restriction_.unrestricted_to_restricted for restriction_ in restriction[0]],
                [restriction_.unrestricted_to_restricted for restriction_ in restriction[1]])
            unrestricted_to_restricted_bs = (
                [restriction_.index_map_bs for restriction_ in restriction[0]],
                [restriction_.index_map_bs for restriction_ in restriction[1]])
            self._unrestricted_index_sets = unrestricted_index_sets
            self._restricted_index_sets = restricted_index_sets
            self._unrestricted_to_restricted = unrestricted_to_restricted
            self._unrestricted_to_restricted_bs = unrestricted_to_restricted_bs

    def __iter__(self) -> typing.Iterator[  # type: ignore[no-any-unimported]
            tuple[int, int, petsc4py.PETSc.Mat]]:
        """Iterate wrapper over blocks."""
        with contextlib.ExitStack() as wrapper_stack:
            for index0, _ in enumerate(self._unrestricted_index_sets[0]):
                for index1, _ in enumerate(self._unrestricted_index_sets[1]):
                    if self._restricted_index_sets is None:
                        wrapper = _MatSubMatrixWrapper(
                            self._A,
                            (self._unrestricted_index_sets[0][index0], self._unrestricted_index_sets[1][index1]))
                    else:
                        assert self._unrestricted_to_restricted is not None
                        assert self._unrestricted_to_restricted_bs is not None
                        wrapper = _MatSubMatrixWrapper(
                            self._A,
                            (self._unrestricted_index_sets[0][index0], self._unrestricted_index_sets[1][index1]),
                            (self._restricted_index_sets[0][index0], self._restricted_index_sets[1][index1]),
                            (self._unrestricted_to_restricted[0][index0], self._unrestricted_to_restricted[1][index1]),
                            (self._unrestricted_to_restricted_bs[0][index0],
                             self._unrestricted_to_restricted_bs[1][index1]))
                    yield (index0, index1, wrapper_stack.enter_context(wrapper))  # type: ignore[arg-type]

    def __enter__(self) -> "BlockMatSubMatrixWrapper":
        """Return this wrapper."""
        return self

    def __exit__(
        self, exception_type: type[BaseException], exception_value: BaseException,
        traceback: types.TracebackType
    ) -> None:
        """Clean up."""
        for i in range(2):
            for index_set in self._unrestricted_index_sets[i]:
                index_set.destroy()
        if self._restricted_index_sets is not None:
            for i in range(2):
                for index_set in self._restricted_index_sets[i]:
                    index_set.destroy()


class NestMatSubMatrixWrapper:
    """Wrap a PETSc Mat object with nested blocks."""

    def __init__(  # type: ignore[no-any-unimported]
        self, A: petsc4py.PETSc.Mat, dofmaps: tuple[typing.Sequence[dcpp.fem.DofMap], typing.Sequence[dcpp.fem.DofMap]],
        restriction: typing.Optional[
            tuple[typing.Sequence[mcpp.fem.DofMapRestriction], typing.Sequence[mcpp.fem.DofMapRestriction]]] = None
    ) -> None:
        self._A = A
        self._dofmaps = dofmaps
        self._restriction = restriction

    def __iter__(self) -> typing.Iterator[  # type: ignore[no-any-unimported]
            tuple[int, int, petsc4py.PETSc.Mat]]:
        """Iterate wrapper over blocks."""
        with contextlib.ExitStack() as wrapper_stack:
            for index0, _ in enumerate(self._dofmaps[0]):
                for index1, _ in enumerate(self._dofmaps[1]):
                    A_sub = self._A.getNestSubMatrix(index0, index1)
                    if self._restriction is None:
                        wrapper_content = A_sub
                    else:
                        wrapper = MatSubMatrixWrapper(
                            A_sub,
                            (self._dofmaps[0][index0], self._dofmaps[1][index1]),
                            (self._restriction[0][index0], self._restriction[1][index1]))
                        wrapper_content = wrapper_stack.enter_context(wrapper)  # type: ignore[arg-type]
                    yield (index0, index1, wrapper_content)
                    A_sub.destroy()

    def __enter__(self) -> "NestMatSubMatrixWrapper":
        """Return this wrapper."""
        return self

    def __exit__(
        self, exception_type: type[BaseException], exception_value: BaseException,
        traceback: types.TracebackType
    ) -> None:
        """Do nothing."""
        pass


@functools.singledispatch
def assemble_matrix(  # type: ignore[no-any-unimported]
    a: DolfinxRank2FormsType,
    bcs: typing.Optional[typing.Sequence[dolfinx.fem.DirichletBC]] = None, diag: float = 1.0,
    constants: DolfinxConstantsType = None, coeffs: DolfinxCoefficientsType = None,
    kind: DolfinxMatrixKindType = None, restriction: MultiphenicsxRank2RestrictionsType = None
) -> petsc4py.PETSc.Mat:
    r"""
    Assemble a bilinear form into a matrix.

    The following cases are supported:

    1. If ``a`` is a single bilinear form, the form is assembled
       into PETSc matrix of type ``kind``.
    #. If ``a`` is a :math:`m \times n` rectangular array of forms the
       forms in ``a`` are assembled into a matrix such that::

            A = [A_00 ... A_0n]
                [A_10 ... A_1n]
                [     ...     ]
                [A_m0 ..  A_mn]

       where ``A_ij`` is the matrix associated with the form
       ``a[i][j]``.

       a. If ``kind`` is a ``PETSc.Mat.Type`` (other than
          ``PETSc.Mat.Type.NEST``) or is ``None``, the matrix type is
          ``kind`` of the default type (if ``kind`` is ``None``).
       #. If ``kind`` is ``PETSc.Mat.Type.NEST`` or a rectangular array
          of PETSc matrix types, the returned matrix has type
          ``PETSc.Mat.Type.NEST``.

    Rows/columns that are constrained by a Dirichlet boundary condition
    are zeroed, with the diagonal to set to ``diag``.

    Constant and coefficient data that appear in the forms(s) can be
    packed outside of this function to avoid re-packing by this
    function. The functions :func:`dolfinx.fem.pack_constants` and
    :func:`dolfinx.fem.pack_coefficients` can be used to 'pre-pack' the
    data.

    Parameters
    ----------
    a
        Bilinear form(s) to assembled into a matrix.
    bcs
        Dirichlet boundary conditions applied to the system.
    diag
        Value to set on the matrix diagonal for Dirichlet
        boundary condition constrained degrees-of-freedom belonging
        to the same trial and test space.
    constants
        Constants appearing the in the form.
    coeffs
        Coefficients appearing the in the form.
    kind
        PETSc matrix type.
    restriction
        A dofmap restriction. If not provided, the unrestricted matrix will be assembled.

    Returns
    -------
    :
        The assembled PETSc matrix.

    Notes
    -----
    The returned matrix is not finalised, i.e. ghost values are not accumulated.
    """
    A = create_matrix(a, kind, restriction)
    return assemble_matrix(A, a, bcs, diag, constants, coeffs, restriction)  # type: ignore[arg-type]

@assemble_matrix.register
def _(  # type: ignore[no-any-unimported]
    A: petsc4py.PETSc.Mat, a: DolfinxRank2FormsType,
    bcs: typing.Optional[typing.Sequence[dolfinx.fem.DirichletBC]] = None, diag: float = 1.0,
    constants: DolfinxConstantsType = None, coeffs: DolfinxCoefficientsType = None,
    restriction: MultiphenicsxRank2RestrictionsType = None
) -> petsc4py.PETSc.Mat:
    """
    Assemble bilinear form into a matrix.

    The matrix vector ``A`` must have been initialized with a
    size/layout that is consistent with the bilinear form(s). The PETSc
    matrix ``A`` is normally created by :func:`create_matrix`.

    Parameters
    ----------
    A
        PETSc matrix to assemble the contribution of the bilinear forms into.
    a
        A bilinear form to assemble into `A`.
    bcs
        Optional list of boundary conditions.
    diag
        Optional diagonal value for boundary conditions application. Assumes 1 by default.
    constants
        Constants that appear in the form. If not provided, any required constants will be computed.
    coeffs
        Coefficients that appear in the form. If not provided, any required coefficients will be computed.
    restriction
        A dofmap restriction. If not provided, the unrestricted matrix will be assembled.

    Returns
    -------
    :
        The assembled PETSc matrix.

    Notes
    -----
    The returned matrix is not finalised, i.e. ghost values are not accumulated.
    """
    bcs_cpp = [bc._cpp_object for bc in bcs] if bcs is not None else []

    if A.getType() == petsc4py.PETSc.Mat.Type.NEST:  # nest matrix
        assert isinstance(a, collections.abc.Sequence)
        function_spaces = _get_block_function_spaces(a)
        dofmaps = (
            [function_space.dofmap for function_space in function_spaces[0]],
            [function_space.dofmap for function_space in function_spaces[1]])

        # Assemble form
        constants = [[  # type: ignore[misc]
            np.array([], dtype=petsc4py.PETSc.ScalarType) if form is None else dcpp.fem.pack_constants(form._cpp_object)
            for form in forms] for forms in a] if constants is None else constants
        coeffs = [[  # type: ignore[misc]
            {} if form is None else dcpp.fem.pack_coefficients(form._cpp_object)
            for form in forms] for forms in a] if coeffs is None else coeffs

        with NestMatSubMatrixWrapper(A, dofmaps, restriction) as nest_A:
            for i, j, A_sub in nest_A:
                a_sub = a[i][j]
                if a_sub is not None:
                    const_sub = constants[i][j]  # type: ignore[index]
                    coeff_sub = coeffs[i][j]  # type: ignore[index]
                    dcpp.fem.petsc.assemble_matrix(A_sub, a_sub._cpp_object, const_sub, coeff_sub, bcs_cpp)
                elif i == j:  # pragma: no cover
                    for bc in bcs_cpp:
                        if function_spaces[0][i].contains(bc.function_space):
                            raise RuntimeError(
                                f"Diagonal sub-block ({i}, {j}) cannot be 'None' and have DirichletBC applied."
                                " Consider assembling a zero block.")

        # Flush to enable switch from add to set in the matrix
        A.assemble(petsc4py.PETSc.Mat.AssemblyType.FLUSH)

        # Set diagonal value
        with NestMatSubMatrixWrapper(A, dofmaps, restriction) as nest_A:
            for i, j, A_sub in nest_A:
                if function_spaces[0][i] is function_spaces[1][j]:
                    a_sub = a[i][j]
                    if a_sub is not None:
                        dcpp.fem.petsc.insert_diagonal(A_sub, function_spaces[0][i], bcs_cpp, diag)
    elif isinstance(a, collections.abc.Sequence):  # block matrix
        constants = [[  # type: ignore[misc]
            np.array([], dtype=petsc4py.PETSc.ScalarType) if form is None else dcpp.fem.pack_constants(form._cpp_object)
            for form in forms] for forms in a] if constants is None else constants
        coeffs = [[  # type: ignore[misc]
            {} if form is None else dcpp.fem.pack_coefficients(form._cpp_object)
            for form in forms] for forms in a] if coeffs is None else coeffs
        function_spaces = _get_block_function_spaces(a)
        dofmaps = (
            [function_space.dofmap for function_space in function_spaces[0]],
            [function_space.dofmap for function_space in function_spaces[1]])

        # Assemble form
        with BlockMatSubMatrixWrapper(A, dofmaps, restriction) as block_A:
            for i, j, A_sub in block_A:
                a_sub = a[i][j]
                if a_sub is not None:
                    const_sub = constants[i][j]  # type: ignore[index]
                    coeff_sub = coeffs[i][j]  # type: ignore[index]
                    dcpp.fem.petsc.assemble_matrix(A_sub, a_sub._cpp_object, const_sub, coeff_sub, bcs_cpp, True)
                elif i == j:  # pragma: no cover
                    for bc in bcs_cpp:
                        if function_spaces[0][i].contains(bc.function_space):
                            raise RuntimeError(
                                f"Diagonal sub-block ({i}, {j}) cannot be 'None' and have DirichletBC applied."
                                " Consider assembling a zero block.")

        # Flush to enable switch from add to set in the matrix
        A.assemble(petsc4py.PETSc.Mat.AssemblyType.FLUSH)

        # Set diagonal
        with BlockMatSubMatrixWrapper(A, dofmaps, restriction) as block_A:
            for i, j, A_sub in block_A:
                if function_spaces[0][i] is function_spaces[1][j]:
                    a_sub = a[i][j]
                    if a_sub is not None:
                        dcpp.fem.petsc.insert_diagonal(A_sub, function_spaces[0][i], bcs_cpp, diag)

    else:  # single form
        constants = dcpp.fem.pack_constants(a._cpp_object) if constants is None else constants
        coeffs = dcpp.fem.pack_coefficients(a._cpp_object) if coeffs is None else coeffs
        function_spaces = a.function_spaces
        if restriction is None:
            # Assemble form
            dcpp.fem.petsc.assemble_matrix(A, a._cpp_object, constants, coeffs, bcs_cpp)

            if function_spaces[0] is function_spaces[1]:
                # Flush to enable switch from add to set in the matrix
                A.assemble(petsc4py.PETSc.Mat.AssemblyType.FLUSH)

                # Set diagonal value
                dcpp.fem.petsc.insert_diagonal(A, function_spaces[0], bcs_cpp, diag)
        else:
            dofmaps = (function_spaces[0].dofmap, function_spaces[1].dofmap)  # type: ignore[assignment]

            # Assemble form
            with MatSubMatrixWrapper(A, dofmaps, restriction) as A_sub:
                dcpp.fem.petsc.assemble_matrix(A_sub, a._cpp_object, constants, coeffs, bcs_cpp)

            if function_spaces[0] is function_spaces[1]:
                # Flush to enable switch from add to set in the matrix
                A.assemble(petsc4py.PETSc.Mat.AssemblyType.FLUSH)

                # Set diagonal value
                with MatSubMatrixWrapper(A, dofmaps, restriction) as A_sub:
                    dcpp.fem.petsc.insert_diagonal(A_sub, function_spaces[0], bcs_cpp, diag)

    return A


# -- Modifiers for Dirichlet conditions ---------------------------------------

def apply_lifting(  # type: ignore[no-any-unimported]
    b: petsc4py.PETSc.Vec,
    a: typing.Union[typing.Sequence[dolfinx.fem.Form], typing.Sequence[typing.Sequence[dolfinx.fem.Form]]],
    bcs: typing.Optional[
        typing.Union[typing.Sequence[dolfinx.fem.DirichletBC],
        typing.Sequence[typing.Sequence[dolfinx.fem.DirichletBC]]]] = None,
    x0: typing.Optional[typing.Sequence[petsc4py.PETSc.Vec]] = None,
    alpha: float = 1.0,
    constants: DolfinxConstantsType = None, coeffs: DolfinxCoefficientsType = None,
    restriction: MultiphenicsxRank1RestrictionsType = None,
    restriction_x0: MultiphenicsxRank1RestrictionsType = None

) -> None:
    r"""
    Apply the function :func:`dolfinx.fem.apply_lifting` to a PETSc vector.

    Parameters
    ----------
    b
        PETSc vector, typically obtained by assembling a linear form with `assemble_vector`.
    a
        A list of bilinear forms.
        If ``b`` is not blocked or a nest,
        then ``a`` is a 1D sequence. If ``b`` is blocked or a nest,
        then ``a`` is  a 2D array of forms, with the ``a[i]`` forms
        used to modify the block/nest vector ``b[i]``.
    bcs
        Boundary conditions used to modify ``b`` (see
        :func:`dolfinx.fem.apply_lifting`). Two cases are supported:

        1. The boundary conditions ``bcs`` are a
           'sequence-of-sequences' such that ``bcs[j]`` are the
           Dirichlet boundary conditions associated with the forms in
           the ``j`` th colulmn of ``a``. Helper functions exist to
           create a sequence-of-sequences of `DirichletBC` from the 2D
           ``a`` and a flat Sequence of `DirichletBC` objects ``bcs``::

               bcs1 = fem.bcs_by_block(
                fem.extract_function_spaces(a, 1), bcs
               )

        2. ``bcs`` is a sequence of :class:`dolfinx.fem.DirichletBC`
           objects. The function deduces which `DiricletBC` objects
           apply to each column of ``a`` by matching the
           :class:`dolfinx.fem.FunctionSpace`.
    x0
        PETSc vector storing the solution to be subtracted to the Dirichlet values.
        Typically the current nonlinear solution in an incremental problem is provided as `x0`.

        - If `b` was obtained by calling `assemble_vector` without `restriction`, then
          `x0` represents the PETSc vector associated to the (unrestricted) solution.
          The `restriction_x0` must not be provided in this case.
        - If `b` was obtained by calling `assemble_vector` with `restriction`, then:

          - if `x0` is the PETSc vector associated to an unrestricted solution, then the argument
            `restriction_x0` must not be provided (and thus get the default argument `None`).
          - if `x0` is the PETSc vector associated to a restricted solution, then the argument
            `restriction_x0` must be provided, and it must contain the `DofMapRestriction` that
            was used to create the restricted solution `x0` out of its unrestricted counterpart.
    alpha
        Scaling factor.
    constants
        Constants that appear in the forms. If not provided, any required constants will be computed.
    coeffs
        Coefficients that appear in the forms. If not provided, any required coefficients will be computed.
    restriction, restriction_x0
        Dofmap restrictions for `b` and `x0`. If not provided, the input vectors will be used as they are.
    """
    if not isinstance(a[0], collections.abc.Sequence):  # single form
        function_spaces = [form.function_spaces[1] for form in a]  # type: ignore[union-attr]
        dofmaps_x0 = [function_space.dofmap for function_space in function_spaces]
        with NestVecSubVectorReadWrapper(x0, dofmaps_x0, restriction_x0) as nest_x0:
            if x0 is not None:
                x0_as_list = [x0_sub.copy() for x0_sub in nest_x0]
            else:
                x0_as_list = []
            if restriction is None:
                with b.localForm() as b_local:
                    dolfinx.fem.assemble.apply_lifting(
                        b_local.array_w, a, bcs, x0_as_list, alpha, constants, coeffs)  # type: ignore[arg-type]
            else:
                assert not isinstance(restriction, collections.abc.Sequence)
                with VecSubVectorWrapper(b, restriction.dofmap, restriction) as b_sub:
                    dolfinx.fem.assemble.apply_lifting(
                        b_sub, a, bcs, x0_as_list, alpha, constants, coeffs)  # type: ignore[arg-type]
    else:  # block or nest vector
        constants = [[  # type: ignore[misc]
            np.array([], dtype=petsc4py.PETSc.ScalarType) if form is None
            else dcpp.fem.pack_constants(form._cpp_object)
            for form in forms] for forms in a] if constants is None else constants  # type: ignore[union-attr]
        coeffs = [[  # type: ignore[misc]
            {} if form is None else dcpp.fem.pack_coefficients(form._cpp_object)
            for form in forms] for forms in a] if coeffs is None else coeffs  # type: ignore[union-attr]

        function_spaces = _get_block_function_spaces(a)
        dofmaps = [function_space.dofmap for function_space in function_spaces[0]]
        dofmaps_x0 = [function_space.dofmap for function_space in function_spaces[1]]

        bcs1 = dolfinx.fem.bcs_by_block(function_spaces[1], bcs)  # type: ignore[arg-type]
        if b.getType() == petsc4py.PETSc.Vec.Type.NEST:  # nest vector
            with NestVecSubVectorWrapper(b, dofmaps, restriction) as nest_b, \
                    NestVecSubVectorReadWrapper(x0, dofmaps_x0, restriction_x0) as nest_x0:
                if x0 is not None:
                    nest_x0_as_list = [x0_sub.copy() for x0_sub in nest_x0]
                else:
                    nest_x0_as_list = []
                for b_sub, a_sub, constants_a, coeffs_a in zip(nest_b, a, constants, coeffs):
                    dolfinx.fem.assemble.apply_lifting(
                        b_sub, a_sub, bcs1, nest_x0_as_list, alpha, constants_a, coeffs_a)  # type: ignore[arg-type]
        else:  # block vector
            assert all(
                _same_dofmap(b_dofmap, dofmap) for (b_dofmap, dofmap) in zip(b.getAttr("_dofmaps"), dofmaps))
            with BlockVecSubVectorWrapper(b, dofmaps, restriction) as block_b, \
                    BlockVecSubVectorReadWrapper(x0, dofmaps_x0, restriction_x0) as block_x0:
                if x0 is not None:
                    block_x0_as_list = [x0_sub.copy() for x0_sub in block_x0]
                else:
                    block_x0_as_list = []
                for b_sub, a_sub, constant_a, coeff_a in zip(block_b, a, constants, coeffs):
                    dolfinx.fem.assemble.apply_lifting(
                        b_sub, a_sub, bcs1, block_x0_as_list, alpha, constant_a, coeff_a)  # type: ignore[arg-type]



def set_bc(  # type: ignore[no-any-unimported]
    b: petsc4py.PETSc.Vec,
    bcs: typing.Union[
        typing.Sequence[dolfinx.fem.DirichletBC],
        typing.Sequence[typing.Sequence[dolfinx.fem.DirichletBC]]],
    x0: typing.Optional[petsc4py.PETSc.Vec] = None,
    alpha: float = 1.0,
    restriction: MultiphenicsxRank1RestrictionsType = None,
    restriction_x0: MultiphenicsxRank1RestrictionsType = None
) -> None:
    r"""
    Apply boundary conditions to a PETSc vector.

    Parameters
    ----------
    b
        PETSc vector, typically obtained by assembling a linear form with `assemble_vector`.
    bcs
        Boundary conditions to apply. If ``b`` is nested or
        blocked, ``bcs`` is a 2D array and ``bcs[i]`` are the
        boundary conditions to apply to block/nest ``i``. Otherwise
        ``bcs`` should be a sequence of ``DirichletBC``\s. For
        block/nest problems, :func:`dolfinx.fem.bcs_by_block` can be
        used to prepare the 2D array of ``DirichletBC`` objects.
    x0
        PETSc vector storing the solution to be subtracted to the Dirichlet values.
        Typically the current nonlinear solution in an incremental problem is provided as `x0`.
        See the documentation of :func:`multiphenicsx.fem.petsc.apply_lifting` for more details about
        how `restriction_x0` is used in combination with `x0`.
    alpha
        Scaling factor.
    restriction, restriction_x0
        Dofmap restrictions for `b` and `x0`. If not provided, the input vectors will be used as they are.
    """
    if len(bcs) == 0:
        return

    if not isinstance(bcs[0], collections.abc.Sequence):  # single form
        if restriction is None:
            if x0 is not None:
                x0 = x0.array_r
            for bc in bcs:
                bc.set(b.array_w, x0, alpha)  # type: ignore[union-attr]
        else:
            if restriction_x0 is None:
                dofmap_x0 = bcs[0].function_space.dofmap
                # cannot uncomment the following assert because DirichletBC.function_space returns
                # at every call a new C++ wrapped object, and hence a new dofmap.
                # assert all(_same_dofmap(bc.function_space.dofmap, dofmap_x0) for bc in bcs[1:])
            else:
                dofmap_x0 = restriction_x0.dofmap  # type: ignore[union-attr]
            restriction_dofmap = restriction.dofmap  # type: ignore[union-attr]
            with VecSubVectorWrapper(b, restriction_dofmap, restriction, ghosted=False) as b_sub, \
                    VecSubVectorReadWrapper(x0, dofmap_x0, restriction_x0, ghosted=False) as x0_sub:
                for bc in bcs:
                    bc.set(b_sub, x0_sub, alpha)  # type: ignore[union-attr]
    elif b.getType() == petsc4py.PETSc.Vec.Type.NEST:  # nest vector
        if restriction is None:
            dofmaps = [None] * len(b.getNestSubVecs())
        else:
            dofmaps = [restriction_.dofmap for restriction_ in restriction]
            assert len(b.getNestSubVecs()) == len(dofmaps)
        assert len(dofmaps) == len(bcs)
        if restriction_x0 is None:
            dofmaps_x0 = [None] * len(dofmaps)
        else:
            dofmaps_x0 = [restriction_.dofmap for restriction_ in restriction_x0]
            assert len(restriction_x0) == len(dofmaps)
        if x0 is not None:
            assert len(dofmaps_x0) == len(x0.getNestSubVecs())
        with NestVecSubVectorWrapper(b, dofmaps, restriction, ghosted=False) as nest_b, \
                NestVecSubVectorReadWrapper(x0, dofmaps_x0, restriction_x0, ghosted=False) as nest_x0:
            for b_sub, bcs_sub, x0_sub in zip(nest_b, bcs, nest_x0):
                for bc in bcs_sub:  # type: ignore[attr-defined]
                    bc.set(b_sub, x0_sub, alpha)
    else:  # block vector
        if restriction is None:
            # cannot deduce dofmaps from input arguments of this function, need to get them
            # from the attribute that was attached when creating the block vector b
            # upstream dolfinx imposes a similar requirement using the attribute _blocks
            dofmaps = b.getAttr("_dofmaps")
            assert dofmaps is not None
        else:
            dofmaps = [restriction_.dofmap for restriction_ in restriction]
        assert len(dofmaps) == len(bcs)
        if restriction_x0 is None:
            if x0 is None:
                dofmaps_x0 = [None] * len(dofmaps)
            else:
                # cannot deduce dofmaps from input arguments of this function, need to get them
                # from the attribute that was attached when creating the block vector x0
                # upstream dolfinx imposes a similar requirement using the attribute _blocks
                dofmaps_x0 = x0.getAttr("_dofmaps")
                assert dofmaps_x0 is not None
        else:
            dofmaps_x0 = [restriction_.dofmap for restriction_ in restriction_x0]
            assert len(restriction_x0) == len(dofmaps)
        with BlockVecSubVectorWrapper(b, dofmaps, restriction) as block_b, \
                BlockVecSubVectorReadWrapper(x0, dofmaps_x0, restriction_x0) as block_x0:
            for b_sub, bcs_sub, x0_sub in zip(block_b, bcs, block_x0):
                for bc_sub in bcs_sub:  # type: ignore[attr-defined]
                    bc_sub.set(b_sub, x0_sub, alpha)

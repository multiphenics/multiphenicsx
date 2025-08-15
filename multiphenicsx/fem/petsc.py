# Copyright (C) 2016-2025 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""High-level solver classes and functions for assembling PETSc objects."""

import collections.abc
import contextlib
import functools
import types
import typing

import dolfinx.cpp as dcpp
import dolfinx.fem
import dolfinx.fem.assemble
import dolfinx.fem.forms
import dolfinx.fem.petsc
import dolfinx.la
import dolfinx.la.petsc
import numpy as np
import numpy.typing as npt
import petsc4py.PETSc
import ufl

from multiphenicsx.cpp import cpp_library as mcpp

UflRank1FormsType = typing.Union[  # type: ignore[no-any-unimported]
    ufl.Form, typing.Sequence[ufl.Form]
]
UflRank2FormsType = typing.Union[  # type: ignore[no-any-unimported]
    ufl.Form, typing.Sequence[typing.Sequence[ufl.Form]]
]
DolfinxRank1FormsType = typing.Union[
    dolfinx.fem.Form, typing.Sequence[dolfinx.fem.Form]
]
DolfinxRank2FormsType = typing.Union[
    dolfinx.fem.Form, typing.Sequence[typing.Sequence[dolfinx.fem.Form]]
]
DolfinxVectorKindType = typing.Optional[str]
DolfinxMatrixKindType = typing.Optional[typing.Union[str, typing.Sequence[typing.Sequence[str]]]]
DolfinxConstantsType_Base = npt.NDArray[petsc4py.PETSc.ScalarType]  # type: ignore[name-defined]
DolfinxConstantsType = typing.Optional[
    typing.Union[DolfinxConstantsType_Base, typing.Sequence[typing.Optional[DolfinxConstantsType_Base]]]
]
DolfinxCoefficientsType_Base = dict[  # type: ignore[no-any-unimported]
    tuple[dolfinx.fem.IntegralType, int],
    npt.NDArray[petsc4py.PETSc.ScalarType]  # type: ignore[name-defined]
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

def create_vector(
    L: DolfinxRank1FormsType, kind: DolfinxVectorKindType = None,
    restriction: MultiphenicsxRank1RestrictionsType = None
) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
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
        function_spaces: list[dolfinx.fem.FunctionSpace] = dolfinx.fem.extract_function_spaces(L)  # type: ignore
        dofmaps = [function_space.dofmap for function_space in function_spaces]
        if restriction is None:
            index_maps = [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps]
        else:
            assert isinstance(restriction, collections.abc.Sequence)
            assert len(restriction) == len(dofmaps)
            assert all(
                _same_dofmap(restriction_.dofmap, dofmap) for (restriction_, dofmap) in zip(restriction, dofmaps))
            index_maps = [(restriction_.index_map, restriction_.index_map_bs) for restriction_ in restriction]
        if kind == petsc4py.PETSc.Vec.Type.NEST:  # type: ignore[attr-defined]
            return dcpp.fem.petsc.create_vector_nest(index_maps)
        elif kind == petsc4py.PETSc.Vec.Type.MPI:  # type: ignore[attr-defined]
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
        assert kind is None or kind == petsc4py.PETSc.Vec.Type.MPI  # type: ignore[attr-defined]
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

def create_matrix(
    a: DolfinxRank2FormsType, kind: DolfinxMatrixKindType = None,
    restriction: MultiphenicsxRank2RestrictionsType = None
) -> petsc4py.PETSc.Mat:  # type: ignore[name-defined]
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
        function_spaces: tuple[list[dolfinx.fem.FunctionSpace], list[dolfinx.fem.FunctionSpace]] = (  # type: ignore
            dolfinx.fem.extract_function_spaces(a, 0), dolfinx.fem.extract_function_spaces(a, 1))
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
                [function_spaces[0][i].dofmap.map() for i in range(rows)],  # type: ignore[attr-defined]
                [function_spaces[1][j].dofmap.map() for j in range(cols)])  # type: ignore[attr-defined]
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
        if kind == petsc4py.PETSc.Mat.Type.NEST:  # type: ignore[attr-defined]
            # Create nest matrix with default types
            return mcpp.fem.petsc.create_matrix_nest(
                a_cpp, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds, None)
        else:
            if kind is None or isinstance(kind, str):
                # Create block matrix
                return mcpp.fem.petsc.create_matrix_block(
                    a_cpp, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds, kind)
            else:
                # Create nest matrix with provided types
                return mcpp.fem.petsc.create_matrix_nest(
                    a_cpp, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds, kind)
    else:
        assert a.rank == 2
        function_spaces: tuple[dolfinx.fem.Function, dolfinx.fem.FunctionSpace] = (  # type: ignore[no-redef]
            a.function_spaces)
        assert all(function_space.mesh == a.mesh for function_space in function_spaces)  # type: ignore[attr-defined]
        if restriction is None:
            index_maps = [  # type: ignore[assignment]
                function_space.dofmap.index_map for function_space in function_spaces]  # type: ignore[attr-defined]
            index_maps_bs = [  # type: ignore[assignment]
                function_space.dofmap.index_map_bs for function_space in function_spaces]  # type: ignore[attr-defined]
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

        def __init__(
            self, b: petsc4py.PETSc.Vec, unrestricted_index_set: petsc4py.PETSc.IS,  # type: ignore[name-defined]
            restricted_index_set: typing.Optional[petsc4py.PETSc.IS] = None,  # type: ignore[name-defined]
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

        def __enter__(self) -> npt.NDArray[petsc4py.PETSc.ScalarType]:  # type: ignore[name-defined]
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
            self, b: typing.Union[petsc4py.PETSc.Vec, None], dofmap: dcpp.fem.DofMap,  # type: ignore[name-defined]
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

        def __enter__(self) -> typing.Optional[  # type: ignore[name-defined]
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
            self, b: typing.Union[petsc4py.PETSc.Vec, None],  # type: ignore[name-defined]
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

        def __iter__(self) -> typing.Optional[  # type: ignore[name-defined, return]
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
            self,
            b: typing.Union[  # type: ignore[name-defined]
                petsc4py.PETSc.Vec, typing.Sequence[petsc4py.PETSc.Vec], None
            ],
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

        def __iter__(self) -> typing.Optional[  # type: ignore[name-defined, return]
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
def assemble_vector(
    L: DolfinxRank1FormsType, constants: DolfinxConstantsType = None, coeffs: DolfinxCoefficientsType = None,
    kind: DolfinxVectorKindType = None, restriction: MultiphenicsxRank1RestrictionsType = None
) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
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
    dolfinx.la.petsc._zero_vector(b)
    return assemble_vector(b, L, constants, coeffs, restriction)  # type: ignore[arg-type]



@assemble_vector.register
def _(
    b: petsc4py.PETSc.Vec, L: DolfinxRank1FormsType,  # type: ignore[name-defined]
    constants: DolfinxConstantsType = None, coeffs: DolfinxCoefficientsType = None,
    restriction: MultiphenicsxRank1RestrictionsType = None
) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
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
    if b.getType() == petsc4py.PETSc.Vec.Type.NEST:  # type: ignore[attr-defined]
        assert isinstance(L, collections.abc.Sequence)
        constants = [None] * len(L) if constants is None else constants
        coeffs = [None] * len(L) if coeffs is None else coeffs
        function_spaces: list[dolfinx.fem.FunctionSpace] = dolfinx.fem.extract_function_spaces(L)  # type: ignore
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
        function_spaces: list[dolfinx.fem.FunctionSpace] = dolfinx.fem.extract_function_spaces(L)  # type: ignore
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

    def __init__(
        self, A: petsc4py.PETSc.Mat,   # type: ignore[name-defined]
        unrestricted_index_sets: tuple[petsc4py.PETSc.IS, petsc4py.PETSc.IS],  # type: ignore[name-defined]
        restricted_index_sets: typing.Optional[  # type: ignore[name-defined]
            tuple[petsc4py.PETSc.IS, petsc4py.PETSc.IS]
        ] = None,
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
        self._cpp_object_mat: typing.Optional[petsc4py.PETSc.Mat] = None  # type: ignore[name-defined]

    def __enter__(self) -> petsc4py.PETSc.Mat:  # type: ignore[name-defined]
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
        self, A: petsc4py.PETSc.Mat, dofmaps: tuple[dcpp.fem.DofMap, dcpp.fem.DofMap],  # type: ignore[name-defined]
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

    def __enter__(self) -> petsc4py.PETSc.Mat:  # type: ignore[name-defined]
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
        self, A: petsc4py.PETSc.Mat,  # type: ignore[name-defined]
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

    def __iter__(self) -> typing.Iterator[  # type: ignore[name-defined]
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
        self, A: petsc4py.PETSc.Mat,   # type: ignore[name-defined]
        dofmaps: tuple[typing.Sequence[dcpp.fem.DofMap], typing.Sequence[dcpp.fem.DofMap]],
        restriction: typing.Optional[
            tuple[typing.Sequence[mcpp.fem.DofMapRestriction], typing.Sequence[mcpp.fem.DofMapRestriction]]] = None
    ) -> None:
        self._A = A
        self._dofmaps = dofmaps
        self._restriction = restriction

    def __iter__(self) -> typing.Iterator[  # type: ignore[name-defined]
            tuple[int, int, petsc4py.PETSc.Mat]]:
        """Iterate wrapper over blocks."""
        with contextlib.ExitStack() as wrapper_stack:
            for index0, _ in enumerate(self._dofmaps[0]):
                for index1, _ in enumerate(self._dofmaps[1]):
                    A_sub = self._A.getNestSubMatrix(index0, index1)
                    if A_sub.handle == 0:
                        # The submatrix corresponds to a None block. Do not try to wrap it,
                        # simply use A_sub as it is, since the submatrix will never be used
                        # in practice
                        wrapper_content = A_sub
                    else:
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
def assemble_matrix(
    a: DolfinxRank2FormsType,
    bcs: typing.Optional[typing.Sequence[dolfinx.fem.DirichletBC]] = None, diag: float = 1.0,
    constants: DolfinxConstantsType = None, coeffs: DolfinxCoefficientsType = None,
    kind: DolfinxMatrixKindType = None, restriction: MultiphenicsxRank2RestrictionsType = None
) -> petsc4py.PETSc.Mat:  # type: ignore[name-defined]
    r"""
    Assemble a bilinear form into a matrix.

    The following cases are supported:

    1. If ``a`` is a single bilinear form, the form is assembled
       into PETSc matrix of type ``kind``.
    #. If ``a`` is a :math:`m \\times n` rectangular array of forms the
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
def _(
    A: petsc4py.PETSc.Mat, a: DolfinxRank2FormsType,  # type: ignore[name-defined]
    bcs: typing.Optional[typing.Sequence[dolfinx.fem.DirichletBC]] = None, diag: float = 1.0,
    constants: DolfinxConstantsType = None, coeffs: DolfinxCoefficientsType = None,
    restriction: MultiphenicsxRank2RestrictionsType = None
) -> petsc4py.PETSc.Mat:  # type: ignore[name-defined]
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

    if A.getType() == petsc4py.PETSc.Mat.Type.NEST:  # type: ignore[attr-defined]
        assert isinstance(a, collections.abc.Sequence)
        function_spaces: tuple[list[dolfinx.fem.FunctionSpace], list[dolfinx.fem.FunctionSpace]] = (  # type: ignore
            dolfinx.fem.extract_function_spaces(a, 0), dolfinx.fem.extract_function_spaces(a, 1))
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
        A.assemble(petsc4py.PETSc.Mat.AssemblyType.FLUSH)  # type: ignore[attr-defined]

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
        function_spaces: tuple[list[dolfinx.fem.FunctionSpace], list[dolfinx.fem.FunctionSpace]] = (  # type: ignore
            dolfinx.fem.extract_function_spaces(a, 0), dolfinx.fem.extract_function_spaces(a, 1))
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
        A.assemble(petsc4py.PETSc.Mat.AssemblyType.FLUSH)  # type: ignore[attr-defined]

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
        function_spaces: tuple[dolfinx.fem.Function, dolfinx.fem.FunctionSpace] = (  # type: ignore[no-redef]
            a.function_spaces)
        if restriction is None:
            # Assemble form
            dcpp.fem.petsc.assemble_matrix(A, a._cpp_object, constants, coeffs, bcs_cpp)

            if function_spaces[0] is function_spaces[1]:
                # Flush to enable switch from add to set in the matrix
                A.assemble(petsc4py.PETSc.Mat.AssemblyType.FLUSH)  # type: ignore[attr-defined]

                # Set diagonal value
                dcpp.fem.petsc.insert_diagonal(A, function_spaces[0], bcs_cpp, diag)
        else:
            dofmaps = (function_spaces[0].dofmap, function_spaces[1].dofmap)  # type: ignore[attr-defined]

            # Assemble form
            with MatSubMatrixWrapper(A, dofmaps, restriction) as A_sub:
                dcpp.fem.petsc.assemble_matrix(A_sub, a._cpp_object, constants, coeffs, bcs_cpp)

            if function_spaces[0] is function_spaces[1]:
                # Flush to enable switch from add to set in the matrix
                A.assemble(petsc4py.PETSc.Mat.AssemblyType.FLUSH)  # type: ignore[attr-defined]

                # Set diagonal value
                with MatSubMatrixWrapper(A, dofmaps, restriction) as A_sub:
                    dcpp.fem.petsc.insert_diagonal(A_sub, function_spaces[0], bcs_cpp, diag)

    return A


# -- Modifiers for Dirichlet conditions ---------------------------------------

def apply_lifting(
    b: petsc4py.PETSc.Vec,  # type: ignore[name-defined]
    a: typing.Union[typing.Sequence[dolfinx.fem.Form], typing.Sequence[typing.Sequence[dolfinx.fem.Form]]],
    bcs: typing.Optional[
        typing.Union[typing.Sequence[dolfinx.fem.DirichletBC],
        typing.Sequence[typing.Sequence[dolfinx.fem.DirichletBC]]]] = None,
    x0: typing.Optional[typing.Sequence[petsc4py.PETSc.Vec]] = None,  # type: ignore[name-defined]
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
        Boundary conditions to apply. If ``b`` is nested or
        blocked, ``bcs`` is a 2D array and ``bcs[i]`` are the
        boundary conditions to apply to block/nest ``i``. Otherwise
        ``bcs`` should be a sequence of ``DirichletBC``\s. For
        block/nest problems, :func:`dolfinx.fem.bcs_by_block` can be
        used to prepare the 2D array of ``DirichletBC`` objects.
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
            np.array([], dtype=petsc4py.PETSc.ScalarType) if form is None  # type: ignore[attr-defined]
            else dcpp.fem.pack_constants(form._cpp_object)
            for form in forms] for forms in a] if constants is None else constants  # type: ignore[union-attr]
        coeffs = [[  # type: ignore[misc]
            {} if form is None else dcpp.fem.pack_coefficients(form._cpp_object)
            for form in forms] for forms in a] if coeffs is None else coeffs  # type: ignore[union-attr]

        function_spaces: tuple[list[dolfinx.fem.FunctionSpace], list[dolfinx.fem.FunctionSpace]] = (  # type: ignore
            dolfinx.fem.extract_function_spaces(a, 0), dolfinx.fem.extract_function_spaces(a, 1))
        dofmaps = [function_space.dofmap for function_space in function_spaces[0]]
        dofmaps_x0 = [function_space.dofmap for function_space in function_spaces[1]]

        if b.getType() == petsc4py.PETSc.Vec.Type.NEST:  # type: ignore[attr-defined]
            with NestVecSubVectorWrapper(b, dofmaps, restriction) as nest_b, \
                    NestVecSubVectorReadWrapper(x0, dofmaps_x0, restriction_x0) as nest_x0:
                if x0 is not None:
                    nest_x0_as_list = [x0_sub.copy() for x0_sub in nest_x0]
                else:
                    nest_x0_as_list = []
                for b_sub, a_sub, constants_a, coeffs_a in zip(nest_b, a, constants, coeffs):
                    dolfinx.fem.assemble.apply_lifting(
                        b_sub, a_sub, bcs, nest_x0_as_list, alpha, constants_a, coeffs_a)  # type: ignore[arg-type]
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
                        b_sub, a_sub, bcs, block_x0_as_list, alpha, constant_a, coeff_a)  # type: ignore[arg-type]



def set_bc(
    b: petsc4py.PETSc.Vec,  # type: ignore[name-defined]
    bcs: typing.Union[
        typing.Sequence[dolfinx.fem.DirichletBC],
        typing.Sequence[typing.Sequence[dolfinx.fem.DirichletBC]]],
    x0: typing.Optional[petsc4py.PETSc.Vec] = None,  # type: ignore[name-defined]
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
    elif b.getType() == petsc4py.PETSc.Vec.Type.NEST:  # type: ignore[attr-defined]
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

# -- assign free function ---------------------------------------


@functools.singledispatch
def assign(
    u: typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]],
    x: petsc4py.PETSc.Vec, restriction: MultiphenicsxRank1RestrictionsType = None  # type: ignore[name-defined]
) -> None:
    """
    Assign :class:`Function` degrees-of-freedom to a vector.

    Assigns degree-of-freedom values in ``u``, which is possibly a sequence of ``Function``s, to ``x``.
    When ``u`` is a sequence of ``Function``s, degrees-of-freedom for the ``Function``s in ``u`` are
    'stacked' and assigned to ``x``.

    Parameters
    ----------
    u
        ``Function`` (s) to assign degree-of-freedom value from.
    x
        Vector to assign degree-of-freedom values in ``u`` to.
    restriction
        The dofmap restriction used when creating the vector ``x``.
        If not provided, ``x`` is assumed to be unrestricted.
    """
    if isinstance(u, collections.abc.Sequence):  # block or nest vector
        if x.getType() == petsc4py.PETSc.Vec.Type().NEST:  # type: ignore[attr-defined]
            BlockNestVecSubVectorWrapper = NestVecSubVectorWrapper
        else:  # block vector
            BlockNestVecSubVectorWrapper = BlockVecSubVectorWrapper
        with BlockNestVecSubVectorWrapper(x, [ui.function_space.dofmap for ui in u], restriction) as x_wrapper:
            for x_wrapper_local, sub_solution in zip(x_wrapper, u):
                with sub_solution.x.petsc_vec.localForm() as sub_solution_local:
                    x_wrapper_local[:] = sub_solution_local
    else:
        assert isinstance(u, dolfinx.fem.Function)
        with VecSubVectorWrapper(x, u.function_space.dofmap, restriction) as x_wrapper_local:
            with u.x.petsc_vec.localForm() as sub_solution_local:
                x_wrapper_local[:] = sub_solution_local


@assign.register
def _(
    x: petsc4py.PETSc.Vec,  # type: ignore[name-defined]
    u: typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]],
    restriction: MultiphenicsxRank1RestrictionsType = None
) -> None:
    """
    Assign vector entries to :class:`Function` degrees-of-freedom.

    Assigns values in ``x`` to the degrees-of-freedom of ``u``, which is possibly a Sequence of ``Function``s.
    When ``u`` is a Sequence of ``Function``s, values in ``x`` are assigned block-wise to the ``Function``s.

    Parameters
    ----------
    x
        Vector with values to assign values from.
    u
        ``Function`` (s) to assign degree-of-freedom values to.
    restriction
        The dofmap restriction used when creating the vector ``x``.
        If not provided, ``x`` is assumed to be unrestricted.
    """
    if isinstance(u, collections.abc.Sequence):  # block or nest vector
        if x.getType() == petsc4py.PETSc.Vec.Type().NEST:  # type: ignore[attr-defined]
            BlockNestVecSubVectorWrapper = NestVecSubVectorWrapper
        else:  # block vector
            BlockNestVecSubVectorWrapper = BlockVecSubVectorWrapper
        with BlockNestVecSubVectorWrapper(x, [ui.function_space.dofmap for ui in u], restriction) as x_wrapper:
            for x_wrapper_local, sub_solution in zip(x_wrapper, u):
                with sub_solution.x.petsc_vec.localForm() as sub_solution_local:
                    sub_solution_local[:] = x_wrapper_local
    else:
        assert isinstance(u, dolfinx.fem.Function)
        with VecSubVectorWrapper(x, u.function_space.dofmap, restriction) as x_wrapper_local:
            with u.x.petsc_vec.localForm() as sub_solution_local:
                sub_solution_local[:] = x_wrapper_local


# -- High-level interface for KSP ---------------------------------------


class LinearProblem:
    r"""
    High-level class for solving a linear variational problem using a PETSc KSP.

    Solves problems of the form :math:`a_{ij}(u, v) = f_i(v), i,j=0,\\ldots,N\\ \\forall v \\in V` where
    :math:`u=(u_0,\\ldots,u_N), v=(v_0,\\ldots,v_N)` using PETSc KSP as the linear solver.


    Notes
    -----
    This high-level class automatically handles PETSc memory management.
    The user does not need to manually call ``.destroy()`` on returned PETSc objects.
    """

    def __init__(
        self, a: UflRank2FormsType, L: UflRank1FormsType, *, petsc_options_prefix: str,
        bcs: typing.Optional[typing.Sequence[dolfinx.fem.DirichletBC]] = None,
        u: typing.Optional[typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]]] = None,
        P: typing.Optional[UflRank2FormsType] = None, kind: DolfinxMatrixKindType = None,
        petsc_options: typing.Optional[dict[str, typing.Any]] = None,
        form_compiler_options: typing.Optional[dict[str, typing.Any]] = None,
        jit_options: typing.Optional[dict[str, typing.Any]] = None,
        restriction: MultiphenicsxRank1RestrictionsType = None
    ) -> None:
        """
        Initialize solver for a linear variational problem.

        By default, the underlying KSP solver uses PETSc's default options, usually GMRES + ILU preconditioning.
        To use the robust combination of LU via MUMPS

            problem = LinearProblem(
                a, L, bcs=[bc0, bc1],
                petsc_options_prefix="basic_linear_problem_",
                petsc_options= {
                  "ksp_type": "preonly",
                  "pc_type": "lu",
                  "pc_factor_mat_solver_type": "mumps"
                },
                restriction=restriction
            )

        This class also supports nested block-structured problems.

            problem = LinearProblem(
                [[a00, a01], [None, a11]], [L0, L1],
                bcs=[bc0, bc1], u=[uh0, uh1],
                kind="nest",
                petsc_options_prefix="nest_linear_problem_",
                restriction=restriction
            )

        Every PETSc object created will have a unique options prefix set.
        We recommend discovering these prefixes dynamically via the petsc4py API rather than hard-coding
        each prefix value into the programme.

            ksp_options_prefix = problem.solver.getOptionsPrefix()
            A_options_prefix = problem.A.getOptionsPrefix()

        Parameters
        ----------
        a
            Bilinear UFL form or a nested sequence of bilinear forms, the left-hand side of the variational problem.
        L
            Linear UFL form or a sequence of linear forms, the right-hand side of the variational problem.
        bcs
            Sequence of Dirichlet boundary conditions to apply to
            the variational problem and the preconditioner matrix.
        u
            Solution function. It is created if not provided.
        P
            Bilinear UFL form or a sequence of sequence of bilinear forms, used as a preconditioner.
        kind
            The PETSc matrix and vector type. Common choices are ``mpi`` and ``nest``.
            See :func:`dolfinx.fem.petsc.create_matrix` and :func:`dolfinx.fem.petsc.create_vector`
            for more information.
        petsc_options_prefix
            Mandatory named argument. Options prefix used as root prefix on all internally created
            PETSc objects. Typically ends with ``_``. Must be the same on all ranks, and is usually unique
            within the programme.
        petsc_options
            Options set on the underlying PETSc KSP only.
            The options must be the same on all ranks. For available choices for the ``petsc_options`` kwarg,
            see the `PETSc KSP documentation
            <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`_.
            Options on other objects (matrices, vectors) should be set explicitly by the user.
        form_compiler_options
            Options used in FFCx compilation of all forms. Run ``ffcx --help`` at the commandline to see
            all available options.
        jit_options
            Options used in CFFI JIT compilation of C code generated by FFCx. See ``python/dolfinx/jit.py`` for
            all available options. Takes priority over all other option values.
        restriction
            A dofmap restriction. If not provided, the unrestricted problem will be solved.
        """
        self._a = dolfinx.fem.form(
            a, dtype=petsc4py.PETSc.ScalarType,  # type: ignore[attr-defined]
            form_compiler_options=form_compiler_options, jit_options=jit_options
        )
        self._L = dolfinx.fem.form(
            L, dtype=petsc4py.PETSc.ScalarType,  # type: ignore[attr-defined]
            form_compiler_options=form_compiler_options, jit_options=jit_options
        )
        self._A = create_matrix(self._a, kind=kind, restriction=(restriction, restriction))
        self._preconditioner = dolfinx.fem.form(
            P, dtype=petsc4py.PETSc.ScalarType,  # type: ignore[attr-defined]
            form_compiler_options=form_compiler_options, jit_options=jit_options
        )
        self._P_mat = (
            create_matrix(self._preconditioner, kind=kind, restriction=(restriction, restriction))
            if self._preconditioner is not None else None
        )

        # For nest matrices kind can be a nested list.
        kind = "nest" if self.A.getType() == petsc4py.PETSc.Mat.Type.NEST else kind  # type: ignore[attr-defined]
        assert kind is None or isinstance(kind, str)
        self._b = create_vector(self.L, kind=kind, restriction=restriction)
        self._x = create_vector(self.L, kind=kind, restriction=restriction)

        if u is None:
            # Extract function space for unknown from the right hand side of the equation.
            if isinstance(L, collections.abc.Sequence):
                self._u = [dolfinx.fem.Function(Li.arguments()[0].ufl_function_space()) for Li in L]
            else:
                self._u = dolfinx.fem.Function(L.arguments()[0].ufl_function_space())
        else:  # pragma: no cover
            self._u = u  # type: ignore[assignment]

        self.bcs = [] if bcs is None else bcs

        self._solver = petsc4py.PETSc.KSP().create(self.A.comm)  # type: ignore[attr-defined]
        self.solver.setOperators(self.A, self.P_mat)

        # Set options prefix for PETSc objects
        if petsc_options_prefix == "":  # pragma: no cover
            raise ValueError("PETSc options prefix cannot be empty.")
        self.solver.setOptionsPrefix(petsc_options_prefix)
        self.A.setOptionsPrefix(f"{petsc_options_prefix}A_")
        self.b.setOptionsPrefix(f"{petsc_options_prefix}b_")
        self.x.setOptionsPrefix(f"{petsc_options_prefix}x_")
        if self.P_mat is not None:  # pragma: no cover
            self.P_mat.setOptionsPrefix(f"{petsc_options_prefix}P_mat_")

        # Set options on KSP only
        if petsc_options is not None:
            opts = petsc4py.PETSc.Options()  # type: ignore[attr-defined]
            opts.prefixPush(self.solver.getOptionsPrefix())

            for k, v in petsc_options.items():
                opts[k] = v

            self.solver.setFromOptions()

            # Tidy up global options
            for k in petsc_options.keys():
                del opts[k]

            opts.prefixPop()

        if self.P_mat is not None and kind == "nest":  # pragma: no cover
            # Transfer nest IS on self.P_mat to PC of main KSP. This allows
            # fieldsplit preconditioning to be applied, if desired.
            nest_IS = self.P_mat.getNestISs()
            fieldsplit_IS = tuple(
                [
                    (f"{u.name + '_' if u.name != 'f' else ''}{i}", IS)
                    for i, (u, IS) in enumerate(zip(self.u, nest_IS[0]))
                ]
            )
            self.solver.getPC().setFieldSplitIS(*fieldsplit_IS)

        self._restriction = restriction

    def __del__(self) -> None:
        """Clean up PETSc data structures."""
        self._solver.destroy()
        self._A.destroy()
        self._b.destroy()
        self._x.destroy()
        if self._P_mat is not None:  # pragma: no cover
            self._P_mat.destroy()

    def solve(self) -> typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]]:
        """
        Solve the problem.

        This method updates the solution ``u`` function(s) stored in the problem instance.

        Returns
        -------
        :
            The solution function(s).

        Notes
        -----
        The user is responsible for asserting convergence of the KSP solver e.g.
        ``problem.solver.getConvergedReason() > 0``.
        Alternatively, pass ``"ksp_error_if_not_converged" : True`` in ``petsc_options`` to raise
        a ``PETScError`` on failure.
        """
        # Assemble lhs
        self.A.zeroEntries()
        assemble_matrix(
            self.A, self.a, bcs=self.bcs,  # type: ignore[arg-type, misc]
            restriction=(self.restriction, self.restriction))
        self.A.assemble()

        # Assemble preconditioner
        if self.P_mat is not None:  # pragma: no cover
            self.P_mat.zeroEntries()
            assemble_matrix(
                self.P_mat, self.preconditioner, bcs=self.bcs,  # type: ignore[arg-type, misc]
                restriction=(self.restriction, self.restriction))
            self.P_mat.assemble()

        # Assemble rhs
        dolfinx.la.petsc._zero_vector(self.b)
        assemble_vector(self.b, self.L, restriction=self.restriction)  # type: ignore[arg-type]

        # Apply boundary conditions to the rhs
        if self.bcs is not None:
            if isinstance(self.u, collections.abc.Sequence):  # block or nest
                assert isinstance(self.a, collections.abc.Sequence)
                function_spaces = (
                    dolfinx.fem.extract_function_spaces(self.a, 0), dolfinx.fem.extract_function_spaces(self.a, 1))
                bcs1 = dolfinx.fem.bcs_by_block(function_spaces[1], self.bcs)
                apply_lifting(self.b, self.a, bcs=bcs1, restriction=self.restriction)
                dolfinx.la.petsc._ghost_update(
                    self.b, petsc4py.PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
                    petsc4py.PETSc.ScatterMode.REVERSE  # type: ignore[attr-defined]
                )
                bcs0 = dolfinx.fem.bcs_by_block(function_spaces[0], self.bcs)
                set_bc(self.b, bcs0, restriction=self.restriction)
            else:  # single
                apply_lifting(self.b, [self.a], bcs=[self.bcs], restriction=self.restriction)  # type: ignore[arg-type]
                dolfinx.la.petsc._ghost_update(
                    self.b, petsc4py.PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
                    petsc4py.PETSc.ScatterMode.REVERSE  # type: ignore[attr-defined]
                )
                set_bc(self.b, self.bcs, restriction=self.restriction)
        else:  # pragma: no cover
            dolfinx.la.petsc._ghost_update(
                self.b, petsc4py.PETSc.InsertMode.ADD,
                petsc4py.PETSc.ScatterMode.REVERSE
            )

        # Solve linear system and update ghost values in the solution
        self.solver.solve(self.b, self.x)
        dolfinx.la.petsc._ghost_update(
            self.x, petsc4py.PETSc.InsertMode.INSERT,   # type: ignore[attr-defined]
            petsc4py.PETSc.ScatterMode.FORWARD  # type: ignore[attr-defined]
        )
        assign(self.x, self.u, self.restriction)
        return self.u

    @property
    def L(self) -> DolfinxRank1FormsType:
        """The compiled linear form representing the left-hand side."""
        return self._L  # type: ignore[no-any-return]

    @property
    def a(self) -> DolfinxRank2FormsType:
        """The compiled bilinear form representing the right-hand side."""
        return self._a  # type: ignore[no-any-return]

    @property
    def preconditioner(self) -> DolfinxRank2FormsType:  # pragma: no cover
        """The compiled bilinear form representing the preconditioner."""
        return self._preconditioner  # type: ignore[no-any-return]

    @property
    def A(self) -> petsc4py.PETSc.Mat:  # type: ignore[name-defined]
        """Left-hand side matrix."""
        return self._A

    @property
    def P_mat(self) -> typing.Optional[petsc4py.PETSc.Mat]:  # type: ignore[name-defined]
        """Preconditioner matrix."""
        return self._P_mat

    @property
    def b(self) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
        """Right-hand side vector."""
        return self._b

    @property
    def x(self) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
        """
        Solution vector.

        Notes
        -----
        The vector does not share memory with the solution function(s) ``u``.
        """
        return self._x

    @property
    def solver(self) -> petsc4py.PETSc.KSP:  # type: ignore[name-defined]
        """The PETSc KSP solver."""
        return self._solver

    @property
    def u(self) -> typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]]:
        """
        Solution function(s).

        Notes
        -----
        The function(s) do not share memory with the solution vector ``x``.
        """
        return self._u

    @property
    def restriction(self) -> MultiphenicsxRank1RestrictionsType:
        """The dofmap restriction."""
        return self._restriction


# -- High-level interface for SNES ---------------------------------------


def assemble_residual(
    u: typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]],
    residual: DolfinxRank1FormsType, jacobian: DolfinxRank2FormsType,
    bcs: typing.Sequence[dolfinx.fem.DirichletBC],
    restriction: MultiphenicsxRank1RestrictionsType,
    restriction_x0: MultiphenicsxRank1RestrictionsType,
    _snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, b: petsc4py.PETSc.Vec  # type: ignore[name-defined]
) -> None:
    """Assemble the residual at ``x`` into the vector ``b``."""
    # Update input vector before assigning
    dolfinx.la.petsc._ghost_update(
        x, petsc4py.PETSc.InsertMode.INSERT,   # type: ignore[attr-defined]
        petsc4py.PETSc.ScatterMode.FORWARD  # type: ignore[attr-defined]
    )

    # Copy the input vector into the `dolfinx.fem.Function` used in the forms
    assign(x, u, restriction)

    # Attach _dofmaps attribute if b contains a block vector
    if (
        isinstance(residual, collections.abc.Sequence)
        and b.getType() != petsc4py.PETSc.Vec.Type.NEST  # type: ignore[attr-defined]
        and (b.getAttr("_dofmaps") is None or x.getAttr("_dofmaps") is None)
    ):
        function_spaces_residual: list[dolfinx.fem.FunctionSpace] = (
            dolfinx.fem.extract_function_spaces(residual))  # type: ignore[assignment]
        dofmaps = [function_space.dofmap for function_space in function_spaces_residual]
        if b.getAttr("_dofmaps") is None:
            b.setAttr("_dofmaps", dofmaps)
        if x.getAttr("_dofmaps") is None:
            x.setAttr("_dofmaps", dofmaps)

    # Assemble the residual
    dolfinx.la.petsc._zero_vector(b)
    assemble_vector(b, residual, restriction=restriction)  # type: ignore[arg-type]

    # Apply lifting and set boundary conditions
    if isinstance(jacobian, collections.abc.Sequence):  # nest or block forms
        function_spaces: tuple[list[dolfinx.fem.FunctionSpace], list[dolfinx.fem.FunctionSpace]] = (  # type: ignore
            dolfinx.fem.extract_function_spaces(jacobian, 0), dolfinx.fem.extract_function_spaces(jacobian, 1))
        bcs1 = dolfinx.fem.bcs_by_block(function_spaces[1], bcs)
        apply_lifting(
            b, jacobian, bcs=bcs1, x0=x, alpha=-1.0, restriction=restriction, restriction_x0=restriction_x0)
        dolfinx.la.petsc._ghost_update(
            b, petsc4py.PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
            petsc4py.PETSc.ScatterMode.REVERSE  # type: ignore[attr-defined]
        )
        bcs0 = dolfinx.fem.bcs_by_block(function_spaces[0], bcs)
        set_bc(b, bcs0, x0=x, alpha=-1.0, restriction=restriction, restriction_x0=restriction_x0)
    else:  # single form
        apply_lifting(
            b, [jacobian], bcs=[bcs], x0=[x], alpha=-1.0, restriction=restriction, restriction_x0=[restriction_x0])
        dolfinx.la.petsc._ghost_update(
            b, petsc4py.PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
            petsc4py.PETSc.ScatterMode.REVERSE  # type: ignore[attr-defined]
        )
        set_bc(b, bcs, x0=x, alpha=-1.0, restriction=restriction, restriction_x0=restriction_x0)
    dolfinx.la.petsc._ghost_update(
        b, petsc4py.PETSc.InsertMode.INSERT,  # type: ignore[attr-defined]
        petsc4py.PETSc.ScatterMode.FORWARD  # type: ignore[attr-defined]
    )


def assemble_jacobian(
    u: typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]],
    jacobian: DolfinxRank2FormsType, preconditioner: typing.Optional[DolfinxRank2FormsType],
    bcs: typing.Sequence[dolfinx.fem.DirichletBC],
    restriction: MultiphenicsxRank1RestrictionsType,
    _snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, J: petsc4py.PETSc.Mat,  # type: ignore[name-defined]
    P_mat: petsc4py.PETSc.Mat  # type: ignore[name-defined]
) -> None:
    """Assemble the Jacobian and preconditioner at ``x`` into matrices ``J`` and ``P_mat``."""
    # Update input vector before assigning
    dolfinx.la.petsc._ghost_update(
        x, petsc4py.PETSc.InsertMode.INSERT,  # type: ignore[attr-defined]
        petsc4py.PETSc.ScatterMode.FORWARD  # type: ignore[attr-defined]
    )

    # Copy the input vector into the `dolfinx.fem.Function` used in the forms
    assign(x, u, restriction)

    # Assemble Jacobian
    J.zeroEntries()
    assemble_matrix(
        J, jacobian, bcs, diag=1.0, restriction=(restriction, restriction))  # type: ignore[arg-type, misc]
    J.assemble()
    if preconditioner is not None:  # pragma: no cover
        P_mat.zeroEntries()
        assemble_matrix(
            P_mat, preconditioner, bcs, diag=1.0,  # type: ignore[arg-type, misc]
            restriction=(restriction, restriction))
        P_mat.assemble()


class NonlinearProblem:
    r"""
    High-level class for solving nonlinear variational problems with PETSc SNES.

    Solves problems of the form :math:`F_i(u, v) = 0, i=0,\\ldots,N\\ \\forall v \\in V` where
    :math:`u=(u_0,\\ldots,u_N), v=(v_0,\\ldots,v_N)` using PETSc SNES as the non-linear solver.
    """

    def __init__(
        self,
        F: UflRank1FormsType,
        u: typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]], *,
        petsc_options_prefix: str,
        bcs: typing.Optional[typing.Sequence[dolfinx.fem.DirichletBC]] = None,
        J: typing.Optional[UflRank2FormsType] = None, P: typing.Optional[UflRank2FormsType] = None,
        kind: DolfinxMatrixKindType = None,
        petsc_options: typing.Optional[dict[str, typing.Any]] = None,
        form_compiler_options: typing.Optional[dict[str, typing.Any]] = None,
        jit_options: typing.Optional[dict[str, typing.Any]] = None,
        restriction: MultiphenicsxRank1RestrictionsType = None
    ) -> None:
        """
        Initialize solver for a nonlinear variational problem.

        By default, the underlying SNES solver uses PETSc's default options.
        To use the robust combination of LU via MUMPS with a backtracking linesearch, pass:

            petsc_options = {"ksp_type": "preonly",
                             "pc_type": "lu",
                             "pc_factor_mat_solver_type": "mumps",
                             "snes_linesearch_type": "bt",
            }

        Every PETSc object will have a unique options prefix set.
        We recommend discovering these prefixes dynamically via the petsc4py API rather than hard-coding
        each prefix value into the programme.

            snes_options_prefix = problem.solver.getOptionsPrefix()
            jacobian_options_prefix = problem.A.getOptionsPrefix()

        Parameters
        ----------
        F
            UFL form(s) representing the residual :math:`F_i`.
        u
            Function(s) used to define the residual and Jacobian.
        bcs
            Dirichlet boundary conditions.
        J
            UFL form(s) representing the Jacobian :math:`J_{ij} = dF_i/du_j`. If not passed, derived automatically.
        P
            UFL form(s) representing the preconditioner.
        kind
            The PETSc matrix and vector type. Common choices are ``mpi`` and ``nest``.
            See :func:`dolfinx.fem.petsc.create_matrix` and :func:`dolfinx.fem.petsc.create_vector`
            for more information.
        petsc_options_prefix
            Mandatory named argument. Options prefix used as root prefix on all internally created
            PETSc objects. Typically ends with `_`. Must be the same on all ranks, and is usually unique
            within the programme.
        petsc_options
            Options set on the underlying PETSc SNES only.
            The options must be the same on all ranks. For available choices for the ``petsc_options`` kwarg,
            see the `PETSc SNES documentation
            <https://petsc4py.readthedocs.io/en/stable/manual/snes/>`_.
            Options on other objects (matrices, vectors) should be set explicitly by the user.
        form_compiler_options
            Options used in FFCx compilation of all forms. Run ``ffcx --help`` at the commandline to see
            all available options.
        jit_options
            Options used in CFFI JIT compilation of C code generated by FFCx. See `python/dolfinx/jit.py` for
            all available options. Takes priority over all other option values.
        restriction
            A dofmap restriction. If not provided, the unrestricted problem will be solved.
        """
        # Compile residual and Jacobian forms
        self._F = dolfinx.fem.form(
            F, dtype=petsc4py.PETSc.ScalarType,  # type: ignore[attr-defined]
            form_compiler_options=form_compiler_options, jit_options=jit_options
        )

        if J is None:
            J = dolfinx.fem.forms.derivative_block(F, u)

        self._J = dolfinx.fem.form(
            J, dtype=petsc4py.PETSc.ScalarType,  # type: ignore[attr-defined]
            form_compiler_options=form_compiler_options, jit_options=jit_options
        )

        if P is not None:  # pragma: no cover
            self._preconditioner = dolfinx.fem.form(
                P, dtype=petsc4py.PETSc.ScalarType,  # type: ignore[attr-defined]
                form_compiler_options=form_compiler_options, jit_options=jit_options
            )
        else:
            self._preconditioner = None

        self._u = u
        bcs = [] if bcs is None else bcs

        self._A = create_matrix(self.J, kind=kind, restriction=(restriction, restriction))
        if self._preconditioner is not None:  # pragma: no cover
            self._P_mat = create_matrix(self._preconditioner, kind=kind, restriction=(restriction, restriction))
        else:
            self._P_mat = None

        # Determine the vector kind based on the matrix type
        kind = "nest" if self._A.getType() == petsc4py.PETSc.Mat.Type.NEST else kind  # type: ignore[attr-defined]
        assert kind is None or isinstance(kind, str)
        self._b = create_vector(self.F, kind=kind, restriction=restriction)
        self._x = create_vector(self.F, kind=kind, restriction=restriction)

        # Create the SNES solver and attach the corresponding Jacobian and esidual computation functions
        self._snes = petsc4py.PETSc.SNES().create(self.A.comm)  # type: ignore[attr-defined]
        self.solver.setJacobian(
            functools.partial(assemble_jacobian, u, self.J, self.preconditioner, bcs, restriction),
            self.A, self.P_mat
        )
        self.solver.setFunction(
            functools.partial(assemble_residual, u, self.F, self.J, bcs, restriction, restriction),
            self.b
        )

        # Set options prefix for PETSc objects
        if petsc_options_prefix == "":  # pragma: no cover
            raise ValueError("PETSc options prefix cannot be empty.")
        self.solver.setOptionsPrefix(petsc_options_prefix)
        self.A.setOptionsPrefix(f"{petsc_options_prefix}A_")
        if self.P_mat is not None:  # pragma: no cover
            self.P_mat.setOptionsPrefix(f"{petsc_options_prefix}P_mat_")
        self.b.setOptionsPrefix(f"{petsc_options_prefix}b_")
        self.x.setOptionsPrefix(f"{petsc_options_prefix}x_")

        # Set options for SNES only
        if petsc_options is not None:
            opts = petsc4py.PETSc.Options()  # type: ignore[attr-defined]
            opts.prefixPush(self.solver.getOptionsPrefix())

            for k, v in petsc_options.items():
                opts[k] = v

            self.solver.setFromOptions()

            # Tidy up global options
            for k in petsc_options.keys():
                del opts[k]

            opts.prefixPop()

        if self.P_mat is not None and kind == "nest":  # pragma: no cover
            # Transfer nest IS on self.P_mat to PC of main KSP. This allows
            # fieldsplit preconditioning to be applied, if desired.
            nest_IS = self.P_mat.getNestISs()
            fieldsplit_IS = tuple(
                [
                    (f"{u.name + '_' if u.name != 'f' else ''}{i}", IS)
                    for i, (u, IS) in enumerate(zip(self.u, nest_IS[0]))
                ]
            )
            self.solver.getKSP().getPC().setFieldSplitIS(*fieldsplit_IS)

        self._restriction = restriction

    def __del__(self) -> None:
        """Clean up PETSc data structures."""
        self._snes.destroy()
        self._x.destroy()
        self._A.destroy()
        self._b.destroy()
        if self._P_mat is not None:  # pragma: no cover
            self._P_mat.destroy()

    def solve(self) -> typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]]:
        """
        Solve the problem.

        This method updates the solution ``u`` function(s) stored in the problem instance.

        Returns
        -------
        :
            The solution function(s).

        Notes
        -----
        The user is responsible for asserting convergence of the KSP solver e.g.
        ``problem.solver.getConvergedReason() > 0``.
        Alternatively, pass ``"snes_error_if_not_converged" : True`` and ``"ksp_error_if_not_converged" : True``
        in ``petsc_options`` to raise a ``PETScError`` on failure.
        """
        # Copy current iterate into the work array.
        assign(self.u, self.x, restriction=self.restriction)

        # Solve problem
        self.solver.solve(None, self.x)
        dolfinx.la.petsc._ghost_update(
            self.x, petsc4py.PETSc.InsertMode.INSERT,  # type: ignore[attr-defined]
            petsc4py.PETSc.ScatterMode.FORWARD  # type: ignore[attr-defined]
        )

        # Copy solution back to function
        assign(self.x, self.u, restriction=self.restriction)
        return self.u

    @property
    def F(self) -> DolfinxRank1FormsType:
        """The compiled residual."""
        return self._F  # type: ignore[no-any-return]

    @property
    def J(self) -> DolfinxRank2FormsType:
        """The compiled Jacobian."""
        return self._J  # type: ignore[no-any-return]

    @property
    def preconditioner(self) -> typing.Optional[DolfinxRank2FormsType]:
        """The compiled preconditioner."""
        return self._preconditioner  # type: ignore[no-any-return]

    @property
    def A(self) -> petsc4py.PETSc.Mat:  # type: ignore[name-defined]
        """Jacobian matrix."""
        return self._A

    @property
    def P_mat(self) -> typing.Optional[petsc4py.PETSc.Mat]:  # type: ignore[name-defined]
        """Preconditioner matrix."""
        return self._P_mat

    @property
    def b(self) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
        """Residual vector."""
        return self._b

    @property
    def x(self) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
        """
        Solution vector.

        Notes
        -----
        The vector does not share memory with the solution function(s) `u`.
        """
        return self._x

    @property
    def solver(self) -> petsc4py.PETSc.SNES:  # type: ignore[name-defined]
        """The SNES solver."""
        return self._snes

    @property
    def u(self) -> typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]]:
        """
        Solution function(s).

        Notes
        -----
        The function(s) do not share memory with the solution vector ``x``.
        """
        return self._u

    @property
    def restriction(self) -> MultiphenicsxRank1RestrictionsType:
        """The dofmap restriction."""
        return self._restriction

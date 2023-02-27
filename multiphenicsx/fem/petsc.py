# Copyright (C) 2016-2023 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Assembly functions for variational forms."""

import contextlib
import functools
import types
import typing

import dolfinx.cpp as dcpp
import dolfinx.fem
import dolfinx.fem.assemble
import dolfinx.fem.forms
import numpy as np
import numpy.typing
import petsc4py.PETSc

from multiphenicsx.cpp import cpp_library as mcpp

DolfinxConstantsType = np.typing.NDArray[petsc4py.PETSc.ScalarType]
DolfinxCoefficientsType = typing.Dict[
    typing.Tuple[dcpp.fem.IntegralType, int],
    np.typing.NDArray[petsc4py.PETSc.ScalarType]
]


def _get_block_function_spaces(block_form: typing.List[typing.Any]) -> typing.List[typing.Any]:
    if isinstance(block_form[0], list):
        return _get_block_function_spaces_rank_2(block_form)
    else:
        return _get_block_function_spaces_rank_1(block_form)


def _get_block_function_spaces_rank_1(  # type: ignore[no-any-unimported]
    block_form: typing.List[dolfinx.fem.forms.form_types]
) -> typing.List[dolfinx.fem.FunctionSpace]:
    cpp_Form = dcpp.fem.Form_float64 if petsc4py.PETSc.ScalarType == np.float64 else dcpp.fem.Form_complex128
    assert all(isinstance(block_form_, cpp_Form) for block_form_ in block_form)
    return [form.function_spaces[0] for form in block_form]


def _get_block_function_spaces_rank_2(  # type: ignore[no-any-unimported]
    block_form: typing.List[typing.List[dolfinx.fem.forms.form_types]]
) -> typing.List[typing.List[dolfinx.fem.FunctionSpace]]:
    cpp_Form = dcpp.fem.Form_float64 if petsc4py.PETSc.ScalarType == np.float64 else dcpp.fem.Form_complex128
    assert all(isinstance(block_form_, list) for block_form_ in block_form)
    assert all(isinstance(form, cpp_Form) or form is None for block_form_ in block_form for form in block_form_)
    a = block_form
    rows = len(a)
    cols = len(a[0])
    assert all(len(a_i) == cols for a_i in a)
    assert all(
        a[i][j] is None or a[i][j].rank == 2 for i in range(rows) for j in range(cols))  # type: ignore[union-attr]
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
    L: dolfinx.fem.forms.form_types, restriction: typing.Optional[mcpp.fem.DofMapRestriction] = None
) -> petsc4py.PETSc.Vec:
    """
    Create a PETSc vector which can be used to assemble the form `L` with restriction `restriction`.

    Parameters
    ----------
    L
        A linear form
    restriction
        A dofmap restriction. If not provided, the unrestricted tensor will be created.

    Returns
    -------
    :
        A PETSc vector with a layout that is compatible with `L` and restriction `restriction`.
    """
    dofmap = L.function_spaces[0].dofmap
    if restriction is None:
        index_map = dofmap.index_map
        index_map_bs = dofmap.index_map_bs
    else:
        assert _same_dofmap(restriction.dofmap, dofmap)
        index_map = restriction.index_map
        index_map_bs = restriction.index_map_bs
    return dcpp.la.petsc.create_vector(index_map, index_map_bs)


def create_vector_block(  # type: ignore[no-any-unimported]
    L: typing.List[dolfinx.fem.forms.form_types],
    restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None
) -> petsc4py.PETSc.Vec:
    """
    Create a block PETSc vector which can be used to assemble the forms `L` with restriction `restriction`.

    Parameters
    ----------
    L
        A list of linear forms.
    restriction
        A dofmap restriction. If not provided, the unrestricted tensor will be created.

    Returns
    -------
    :
        A PETSc vector with a blocked layout that is compatible with `L` and restriction `restriction`.
    """
    function_spaces = _get_block_function_spaces(L)
    dofmaps = [function_space.dofmap for function_space in function_spaces]
    if restriction is None:
        index_maps = [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps]
    else:
        assert len(restriction) == len(dofmaps)
        assert all(_same_dofmap(restriction_.dofmap, dofmap) for (restriction_, dofmap) in zip(restriction, dofmaps))
        index_maps = [(restriction_.index_map, restriction_.index_map_bs) for restriction_ in restriction]
    return dcpp.fem.petsc.create_vector_block(index_maps)


def create_vector_nest(  # type: ignore[no-any-unimported]
    L: typing.List[dolfinx.fem.forms.form_types],
    restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None
) -> petsc4py.PETSc.Vec:
    """
    Create a nest PETSc vector which can be used to assemble the forms `L` with restriction `restriction`.

    Parameters
    ----------
    L
        A list of linear forms.
    restriction
        A dofmap restriction. If not provided, the unrestricted tensor will be created.

    Returns
    -------
    :
        A PETSc vector with a nest layout that is compatible with `L` and restriction `restriction`.
    """
    function_spaces = _get_block_function_spaces(L)
    dofmaps = [function_space.dofmap for function_space in function_spaces]
    if restriction is None:
        index_maps = [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps]
    else:
        assert len(restriction) == len(dofmaps)
        assert all(_same_dofmap(restriction_.dofmap, dofmap) for (restriction_, dofmap) in zip(restriction, dofmaps))
        index_maps = [(restriction_.index_map, restriction_.index_map_bs) for restriction_ in restriction]
    return dcpp.fem.petsc.create_vector_nest(index_maps)


# -- Matrix instantiation ----------------------------------------------------

def create_matrix(  # type: ignore[no-any-unimported]
    a: dolfinx.fem.forms.form_types,
    restriction: typing.Optional[typing.Tuple[mcpp.fem.DofMapRestriction, mcpp.fem.DofMapRestriction]] = None,
    mat_type: typing.Optional[str] = None
) -> petsc4py.PETSc.Mat:
    """
    Create a PETSc matrix which can be used to assemble the bilinear form `a` with restriction `restriction`.

    Parameters
    ----------
    a
        A bilinear form
    restriction
        A dofmap restriction. If not provided, the unrestricted tensor will be created.
    mat_type
        The PETSc matrix type (``MatType``).

    Returns
    -------
    :
        A PETSc matrix with a layout that is compatible with `a` and restriction `restriction`.
    """
    assert a.rank == 2  # type: ignore[union-attr]
    mesh = a.mesh
    function_spaces = a.function_spaces
    assert all(function_space.mesh == mesh for function_space in function_spaces)
    if restriction is None:
        index_maps = [function_space.dofmap.index_map for function_space in function_spaces]
        index_maps_bs = [function_space.dofmap.index_map_bs for function_space in function_spaces]
    else:
        assert len(restriction) == 2
        index_maps = [restriction_.index_map for restriction_ in restriction]
        index_maps_bs = [restriction_.index_map_bs for restriction_ in restriction]
    integral_types = list(mcpp.fem.get_integral_types_from_form(a))
    if restriction is None:
        dofmaps_lists = [function_space.dofmap.list() for function_space in function_spaces]
    else:
        dofmaps_lists = [restriction_.list() for restriction_ in restriction]
    if mat_type is not None:
        return mcpp.fem.petsc.create_matrix(mesh, index_maps, index_maps_bs, integral_types, dofmaps_lists, mat_type)
    else:
        return mcpp.fem.petsc.create_matrix(mesh, index_maps, index_maps_bs, integral_types, dofmaps_lists)


def _create_matrix_block_or_nest(  # type: ignore[no-any-unimported]
    a: typing.List[typing.List[dolfinx.fem.forms.form_types]],
    restriction: typing.Optional[
        typing.Tuple[typing.List[mcpp.fem.DofMapRestriction], typing.List[mcpp.fem.DofMapRestriction]]],
    mat_type: typing.Optional[typing.Union[str, typing.List[str]]],
    cpp_create_function: typing.Callable  # type: ignore[type-arg]
) -> petsc4py.PETSc.Mat:
    function_spaces = _get_block_function_spaces(a)
    rows, cols = len(function_spaces[0]), len(function_spaces[1])
    mesh = None
    for j in range(cols):
        for i in range(rows):
            if a[i][j] is not None:
                mesh = a[i][j].mesh
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
    integral_types_set: typing.List[typing.List[  # type: ignore[no-any-unimported]
        typing.Set[dcpp.fem.IntegralType]]] = [[set() for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if a[i][j] is not None:
                integral_types_set[i][j].update(mcpp.fem.get_integral_types_from_form(a[i][j]))
    integral_types = [[
        list(integral_types_set[row][col]) for col in range(cols)] for row in range(rows)]
    if restriction is None:
        dofmaps_lists = (
            [function_spaces[0][i].dofmap.list() for i in range(rows)],
            [function_spaces[1][j].dofmap.list() for j in range(cols)])
    else:
        dofmaps_lists = (
            [restriction[0][i].list() for i in range(rows)],
            [restriction[1][j].list() for j in range(cols)])
    if mat_type is not None:
        return cpp_create_function(mesh, index_maps, index_maps_bs, integral_types, dofmaps_lists, mat_type)
    else:
        return cpp_create_function(mesh, index_maps, index_maps_bs, integral_types, dofmaps_lists)


def create_matrix_block(  # type: ignore[no-any-unimported]
    a: typing.List[typing.List[dolfinx.fem.forms.form_types]],
    restriction: typing.Optional[
        typing.Tuple[typing.List[mcpp.fem.DofMapRestriction], typing.List[mcpp.fem.DofMapRestriction]]],
    mat_type: typing.Optional[str] = None
) -> petsc4py.PETSc.Mat:
    """
    Create a block PETSc matrix which can be used to assemble the bilinear forms `a` with restriction `restriction`.

    Parameters
    ----------
    a
        A rectangular array of bilinear forms.
    restriction
        A dofmap restriction. If not provided, the unrestricted tensor will be created.
    mat_type
        The PETSc matrix type (``MatType``).

    Returns
    -------
    :
        A PETSc matrix with a blocked layout that is compatible with `a` and restriction `restriction`.
    """
    return _create_matrix_block_or_nest(a, restriction, mat_type, mcpp.fem.petsc.create_matrix_block)


def create_matrix_nest(  # type: ignore[no-any-unimported]
    a: typing.List[typing.List[dolfinx.fem.forms.form_types]],
    restriction: typing.Optional[
        typing.Tuple[typing.List[mcpp.fem.DofMapRestriction], typing.List[mcpp.fem.DofMapRestriction]]],
    mat_types: typing.Optional[typing.List[str]] = None
) -> petsc4py.PETSc.Mat:
    """
    Create a nest PETSc matrix which can be used to assemble the bilinear forms `a` with restriction `restriction`.

    Parameters
    ----------
    a
        A rectangular array of bilinear forms.
    restriction
        A dofmap restriction. If not provided, the unrestricted tensor will be created.
    mat_types
        The PETSc matrix types (``MatType``).

    Returns
    -------
    :
        A PETSc matrix with a nest layout that is compatible with `a` and restriction `restriction`.
    """
    return _create_matrix_block_or_nest(a, restriction, mat_types, mcpp.fem.petsc.create_matrix_nest)


# -- Vector assembly ---------------------------------------------------------

def _VecSubVectorWrapperBase(CppWrapperClass: typing.Type) -> typing.Type:  # type: ignore[type-arg]

    class _VecSubVectorWrapperBase_Class(object):
        """Wrap a PETSc Vec object."""

        def __init__(  # type: ignore[no-any-unimported]
            self, b: petsc4py.PETSc.Vec, unrestricted_index_set: petsc4py.PETSc.IS,
            restricted_index_set: typing.Optional[petsc4py.PETSc.IS] = None,
            unrestricted_to_restricted: typing.Optional[typing.Dict[int, int]] = None,
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

        def __enter__(self) -> np.typing.NDArray[petsc4py.PETSc.ScalarType]:  # type: ignore[no-any-unimported]
            """Return Vec content when entering the context."""
            return self._cpp_object.content  # type: ignore[no-any-return]

        def __exit__(
            self, exception_type: typing.Type[BaseException], exception_value: BaseException,
            traceback: types.TracebackType
        ) -> None:
            """Do nothing when leaving the context."""
            pass

    return _VecSubVectorWrapperBase_Class


_VecSubVectorReadWrapper = _VecSubVectorWrapperBase(mcpp.la.petsc.VecSubVectorReadWrapper)


class _VecSubVectorWrapper(_VecSubVectorWrapperBase(mcpp.la.petsc.VecSubVectorWrapper)):  # type: ignore[misc]
    def __exit__(
        self, exception_type: typing.Type[BaseException], exception_value: BaseException,
        traceback: types.TracebackType
    ) -> None:
        """Restore the Vec content when leaving the context."""
        self._cpp_object.restore()


def VecSubVectorWrapperBase(_VecSubVectorWrapperClass: typing.Type) -> typing.Type:  # type: ignore[type-arg]
    """Return the base class to wrap VecSubVectorWrapper or VecSubVectorReadWrapper."""

    class VecSubVectorWrapperBase_Class(object):
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
                np.typing.NDArray[petsc4py.PETSc.ScalarType]]:
            """Return Vec content when entering the context."""
            if self._wrapper is not None:
                return self._wrapper.__enter__()  # type: ignore[no-any-return]
            else:
                return None

        def __exit__(
            self, exception_type: typing.Type[BaseException], exception_value: BaseException,
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


def BlockVecSubVectorWrapperBase(_VecSubVectorWrapperClass: typing.Type) -> typing.Type:  # type: ignore[type-arg]
    """Return the base class to wrap BlockVecSubVectorWrapper or BlockVecSubVectorReadWrapper."""

    class BlockVecSubVectorWrapperBase_Class(object):
        """Wrap a PETSc Vec object with multiple blocks."""

        def __init__(  # type: ignore[no-any-unimported]
            self, b: typing.Union[petsc4py.PETSc.Vec, None],
            dofmaps: typing.List[dcpp.fem.DofMap],
            restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None,
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

        def __iter__(self) -> typing.Optional[  # type: ignore[no-any-unimported]
                typing.Iterator[np.typing.NDArray[petsc4py.PETSc.ScalarType]]]:
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
            self, exception_type: typing.Type[BaseException], exception_value: BaseException,
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


def NestVecSubVectorWrapperBase(VecSubVectorWrapperClass: typing.Type) -> typing.Type:  # type: ignore[type-arg]
    """Return the base class to wrap NestVecSubVectorWrapper or NestVecSubVectorReadWrapper."""

    class NestVecSubVectorWrapperBase_Class(object):
        """Wrap a PETSc Vec object with nested blocks."""

        def __init__(  # type: ignore[no-any-unimported]
            self, b: typing.Union[petsc4py.PETSc.Vec, typing.List[petsc4py.PETSc.Vec], None],
            dofmaps: typing.List[dcpp.fem.DofMap],
            restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None,
            ghosted: bool = True
        ) -> None:
            if b is not None:
                if isinstance(b, list):
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

        def __iter__(self) -> typing.Optional[  # type: ignore[no-any-unimported]
                typing.Iterator[np.typing.NDArray[petsc4py.PETSc.ScalarType]]]:
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
            self, exception_type: typing.Type[BaseException], exception_value: BaseException,
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
    L: dolfinx.fem.forms.form_types,
    constants: typing.Optional[DolfinxConstantsType] = None, coeffs: typing.Optional[DolfinxCoefficientsType] = None,
    restriction: typing.Optional[mcpp.fem.DofMapRestriction] = None
) -> petsc4py.PETSc.Vec:
    """
    Assemble linear form into a new PETSc vector.

    Parameters
    ----------
    L
        A linear form
    constants
        Constants that appear in the form. If not provided, any required constants will be computed.
    coeffs
        Coefficients that appear in the form. If not provided, any required coefficients will be computed.
    restriction
        A dofmap restriction. If not provided, the unrestricted tensor will be assembled.

    Returns
    -------
    :
        The assembled PETSc vector.

    Notes
    -----
    The returned vector is not finalised, i.e. ghost values are not accumulated on the owning processes.
    """
    b = create_vector(L, restriction)
    with b.localForm() as b_local:
        b_local.set(0.0)
    return assemble_vector(b, L, constants, coeffs, restriction)  # type: ignore[call-arg, arg-type]


@assemble_vector.register
def _(  # type: ignore[no-any-unimported]
    b: petsc4py.PETSc.Vec, L: dolfinx.fem.forms.form_types,
    constants: typing.Optional[DolfinxConstantsType] = None, coeffs: typing.Optional[DolfinxCoefficientsType] = None,
    restriction: typing.Optional[mcpp.fem.DofMapRestriction] = None
) -> petsc4py.PETSc.Vec:
    """
    Assemble linear form into an existing PETSc vector.

    Parameters
    ----------
    b
        PETSc vector to assemble the contribution of the linear form into.
    L
        A linear form to assemble into `b`.
    constants
        Constants that appear in the form. If not provided, any required constants will be computed.
    coeffs
        Coefficients that appear in the form. If not provided, any required coefficients will be computed.
    restriction
        A dofmap restriction. If not provided, the unrestricted tensor will be assembled.

    Returns
    -------
    :
        The assembled PETSc vector.

    Notes
    -----
    The vector is not zeroed before assembly and it is not finalised, i.e. ghost values are not accumulated
    on the owning processes.
    """
    if restriction is None:
        with b.localForm() as b_local:
            dolfinx.fem.assemble.assemble_vector(b_local.array_w, L, constants, coeffs)  # type: ignore[call-arg]
    else:
        with VecSubVectorWrapper(b, L.function_spaces[0].dofmap, restriction) as b_sub:
            dolfinx.fem.assemble.assemble_vector(b_sub, L, constants, coeffs)  # type: ignore[call-arg]
    return b


@functools.singledispatch
def assemble_vector_nest(  # type: ignore[no-any-unimported]
    L: typing.List[dolfinx.fem.forms.form_types],
    constants: typing.Optional[typing.Sequence[typing.Optional[DolfinxConstantsType]]] = None,
    coeffs: typing.Optional[typing.Sequence[typing.Optional[DolfinxCoefficientsType]]] = None,
    restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None
) -> petsc4py.PETSc.Vec:
    """
    Assemble linear forms into a new nested PETSc vector.

    Parameters
    ----------
    L
        A list of linear forms.
    constants
        Constants that appear in the form. If not provided, any required constants will be computed.
    coeffs
        Coefficients that appear in the form. If not provided, any required coefficients will be computed.
    restriction
        A dofmap restriction. If not provided, the unrestricted tensor will be assembled.

    Returns
    -------
    :
        The assembled nested PETSc vector.

    Notes
    -----
    The returned vector is not finalised, i.e. ghost values are not accumulated on the owning processes.
    """
    b = create_vector_nest(L, restriction)
    for b_sub in b.getNestSubVecs():
        with b_sub.localForm() as b_local:
            b_local.set(0.0)
        b_sub.destroy()
    return assemble_vector_nest(b, L, constants, coeffs, restriction)  # type: ignore[call-arg, arg-type]


@assemble_vector_nest.register
def _(  # type: ignore[no-any-unimported]
    b: petsc4py.PETSc.Vec, L: typing.List[dolfinx.fem.forms.form_types],
    constants: typing.Optional[typing.Sequence[typing.Optional[DolfinxConstantsType]]] = None,
    coeffs: typing.Optional[typing.Sequence[typing.Optional[DolfinxCoefficientsType]]] = None,
    restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None
) -> petsc4py.PETSc.Vec:
    """
    Assemble linear forms into an existing nested PETSc vector.

    Parameters
    ----------
    b
        Nested PETSc vector to assemble the contribution of the linear forms into.
    L
        A list of linear forms to assemble into `b`.
    constants
        Constants that appear in the form. If not provided, any required constants will be computed.
    coeffs
        Coefficients that appear in the form. If not provided, any required coefficients will be computed.
    restriction
        A dofmap restriction. If not provided, the unrestricted tensor will be assembled.

    Returns
    -------
    :
        The assembled nested PETSc vector.

    Notes
    -----
    The vector is not zeroed before assembly and it is not finalised, i.e. ghost values are not accumulated
    on the owning processes.
    """
    constants = [None] * len(L) if constants is None else constants
    coeffs = [None] * len(L) if coeffs is None else coeffs
    function_spaces = _get_block_function_spaces(L)
    dofmaps = [function_space.dofmap for function_space in function_spaces]
    with NestVecSubVectorWrapper(b, dofmaps, restriction) as nest_b:
        for b_sub, L_sub, constant, coeff in zip(nest_b, L, constants, coeffs):
            dolfinx.fem.assemble.assemble_vector(b_sub, L_sub, constant, coeff)  # type: ignore[call-arg]
    return b


@functools.singledispatch
def assemble_vector_block(  # type: ignore[no-any-unimported]
    L: typing.List[dolfinx.fem.forms.form_types], a: typing.List[typing.List[dolfinx.fem.forms.form_types]],
    bcs: typing.List[dolfinx.fem.DirichletBCMetaClass] = [],
    x0: typing.Optional[petsc4py.PETSc.Vec] = None,
    scale: float = 1.0,
    constants_L: typing.Optional[typing.Sequence[typing.Optional[DolfinxConstantsType]]] = None,
    coeffs_L: typing.Optional[typing.Sequence[typing.Optional[DolfinxCoefficientsType]]] = None,
    constants_a: typing.Optional[typing.Sequence[typing.Sequence[typing.Optional[DolfinxConstantsType]]]] = None,
    coeffs_a: typing.Optional[typing.Sequence[typing.Sequence[typing.Optional[DolfinxCoefficientsType]]]] = None,
    restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None,
    restriction_x0: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None
) -> petsc4py.PETSc.Vec:
    """
    Assemble linear forms into a new block PETSc vector.

    Parameters
    ----------
    L
        A list of linear forms.
    bcs
        Optional list of boundary conditions.
    x0
        Optional vector storing the solution.
    scale
        Optional scaling factor for boundary conditions application.
    constants_L, constants_a
        Constants that appear in the form. If not provided, any required constants will be computed.
    coeffs_L, coeffs_a
        Coefficients that appear in the form. If not provided, any required coefficients will be computed.
    restriction, restriction_x0
        A dofmap restriction. If not provided, the unrestricted tensor will be assembled.

    Returns
    -------
    :
        The assembled block PETSc vector.

    Notes
    -----
    The returned vector is not finalised, i.e. ghost values are not accumulated on the owning processes.
    """
    b = create_vector_block(L, restriction)
    with b.localForm() as b_local:
        b_local.set(0.0)
    return assemble_vector_block(  # type: ignore[call-arg]
        b, L, a, bcs, x0, scale, constants_L, coeffs_L, constants_a, coeffs_a,  # type: ignore[arg-type]
        restriction, restriction_x0)


@assemble_vector_block.register
def _(  # type: ignore[no-any-unimported]
    b: petsc4py.PETSc.Vec, L: typing.List[dolfinx.fem.forms.form_types],
    a: typing.List[typing.List[dolfinx.fem.forms.form_types]],
    bcs: typing.List[dolfinx.fem.DirichletBCMetaClass] = [],
    x0: typing.Optional[petsc4py.PETSc.Vec] = None,
    scale: float = 1.0,
    constants_L: typing.Optional[typing.Sequence[typing.Optional[DolfinxConstantsType]]] = None,
    coeffs_L: typing.Optional[typing.Sequence[typing.Optional[DolfinxCoefficientsType]]] = None,
    constants_a: typing.Optional[typing.Sequence[typing.Sequence[typing.Optional[DolfinxConstantsType]]]] = None,
    coeffs_a: typing.Optional[typing.Sequence[typing.Sequence[typing.Optional[DolfinxCoefficientsType]]]] = None,
    restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None,
    restriction_x0: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None
) -> petsc4py.PETSc.Vec:
    """
    Assemble linear forms into an existing block PETSc vector.

    Parameters
    ----------
    b
        Block PETSc vector to assemble the contribution of the linear forms into.
    L
        A list of linear forms to assemble into `b`.
    bcs
        Optional list of boundary conditions.
    x0
        Optional vector storing the solution.
    scale
        Optional scaling factor for boundary conditions application.
    constants_L, constants_a
        Constants that appear in the form. If not provided, any required constants will be computed.
    coeffs_L, coeffs_a
        Coefficients that appear in the form. If not provided, any required coefficients will be computed.
    restriction, restriction_x0
        A dofmap restriction. If not provided, the unrestricted tensor will be assembled.

    Returns
    -------
    :
        The assembled block PETSc vector.

    Notes
    -----
    The vector is not zeroed before assembly and it is not finalised, i.e. ghost values are not accumulated
    on the owning processes.
    """
    constants_L = [
        None if form is None else dcpp.fem.pack_constants(form) for form in L] if constants_L is None else constants_L
    coeffs_L = [
        {} if form is None else dcpp.fem.pack_coefficients(form) for form in L] if coeffs_L is None else coeffs_L
    constants_a = [[
        None if form is None else dcpp.fem.pack_constants(form) for form in forms]
        for forms in a] if constants_a is None else constants_a
    coeffs_a = [[
        {} if form is None else dcpp.fem.pack_coefficients(form) for form in forms]
        for forms in a] if coeffs_a is None else coeffs_a

    function_spaces = _get_block_function_spaces(a)
    dofmaps = [function_space.dofmap for function_space in function_spaces[0]]
    dofmaps_x0 = [function_space.dofmap for function_space in function_spaces[1]]

    bcs1 = dolfinx.fem.bcs_by_block(function_spaces[1], bcs)
    with BlockVecSubVectorWrapper(b, dofmaps, restriction) as block_b, \
            BlockVecSubVectorReadWrapper(x0, dofmaps_x0, restriction_x0) as block_x0:
        for b_sub, L_sub, a_sub, constant_L, coeff_L, constant_a, coeff_a in zip(
                block_b, L, a, constants_L, coeffs_L, constants_a, coeffs_a):
            dcpp.fem.assemble_vector(b_sub, L_sub, constant_L, coeff_L)
            for a_sub_, bcs1_sub, x0_sub, constant_a_sub, coeff_a_sub in zip(
                    a_sub, bcs1, block_x0, constant_a, coeff_a):
                if x0_sub is None:
                    x0_sub_as_list = []
                else:
                    x0_sub_as_list = [x0_sub]
                dcpp.fem.apply_lifting(
                    b_sub, [a_sub_], [constant_a_sub], [coeff_a_sub], [bcs1_sub], x0_sub_as_list, scale)
    b.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)

    bcs0 = dolfinx.fem.bcs_by_block(function_spaces[0], bcs)
    with BlockVecSubVectorWrapper(b, dofmaps, restriction) as block_b, \
            BlockVecSubVectorReadWrapper(x0, dofmaps_x0, restriction_x0) as block_x0:
        for b_sub, bcs0_sub, x0_sub in zip(block_b, bcs0, block_x0):
            dcpp.fem.set_bc(b_sub, bcs0_sub, x0_sub, scale)
    return b


# -- Matrix assembly ---------------------------------------------------------


class _MatSubMatrixWrapper(object):
    """Wrap a PETSc Mat object."""

    def __init__(  # type: ignore[no-any-unimported]
        self, A: petsc4py.PETSc.Mat, unrestricted_index_sets: typing.Tuple[petsc4py.PETSc.IS, petsc4py.PETSc.IS],
        restricted_index_sets: typing.Optional[typing.Tuple[petsc4py.PETSc.IS, petsc4py.PETSc.IS]] = None,
        unrestricted_to_restricted: typing.Optional[typing.Tuple[typing.Dict[int, int], typing.Dict[int, int]]] = None,
        unrestricted_to_restricted_bs: typing.Optional[typing.Tuple[int, int]] = None
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

    def __enter__(self) -> petsc4py.PETSc.Mat:  # type: ignore[no-any-unimported]
        """Return submatrix content."""
        return self._cpp_object.mat()

    def __exit__(
        self, exception_type: typing.Type[BaseException], exception_value: BaseException,
        traceback: types.TracebackType
    ) -> None:
        """Restore submatrix content."""
        self._cpp_object.restore()


class MatSubMatrixWrapper(object):
    """Wrap a PETSc Mat object."""

    def __init__(  # type: ignore[no-any-unimported]
        self, A: petsc4py.PETSc.Mat, dofmaps: typing.Tuple[dcpp.fem.DofMap, dcpp.fem.DofMap],
        restriction: typing.Optional[typing.Tuple[mcpp.fem.DofMapRestriction, mcpp.fem.DofMapRestriction]] = None
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
        self, exception_type: typing.Type[BaseException], exception_value: BaseException,
        traceback: types.TracebackType
    ) -> None:
        """Restore submatrix content."""
        self._wrapper.__exit__(exception_type, exception_value, traceback)
        self._unrestricted_index_sets[0].destroy()
        self._unrestricted_index_sets[1].destroy()
        if self._restricted_index_sets is not None:
            self._restricted_index_sets[0].destroy()
            self._restricted_index_sets[1].destroy()


class BlockMatSubMatrixWrapper(object):
    """Wrap a PETSc Mat object with several blocks."""

    def __init__(  # type: ignore[no-any-unimported]
        self, A: petsc4py.PETSc.Mat,
        dofmaps: typing.Tuple[typing.List[dcpp.fem.DofMap], typing.List[dcpp.fem.DofMap]],
        restriction: typing.Optional[
            typing.Tuple[typing.List[mcpp.fem.DofMapRestriction], typing.List[mcpp.fem.DofMapRestriction]]] = None
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
            typing.Tuple[int, int, petsc4py.PETSc.Mat]]:
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
        self, exception_type: typing.Type[BaseException], exception_value: BaseException,
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


class NestMatSubMatrixWrapper(object):
    """Wrap a PETSc Mat object with nested blocks."""

    def __init__(  # type: ignore[no-any-unimported]
        self, A: petsc4py.PETSc.Mat, dofmaps: typing.Tuple[typing.List[dcpp.fem.DofMap], typing.List[dcpp.fem.DofMap]],
        restriction: typing.Optional[
            typing.Tuple[typing.List[mcpp.fem.DofMapRestriction], typing.List[mcpp.fem.DofMapRestriction]]] = None
    ) -> None:
        self._A = A
        self._dofmaps = dofmaps
        self._restriction = restriction

    def __iter__(self) -> typing.Iterator[  # type: ignore[no-any-unimported]
            typing.Tuple[int, int, petsc4py.PETSc.Mat]]:
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
        self, exception_type: typing.Type[BaseException], exception_value: BaseException,
        traceback: types.TracebackType
    ) -> None:
        """Do nothing."""
        pass


@functools.singledispatch
def assemble_matrix(  # type: ignore[no-any-unimported]
    a: dolfinx.fem.forms.form_types, bcs: typing.List[dolfinx.fem.DirichletBCMetaClass] = [],
    mat_type: typing.Optional[str] = None, diagonal: float = 1.0,
    constants: typing.Optional[DolfinxConstantsType] = None, coeffs: typing.Optional[DolfinxCoefficientsType] = None,
    restriction: typing.Optional[typing.Tuple[mcpp.fem.DofMapRestriction, mcpp.fem.DofMapRestriction]] = None
) -> petsc4py.PETSc.Mat:
    """
    Assemble bilinear form into a new PETSc matrix.

    Parameters
    ----------
    a
        A bilinear form
    bcs
        Optional list of boundary conditions.
    mat_type
        The PETSc matrix type (``MatType``).
    diagonal
        Optional diagonal value for boundary conditions application. Assumes 1 by default.
    constants
        Constants that appear in the form. If not provided, any required constants will be computed.
    coeffs
        Coefficients that appear in the form. If not provided, any required coefficients will be computed.
    restriction
        A dofmap restriction. If not provided, the unrestricted tensor will be assembled.

    Returns
    -------
    :
        The assembled PETSc matrix.

    Notes
    -----
    The returned matrix is not finalised, i.e. ghost values are not accumulated.
    """
    A = create_matrix(a, restriction, mat_type)
    return assemble_matrix(A, a, bcs, diagonal, constants, coeffs, restriction)  # type: ignore[arg-type]


@assemble_matrix.register
def _(  # type: ignore[no-any-unimported]
    A: petsc4py.PETSc.Mat, a: dolfinx.fem.forms.form_types,
    bcs: typing.List[dolfinx.fem.DirichletBCMetaClass] = [],
    diagonal: float = 1.0,
    constants: typing.Optional[DolfinxConstantsType] = None, coeffs: typing.Optional[DolfinxCoefficientsType] = None,
    restriction: typing.Optional[typing.Tuple[mcpp.fem.DofMapRestriction, mcpp.fem.DofMapRestriction]] = None
) -> petsc4py.PETSc.Mat:
    """
    Assemble bilinear form into an existing PETSc matrix.

    Parameters
    ----------
    A
        PETSc matrix to assemble the contribution of the bilinear forms into.
    a
        A bilinear form to assemble into `A`.
    bcs
        Optional list of boundary conditions.
    diagonal
        Optional diagonal value for boundary conditions application. Assumes 1 by default.
    constants
        Constants that appear in the form. If not provided, any required constants will be computed.
    coeffs
        Coefficients that appear in the form. If not provided, any required coefficients will be computed.
    restriction
        A dofmap restriction. If not provided, the unrestricted tensor will be assembled.

    Returns
    -------
    :
        The assembled PETSc matrix.

    Notes
    -----
    The returned matrix is not finalised, i.e. ghost values are not accumulated.
    """
    constants = dcpp.fem.pack_constants(a) if constants is None else constants
    coeffs = dcpp.fem.pack_coefficients(a) if coeffs is None else coeffs
    function_spaces = a.function_spaces
    if restriction is None:
        # Assemble form
        dcpp.fem.petsc.assemble_matrix(A, a, constants, coeffs, bcs)

        if function_spaces[0] is function_spaces[1]:
            # Flush to enable switch from add to set in the matrix
            A.assemble(petsc4py.PETSc.Mat.AssemblyType.FLUSH)

            # Set diagonal
            dcpp.fem.petsc.insert_diagonal(A, function_spaces[0], bcs, diagonal)
    else:
        dofmaps = (function_spaces[0].dofmap, function_spaces[1].dofmap)

        # Assemble form
        with MatSubMatrixWrapper(A, dofmaps, restriction) as A_sub:
            dcpp.fem.petsc.assemble_matrix(A_sub, a, constants, coeffs, bcs)

        if function_spaces[0] is function_spaces[1]:
            # Flush to enable switch from add to set in the matrix
            A.assemble(petsc4py.PETSc.Mat.AssemblyType.FLUSH)

            # Set diagonal
            with MatSubMatrixWrapper(A, dofmaps, restriction) as A_sub:
                dcpp.fem.petsc.insert_diagonal(A_sub, function_spaces[0], bcs, diagonal)
    return A


@functools.singledispatch
def assemble_matrix_nest(  # type: ignore[no-any-unimported]
    a: typing.List[typing.List[dolfinx.fem.forms.form_types]],
    bcs: typing.List[dolfinx.fem.DirichletBCMetaClass] = [],
    mat_types: typing.List[str] = [], diagonal: float = 1.0,
    constants: typing.Optional[typing.Sequence[typing.Sequence[typing.Optional[DolfinxConstantsType]]]] = None,
    coeffs: typing.Optional[typing.Sequence[typing.Sequence[typing.Optional[DolfinxCoefficientsType]]]] = None,
    restriction: typing.Optional[
        typing.Tuple[typing.List[mcpp.fem.DofMapRestriction], typing.List[mcpp.fem.DofMapRestriction]]] = None
) -> petsc4py.PETSc.Mat:
    """
    Assemble bilinear forms into a new nest PETSc matrix.

    Parameters
    ----------
    a
        A rectangular array of bilinear forms.
    bcs
        Optional list of boundary conditions.
    mat_types
        The PETSc matrix types (``MatType``).
    diagonal
        Optional diagonal value for boundary conditions application. Assumes 1 by default.
    constants
        Constants that appear in the form. If not provided, any required constants will be computed.
    coeffs
        Coefficients that appear in the form. If not provided, any required coefficients will be computed.
    restriction
        A dofmap restriction. If not provided, the unrestricted tensor will be assembled.

    Returns
    -------
    :
        The assembled nest PETSc matrix.
    """
    A = create_matrix_nest(a, restriction, mat_types)
    return assemble_matrix_nest(A, a, bcs, diagonal, constants, coeffs, restriction)  # type: ignore[arg-type]


@assemble_matrix_nest.register
def _(  # type: ignore[no-any-unimported]
    A: petsc4py.PETSc.Mat,
    a: typing.List[typing.List[dolfinx.fem.forms.form_types]],
    bcs: typing.List[dolfinx.fem.DirichletBCMetaClass] = [],
    diagonal: float = 1.0,
    constants: typing.Optional[typing.Sequence[typing.Sequence[typing.Optional[DolfinxConstantsType]]]] = None,
    coeffs: typing.Optional[typing.Sequence[typing.Sequence[typing.Optional[DolfinxCoefficientsType]]]] = None,
    restriction: typing.Optional[
        typing.Tuple[typing.List[mcpp.fem.DofMapRestriction], typing.List[mcpp.fem.DofMapRestriction]]] = None
) -> petsc4py.PETSc.Mat:
    """
    Assemble bilinear forms into an existing nest PETSc matrix.

    Parameters
    ----------
    A
        Nest PETSc matrix to assemble the contribution of the bilinear forms into.
    a
        A rectangular array of bilinear forms to assemble into `A`.
    bcs
        Optional list of boundary conditions.
    diagonal
        Optional diagonal value for boundary conditions application. Assumes 1 by default.
    constants
        Constants that appear in the form. If not provided, any required constants will be computed.
    coeffs
        Coefficients that appear in the form. If not provided, any required coefficients will be computed.
    restriction
        A dofmap restriction. If not provided, the unrestricted tensor will be assembled.

    Returns
    -------
    :
        The assembled nest PETSc matrix.
    """
    function_spaces = _get_block_function_spaces(a)
    dofmaps = (
        [function_space.dofmap for function_space in function_spaces[0]],
        [function_space.dofmap for function_space in function_spaces[1]])

    # Assemble form
    constants = [[
        None if form is None else dcpp.fem.pack_constants(form) for form in forms]
        for forms in a] if constants is None else constants
    coeffs = [[
        {} if form is None else dcpp.fem.pack_coefficients(form) for form in forms]
        for forms in a] if coeffs is None else coeffs
    with NestMatSubMatrixWrapper(A, dofmaps, restriction) as nest_A:
        for i, j, A_sub in nest_A:
            a_sub = a[i][j]
            if a_sub is not None:
                const_sub = constants[i][j]
                coeff_sub = coeffs[i][j]
                dcpp.fem.petsc.assemble_matrix(A_sub, a_sub, const_sub, coeff_sub, bcs)
            elif i == j:  # pragma: no cover
                for bc in bcs:
                    if function_spaces[0][i].contains(bc.function_space):
                        raise RuntimeError(
                            f"Diagonal sub-block ({i}, {j}) cannot be 'None' and have DirichletBC applied."
                            " Consider assembling a zero block.")

    # Flush to enable switch from add to set in the matrix
    A.assemble(petsc4py.PETSc.Mat.AssemblyType.FLUSH)

    # Set diagonal
    with NestMatSubMatrixWrapper(A, dofmaps, restriction) as nest_A:
        for i, j, A_sub in nest_A:
            if function_spaces[0][i] is function_spaces[1][j]:
                a_sub = a[i][j]
                if a_sub is not None:
                    dcpp.fem.petsc.insert_diagonal(A_sub, function_spaces[0][i], bcs, diagonal)

    return A


@functools.singledispatch
def assemble_matrix_block(  # type: ignore[no-any-unimported]
    a: typing.List[typing.List[dolfinx.fem.forms.form_types]],
    bcs: typing.List[dolfinx.fem.DirichletBCMetaClass] = [],
    mat_type: typing.Optional[str] = None, diagonal: float = 1.0,
    constants: typing.Optional[typing.Sequence[typing.Sequence[typing.Optional[DolfinxConstantsType]]]] = None,
    coeffs: typing.Optional[typing.Sequence[typing.Sequence[typing.Optional[DolfinxCoefficientsType]]]] = None,
    restriction: typing.Optional[
        typing.Tuple[typing.List[mcpp.fem.DofMapRestriction], typing.List[mcpp.fem.DofMapRestriction]]] = None
) -> petsc4py.PETSc.Mat:
    """
    Assemble bilinear forms into a new block PETSc matrix.

    Parameters
    ----------
    a
        A rectangular array of bilinear forms.
    bcs
        Optional list of boundary conditions.
    mat_type
        The PETSc matrix type (``MatType``).
    diagonal
        Optional diagonal value for boundary conditions application. Assumes 1 by default.
    constants
        Constants that appear in the form. If not provided, any required constants will be computed.
    coeffs
        Coefficients that appear in the form. If not provided, any required coefficients will be computed.
    restriction
        A dofmap restriction. If not provided, the unrestricted tensor will be assembled.

    Returns
    -------
    :
        The assembled block PETSc matrix.
    """
    A = create_matrix_block(a, restriction, mat_type)
    return assemble_matrix_block(A, a, bcs, diagonal, constants, coeffs, restriction)  # type: ignore[arg-type]


@assemble_matrix_block.register
def _(  # type: ignore[no-any-unimported]
    A: petsc4py.PETSc.Mat,
    a: typing.List[typing.List[dolfinx.fem.forms.form_types]],
    bcs: typing.List[dolfinx.fem.DirichletBCMetaClass] = [],
    diagonal: float = 1.0,
    constants: typing.Optional[typing.Sequence[typing.Sequence[typing.Optional[DolfinxConstantsType]]]] = None,
    coeffs: typing.Optional[typing.Sequence[typing.Sequence[typing.Optional[DolfinxCoefficientsType]]]] = None,
    restriction: typing.Optional[
        typing.Tuple[typing.List[mcpp.fem.DofMapRestriction], typing.List[mcpp.fem.DofMapRestriction]]] = None
) -> petsc4py.PETSc.Mat:
    """
    Assemble bilinear forms into an existing block PETSc matrix.

    Parameters
    ----------
    A
        Block PETSc matrix to assemble the contribution of the bilinear forms into.
    a
        A rectangular array of bilinear forms to assemble into `A`.
    bcs
        Optional list of boundary conditions.
    diagonal
        Optional diagonal value for boundary conditions application. Assumes 1 by default.
    constants
        Constants that appear in the form. If not provided, any required constants will be computed.
    coeffs
        Coefficients that appear in the form. If not provided, any required coefficients will be computed.
    restriction
        A dofmap restriction. If not provided, the unrestricted tensor will be assembled.

    Returns
    -------
    :
        The assembled block PETSc matrix.
    """
    constants = [[
        None if form is None else dcpp.fem.pack_constants(form) for form in forms]
        for forms in a] if constants is None else constants
    coeffs = [[
        {} if form is None else dcpp.fem.pack_coefficients(form) for form in forms]
        for forms in a] if coeffs is None else coeffs
    function_spaces = _get_block_function_spaces(a)
    dofmaps = (
        [function_space.dofmap for function_space in function_spaces[0]],
        [function_space.dofmap for function_space in function_spaces[1]])

    # Assemble form
    with BlockMatSubMatrixWrapper(A, dofmaps, restriction) as block_A:
        for i, j, A_sub in block_A:
            a_sub = a[i][j]
            if a_sub is not None:
                const_sub = constants[i][j]
                coeff_sub = coeffs[i][j]
                dcpp.fem.petsc.assemble_matrix(A_sub, a_sub, const_sub, coeff_sub, bcs, True)
            elif i == j:  # pragma: no cover
                for bc in bcs:
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
                    dcpp.fem.petsc.insert_diagonal(A_sub, function_spaces[0][i], bcs, diagonal)

    return A


# -- Modifiers for Dirichlet conditions ---------------------------------------

def apply_lifting(  # type: ignore[no-any-unimported]
    b: petsc4py.PETSc.Vec, a: typing.List[dolfinx.fem.forms.form_types],
    bcs: typing.List[typing.List[dolfinx.fem.DirichletBCMetaClass]] = [],
    x0: typing.Optional[typing.List[petsc4py.PETSc.Vec]] = None,
    scale: float = 1.0,
    constants: typing.Optional[typing.Sequence[typing.Optional[DolfinxConstantsType]]] = None,
    coeffs: typing.Optional[typing.Sequence[typing.Optional[DolfinxCoefficientsType]]] = None,
    restriction: typing.Optional[mcpp.fem.DofMapRestriction] = None,
    restriction_x0: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None
) -> None:
    """Apply the function :func:`dolfinx.fem.apply_lifting` to a PETSc Vector."""
    function_spaces = [form.function_spaces[1] for form in a]
    dofmaps_x0 = [function_space.dofmap for function_space in function_spaces]
    if restriction is None:
        with b.localForm() as b_local, NestVecSubVectorReadWrapper(x0, dofmaps_x0) as nest_x0:
            if x0 is None:
                x0_as_list = []
            else:
                x0_as_list = [x0_ for x0_ in nest_x0]
            dolfinx.fem.assemble.apply_lifting(
                b_local.array_w, a, bcs, x0_as_list, scale, constants, coeffs)
    else:
        with VecSubVectorWrapper(b, restriction.dofmap, restriction) as b_sub, \
                NestVecSubVectorReadWrapper(x0, dofmaps_x0, restriction_x0) as nest_x0:
            for a_sub, bcs_sub, x0_sub in zip(a, bcs, nest_x0):
                if x0_sub is None:
                    x0_sub_as_list = []
                else:
                    x0_sub_as_list = [x0_sub]
                dolfinx.fem.assemble.apply_lifting(
                    b_sub, [a_sub], [bcs_sub], x0_sub_as_list, scale, constants, coeffs)


def apply_lifting_nest(  # type: ignore[no-any-unimported]
    b: petsc4py.PETSc.Vec, a: typing.List[typing.List[dolfinx.fem.forms.form_types]],
    bcs: typing.List[dolfinx.fem.DirichletBCMetaClass] = [],
    x0: typing.Optional[petsc4py.PETSc.Vec] = None,
    scale: float = 1.0,
    constants: typing.Optional[typing.Sequence[typing.Sequence[typing.Optional[DolfinxConstantsType]]]] = None,
    coeffs: typing.Optional[typing.Sequence[typing.Sequence[typing.Optional[DolfinxCoefficientsType]]]] = None,
    restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None,
    restriction_x0: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None
) -> petsc4py.PETSc.Vec:
    """Apply the function :func:`dolfinx.fem.apply_lifting` to each sub-vector in a nested PETSc Vector."""
    constants = [[
        None if form is None else dcpp.fem.pack_constants(form) for form in forms]
        for forms in a] if constants is None else constants
    coeffs = [[
        {} if form is None else dcpp.fem.pack_coefficients(form) for form in forms]
        for forms in a] if coeffs is None else coeffs
    function_spaces = _get_block_function_spaces(a)
    dofmaps = [function_space.dofmap for function_space in function_spaces[0]]
    dofmaps_x0 = [function_space.dofmap for function_space in function_spaces[1]]
    bcs1 = dolfinx.fem.bcs_by_block(function_spaces[1], bcs)
    with NestVecSubVectorWrapper(b, dofmaps, restriction) as nest_b, \
            NestVecSubVectorReadWrapper(x0, dofmaps_x0, restriction_x0) as nest_x0:
        for b_sub, a_sub, constants_a, coeffs_a in zip(nest_b, a, constants, coeffs):
            for a_sub_, bcs1_sub, x0_sub, constant_a_sub, coeff_a_sub in zip(
                    a_sub, bcs1, nest_x0, constants_a, coeffs_a):
                if x0_sub is None:
                    x0_sub_as_list = []
                else:
                    x0_sub_as_list = [x0_sub]
                dolfinx.fem.assemble.apply_lifting(
                    b_sub, [a_sub_], [bcs1_sub], x0_sub_as_list, scale, [constant_a_sub], [coeff_a_sub])
    return b


def set_bc(  # type: ignore[no-any-unimported]
    b: petsc4py.PETSc.Vec, bcs: typing.List[dolfinx.fem.DirichletBCMetaClass] = [],
    x0: typing.Optional[petsc4py.PETSc.Vec] = None,
    scale: float = 1.0,
    restriction: typing.Optional[mcpp.fem.DofMapRestriction] = None
) -> None:
    """Apply the function :func:`dolfinx.fem.set_bc` to a PETSc Vector."""
    if restriction is None:
        if x0 is not None:
            x0 = x0.array_r
        dolfinx.fem.assemble.set_bc(b.array_w, bcs, x0, scale)
    else:
        with VecSubVectorWrapper(b, restriction.dofmap, restriction, ghosted=False) as b_sub, \
                VecSubVectorReadWrapper(x0, restriction.dofmap, restriction, ghosted=False) as x0_sub:
            dolfinx.fem.assemble.set_bc(b_sub, bcs, x0_sub, scale)


def set_bc_nest(  # type: ignore[no-any-unimported]
    b: petsc4py.PETSc.Vec, bcs: typing.List[typing.List[dolfinx.fem.DirichletBCMetaClass]] = [],
    x0: typing.Optional[petsc4py.PETSc.Vec] = None,
    scale: float = 1.0,
    restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None
) -> None:
    """Apply the function :func:`dolfinx.fem.set_bc` to each sub-vector of a nested PETSc Vector."""
    if restriction is None:
        dofmaps = [None] * len(b.getNestSubVecs())
    else:
        dofmaps = [restriction_.dofmap for restriction_ in restriction]
    dofmaps_x0 = dofmaps
    restriction_x0 = restriction
    with NestVecSubVectorWrapper(b, dofmaps, restriction, ghosted=False) as nest_b, \
            NestVecSubVectorReadWrapper(x0, dofmaps_x0, restriction_x0, ghosted=False) as nest_x0:
        for b_sub, bcs_sub, x0_sub in zip(nest_b, bcs, nest_x0):
            dolfinx.fem.assemble.set_bc(b_sub, bcs_sub, x0_sub, scale)

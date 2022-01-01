# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Assembly functions for variational forms."""

import functools
import typing
from contextlib import ExitStack

import numpy as np
from petsc4py import PETSc

import dolfinx.cpp as dcpp
from dolfinx.fem.dirichletbc import bcs_by_block, DirichletBC
from dolfinx.fem.form import Form
from dolfinx.fem.assemble import Coefficients, _cpp_dirichletbc, _create_cpp_form, pack_coefficients, pack_constants

from multiphenicsx.cpp import cpp as mcpp


def _get_block_function_spaces(block_form):
    cpp_Form = dcpp.fem.Form_float64 if PETSc.ScalarType == np.float64 else dcpp.fem.Form_complex128
    assert all(isinstance(block_form_, (cpp_Form, list)) for block_form_ in block_form)
    if isinstance(block_form[0], list):
        assert all(isinstance(form, cpp_Form) or form is None for block_form_ in block_form
                   for form in block_form_)
        _a = block_form
        rows = len(_a)
        cols = len(_a[0])
        assert all(len(a_i) == cols for a_i in _a)
        assert all(_a[i][j] is None or _a[i][j].rank == 2 for i in range(rows) for j in range(cols))
        function_spaces_0 = list()
        for i in range(rows):
            function_spaces_0_i = None
            for j in range(cols):
                if _a[i][j] is not None:
                    function_spaces_0_i = _a[i][j].function_spaces[0]
                    break
            assert function_spaces_0_i is not None
            function_spaces_0.append(function_spaces_0_i)
        function_spaces_1 = list()
        for j in range(cols):
            function_spaces_1_j = None
            for i in range(rows):
                if _a[i][j] is not None:
                    function_spaces_1_j = _a[i][j].function_spaces[1]
                    break
            assert function_spaces_1_j is not None
            function_spaces_1.append(function_spaces_1_j)
        function_spaces = [function_spaces_0, function_spaces_1]
        assert all(_a[i][j] is None or _a[i][j].function_spaces[0] == function_spaces[0][i]
                   for i in range(rows) for j in range(cols))
        assert all(_a[i][j] is None or _a[i][j].function_spaces[1] == function_spaces[1][j]
                   for i in range(rows) for j in range(cols))
        return function_spaces
    else:
        return [form.function_spaces[0] for form in block_form]


def _same_dofmap(dofmap1, dofmap2):
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

def create_vector(L: Form,
                  restriction: typing.Optional[mcpp.fem.DofMapRestriction] = None) -> PETSc.Vec:
    dofmap = _create_cpp_form(L).function_spaces[0].dofmap
    if restriction is None:
        index_map = dofmap.index_map
        index_map_bs = dofmap.index_map_bs
    else:
        assert _same_dofmap(restriction.dofmap, dofmap)
        index_map = restriction.index_map
        index_map_bs = restriction.index_map_bs
    return dcpp.la.petsc.create_vector(index_map, index_map_bs)


def create_vector_block(L: typing.List[Form],
                        restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None) -> PETSc.Vec:
    function_spaces = _get_block_function_spaces(_create_cpp_form(L))
    dofmaps = [function_space.dofmap for function_space in function_spaces]
    if restriction is None:
        index_maps = [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps]
    else:
        assert len(restriction) == len(dofmaps)
        assert all(_same_dofmap(restriction_.dofmap, dofmap) for (restriction_, dofmap) in zip(restriction, dofmaps))
        index_maps = [(restriction_.index_map, restriction_.index_map_bs) for restriction_ in restriction]
    return dcpp.fem.petsc.create_vector_block(index_maps)


def create_vector_nest(L: typing.List[Form],
                       restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None) -> PETSc.Vec:
    function_spaces = _get_block_function_spaces(_create_cpp_form(L))
    dofmaps = [function_space.dofmap for function_space in function_spaces]
    if restriction is None:
        index_maps = [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps]
    else:
        assert len(restriction) == len(dofmaps)
        assert all(_same_dofmap(restriction_.dofmap, dofmap) for (restriction_, dofmap) in zip(restriction, dofmaps))
        index_maps = [(restriction_.index_map, restriction_.index_map_bs) for restriction_ in restriction]
    return dcpp.fem.petsc.create_vector_nest(index_maps)


# -- Matrix instantiation ----------------------------------------------------

def create_matrix(a: Form,
                  restriction: typing.Optional[typing.Tuple[mcpp.fem.DofMapRestriction]] = None,
                  mat_type=None) -> PETSc.Mat:
    _a = _create_cpp_form(a)
    assert _a.rank == 2
    mesh = _a.mesh
    function_spaces = _a.function_spaces
    assert all(function_space.mesh == mesh for function_space in function_spaces)
    if restriction is None:
        index_maps = [function_space.dofmap.index_map for function_space in function_spaces]
        index_maps_bs = [function_space.dofmap.index_map_bs for function_space in function_spaces]
    else:
        assert len(restriction) == 2
        index_maps = [restriction_.index_map for restriction_ in restriction]
        index_maps_bs = [restriction_.index_map_bs for restriction_ in restriction]
    integral_types = mcpp.fem.get_integral_types_from_form(_a)
    integral_types = list(integral_types)  # TODO Remove this when pybind11#2122 is fixed.
    if restriction is None:
        dofmaps_lists = [function_space.dofmap.list() for function_space in function_spaces]
    else:
        dofmaps_lists = [restriction_.list() for restriction_ in restriction]
    if mat_type is not None:
        return mcpp.fem.petsc.create_matrix(mesh, index_maps, index_maps_bs, integral_types, dofmaps_lists, mat_type)
    else:
        return mcpp.fem.petsc.create_matrix(mesh, index_maps, index_maps_bs, integral_types, dofmaps_lists)


def _create_matrix_block_or_nest(a, restriction, mat_type, cpp_create_function):
    _a = _create_cpp_form(a)
    function_spaces = _get_block_function_spaces(_a)
    rows, cols = len(function_spaces[0]), len(function_spaces[1])
    mesh = None
    for j in range(cols):
        for i in range(rows):
            if _a[i][j] is not None:
                mesh = _a[i][j].mesh
    assert mesh is not None
    assert all(_a[i][j] is None or _a[i][j].mesh == mesh for i in range(rows) for j in range(cols))
    assert all(function_space.mesh == mesh for function_space in function_spaces[0])
    assert all(function_space.mesh == mesh for function_space in function_spaces[1])
    if restriction is None:
        index_maps = ([function_spaces[0][i].dofmap.index_map for i in range(rows)],
                      [function_spaces[1][j].dofmap.index_map for j in range(cols)])
        index_maps_bs = ([function_spaces[0][i].dofmap.index_map_bs for i in range(rows)],
                         [function_spaces[1][j].dofmap.index_map_bs for j in range(cols)])
    else:
        assert len(restriction) == 2
        assert len(restriction[0]) == rows
        assert len(restriction[1]) == cols
        index_maps = ([restriction[0][i].index_map for i in range(rows)],
                      [restriction[1][j].index_map for j in range(cols)])
        index_maps_bs = ([restriction[0][i].index_map_bs for i in range(rows)],
                         [restriction[1][j].index_map_bs for j in range(cols)])
    integral_types = [[set() for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if _a[i][j] is not None:
                integral_types[i][j].update(mcpp.fem.get_integral_types_from_form(_a[i][j]))
    integral_types = [[list(integral_types[row][col]) for col in range(cols)]
                      for row in range(rows)]  # TODO Remove this when pybind11#2122 is fixed.
    if restriction is None:
        dofmaps_lists = ([function_spaces[0][i].dofmap.list() for i in range(rows)],
                         [function_spaces[1][j].dofmap.list() for j in range(cols)])
    else:
        dofmaps_lists = ([restriction[0][i].list() for i in range(rows)],
                         [restriction[1][j].list() for j in range(cols)])
    if mat_type is not None:
        return cpp_create_function(mesh, index_maps, index_maps_bs, integral_types, dofmaps_lists, mat_type)
    else:
        return cpp_create_function(mesh, index_maps, index_maps_bs, integral_types, dofmaps_lists)


def create_matrix_block(a: typing.List[typing.List[Form]],
                        restriction: typing.Optional[typing.Tuple[
                                                     typing.List[mcpp.fem.DofMapRestriction]]] = None,
                        mat_type=None) -> PETSc.Mat:
    return _create_matrix_block_or_nest(a, restriction, mat_type, mcpp.fem.petsc.create_matrix_block)


def create_matrix_nest(a: typing.List[typing.List[Form]],
                       restriction: typing.Optional[typing.Tuple[
                                                    typing.List[mcpp.fem.DofMapRestriction]]] = None,
                       mat_types=None) -> PETSc.Mat:
    return _create_matrix_block_or_nest(a, restriction, mat_types, mcpp.fem.petsc.create_matrix_nest)


# -- Scalar assembly ---------------------------------------------------------


def assemble_scalar(M: Form, coeffs=Coefficients(None, None)) -> PETSc.ScalarType:
    """Assemble functional. The returned value is local and not
    accumulated across processes.

    """
    _M = _create_cpp_form(M)
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(_M),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(_M))
    return dcpp.fem.assemble_scalar(_M, c[0], c[1])


# -- Vector assembly ---------------------------------------------------------

def _VecSubVectorWrapperBase(CppWrapperClass):

    class _VecSubVectorWrapperBase_Class(object):
        def __init__(self,
                     b: PETSc.Vec, unrestricted_index_set: PETSc.IS,
                     restricted_index_set: typing.Optional[PETSc.IS] = None,
                     unrestricted_to_restricted: typing.Optional[typing.Dict[int, int]] = None,
                     unrestricted_to_restricted_bs: typing.Optional[int] = None):
            if restricted_index_set is None:
                assert unrestricted_to_restricted is None
                assert unrestricted_to_restricted_bs is None
                self._cpp_object = CppWrapperClass(b, unrestricted_index_set)
            else:
                self._cpp_object = CppWrapperClass(b, unrestricted_index_set, restricted_index_set,
                                                   unrestricted_to_restricted, unrestricted_to_restricted_bs)

        def __enter__(self):
            return self._cpp_object.content

        def __exit__(self, exception_type, exception_value, traceback):
            pass

    return _VecSubVectorWrapperBase_Class


_VecSubVectorReadWrapper = _VecSubVectorWrapperBase(mcpp.la.petsc.VecSubVectorReadWrapper)


class _VecSubVectorWrapper(_VecSubVectorWrapperBase(mcpp.la.petsc.VecSubVectorWrapper)):
    def __exit__(self, exception_type, exception_value, traceback):
        self._cpp_object.restore()


def VecSubVectorWrapperBase(_VecSubVectorWrapperClass):

    class VecSubVectorWrapperBase_Class(object):
        def __init__(self,
                     b: typing.Union[PETSc.Vec, None], dofmap: dcpp.fem.DofMap,
                     restriction: typing.Optional[mcpp.fem.DofMapRestriction] = None,
                     ghosted: bool = True):
            if b is None:
                self._wrapper = None
            else:
                if restriction is None:
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
                    self._wrapper = _VecSubVectorWrapperClass(b, unrestricted_index_set, restricted_index_set,
                                                              unrestricted_to_restricted, unrestricted_to_restricted_bs)
                    self._unrestricted_index_set = unrestricted_index_set
                    self._restricted_index_set = restricted_index_set
                    self._unrestricted_to_restricted = unrestricted_to_restricted
                    self._unrestricted_to_restricted_bs = unrestricted_to_restricted_bs

        def __enter__(self):
            if self._wrapper is not None:
                return self._wrapper.__enter__()

        def __exit__(self, exception_type, exception_value, traceback):
            if self._wrapper is not None:
                self._wrapper.__exit__(exception_type, exception_value, traceback)
                self._unrestricted_index_set.destroy()
                if self._restricted_index_set is not None:
                    self._restricted_index_set.destroy()

    return VecSubVectorWrapperBase_Class


VecSubVectorReadWrapper = VecSubVectorWrapperBase(_VecSubVectorReadWrapper)


VecSubVectorWrapper = VecSubVectorWrapperBase(_VecSubVectorWrapper)


def BlockVecSubVectorWrapperBase(_VecSubVectorWrapperClass):
    class BlockVecSubVectorWrapperBase_Class(object):
        def __init__(self,
                     b: typing.Union[PETSc.Vec, None], dofmaps: typing.List[dcpp.fem.DofMap],
                     restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None,
                     ghosted: bool = True):
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
                    assert all([_same_dofmap(dofmap, restriction_.dofmap)
                                for (dofmap, restriction_) in zip(dofmaps, restriction)])
                    unrestricted_index_maps = [(dofmap.index_map, dofmap.index_map_bs)
                                               for dofmap in dofmaps]
                    unrestricted_index_sets = mcpp.la.petsc.create_index_sets(
                        unrestricted_index_maps, [1] * len(unrestricted_index_maps),
                        ghost_block_layout=mcpp.la.petsc.GhostBlockLayout.trailing)
                    restricted_index_maps = [(restriction_.index_map, restriction_.index_map_bs)
                                             for restriction_ in restriction]
                    restricted_index_sets = mcpp.la.petsc.create_index_sets(
                        restricted_index_maps, [1] * len(restricted_index_maps),
                        ghosted=ghosted, ghost_block_layout=mcpp.la.petsc.GhostBlockLayout.trailing)
                    unrestricted_to_restricted = [restriction_.unrestricted_to_restricted
                                                  for restriction_ in restriction]
                    unrestricted_to_restricted_bs = [restriction_.index_map_bs
                                                     for restriction_ in restriction]
                    self._unrestricted_index_sets = unrestricted_index_sets
                    self._restricted_index_sets = restricted_index_sets
                    self._unrestricted_to_restricted = unrestricted_to_restricted
                    self._unrestricted_to_restricted_bs = unrestricted_to_restricted_bs

        def __iter__(self):
            with ExitStack() as wrapper_stack:
                for index in range(self._len):
                    if self._b is None:
                        yield None
                    else:
                        if self._restricted_index_sets is None:
                            wrapper = _VecSubVectorWrapperClass(self._b,
                                                                self._unrestricted_index_sets[index])
                        else:
                            wrapper = _VecSubVectorWrapperClass(self._b,
                                                                self._unrestricted_index_sets[index],
                                                                self._restricted_index_sets[index],
                                                                self._unrestricted_to_restricted[index],
                                                                self._unrestricted_to_restricted_bs[index])
                        yield wrapper_stack.enter_context(wrapper)

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception_value, traceback):
            if self._b is not None:
                for index_set in self._unrestricted_index_sets:
                    index_set.destroy()
                if self._restricted_index_sets is not None:
                    for index_set in self._restricted_index_sets:
                        index_set.destroy()

    return BlockVecSubVectorWrapperBase_Class


BlockVecSubVectorReadWrapper = BlockVecSubVectorWrapperBase(_VecSubVectorReadWrapper)


BlockVecSubVectorWrapper = BlockVecSubVectorWrapperBase(_VecSubVectorWrapper)


def NestVecSubVectorWrapperBase(VecSubVectorWrapperClass):
    class NestVecSubVectorWrapperBase_Class(object):
        def __init__(self,
                     b: typing.Union[PETSc.Vec, typing.List[PETSc.Vec], None],
                     dofmaps: typing.List[dcpp.fem.DofMap],
                     restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None,
                     ghosted: bool = True):
            if b is not None:
                if isinstance(b, list):
                    self._b = b
                else:
                    self._b = b.getNestSubVecs()
                assert len(self._b) == len(dofmaps)
            else:
                self._b = [None] * len(dofmaps)
            self._dofmaps = dofmaps
            self._restriction = restriction
            self._ghosted = ghosted

        def __iter__(self):
            with ExitStack() as wrapper_stack:
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
                            wrapper = VecSubVectorWrapperClass(b_index,
                                                               self._dofmaps[index],
                                                               self._restriction[index],
                                                               ghosted=self._ghosted)
                            yield wrapper_stack.enter_context(wrapper)

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception_value, traceback):
            pass

    return NestVecSubVectorWrapperBase_Class


NestVecSubVectorReadWrapper = NestVecSubVectorWrapperBase(VecSubVectorReadWrapper)


NestVecSubVectorWrapper = NestVecSubVectorWrapperBase(VecSubVectorWrapper)


@functools.singledispatch
def assemble_vector(L: Form,
                    coeffs=Coefficients(None, None),
                    restriction: typing.Optional[mcpp.fem.DofMapRestriction] = None) -> PETSc.Vec:
    """Assemble linear form into a new PETSc vector. The returned vector is
    not finalised, i.e. ghost values are not accumulated on the owning
    processes.

    """
    _L = _create_cpp_form(L)
    b = create_vector(_L, restriction)
    with b.localForm() as b_local:
        b_local.set(0.0)
    return assemble_vector(b, _L, coeffs, restriction)


@assemble_vector.register(PETSc.Vec)
def _(b: PETSc.Vec,
      L: Form,
      coeffs=Coefficients(None, None),
      restriction: typing.Optional[mcpp.fem.DofMapRestriction] = None) -> PETSc.Vec:
    """Assemble linear form into an existing PETSc vector. The vector is
    not zeroed before assembly and it is not finalised, i.e. ghost
    values are not accumulated on the owning processes.

    """
    _L = _create_cpp_form(L)
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(_L),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(_L))
    if restriction is None:
        with b.localForm() as b_local:
            dcpp.fem.assemble_vector(b_local.array_w, _L, c[0], c[1])
    else:
        with VecSubVectorWrapper(b, _L.function_spaces[0].dofmap, restriction) as b_sub:
            dcpp.fem.assemble_vector(b_sub, _L, c[0], c[1])
    return b


@functools.singledispatch
def assemble_vector_nest(L: Form,
                         coeffs=Coefficients(None, None),
                         restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None) -> PETSc.Vec:
    """Assemble linear forms into a new nested PETSc (VecNest) vector.
    The returned vector is not finalised, i.e. ghost values are not
    accumulated on the owning processes.

    """
    _L = _create_cpp_form(L)
    b = create_vector_nest(_L, restriction)
    for b_sub in b.getNestSubVecs():
        with b_sub.localForm() as b_local:
            b_local.set(0.0)
    return assemble_vector_nest(b, L, coeffs, restriction)


@assemble_vector_nest.register(PETSc.Vec)
def _(b: PETSc.Vec,
      L: typing.List[Form],
      coeffs=Coefficients(None, None),
      restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None) -> PETSc.Vec:
    """Assemble linear forms into a nested PETSc (VecNest) vector. The
    vector is not zeroed before assembly and it is not finalised, i.e.
    ghost values are not accumulated on the owning processes.

    """
    _L = _create_cpp_form(L)
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(_L),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(_L))
    function_spaces = _get_block_function_spaces(_L)
    dofmaps = [function_space.dofmap for function_space in function_spaces]
    with NestVecSubVectorWrapper(b, dofmaps, restriction) as nest_b:
        for b_sub, L_sub, constant, coeff in zip(nest_b, _L, c[0], c[1]):
            dcpp.fem.assemble_vector(b_sub, L_sub, constant, coeff)
    return b


# FIXME: Revise this interface
@functools.singledispatch
def assemble_vector_block(L: typing.List[Form],
                          a: typing.List[typing.List[Form]],
                          bcs: typing.List[DirichletBC] = [],
                          x0: typing.Optional[PETSc.Vec] = None,
                          scale: float = 1.0,
                          coeffs_L=Coefficients(None, None),
                          coeffs_a=Coefficients(None, None),
                          restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None,
                          restriction_x0: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector. The vector is not
    finalised, i.e. ghost values are not accumulated.

    """
    _L = _create_cpp_form(L)
    b = create_vector_block(_L, restriction)
    with b.localForm() as b_local:
        b_local.set(0.0)
    return assemble_vector_block(b, _L, a, bcs, x0, scale, coeffs_L, coeffs_a, restriction, restriction_x0)


@assemble_vector_block.register(PETSc.Vec)
def _(b: PETSc.Vec,
      L: typing.List[Form],
      a: typing.List[typing.List[Form]],
      bcs: typing.List[DirichletBC] = [],
      x0: typing.Optional[PETSc.Vec] = None,
      scale: float = 1.0,
      coeffs_L=Coefficients(None, None),
      coeffs_a=Coefficients(None, None),
      restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None,
      restriction_x0: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector. The vector is not
    zeroed and it is not finalised, i.e. ghost values are not
    accumulated.

    """
    _L = _create_cpp_form(L)
    _a = _create_cpp_form(a)
    c_L = (coeffs_L[0] if coeffs_L[0] is not None else pack_constants(_L),
           coeffs_L[1] if coeffs_L[1] is not None else pack_coefficients(_L))
    c_a = (coeffs_a[0] if coeffs_a[0] is not None else pack_constants(_a),
           coeffs_a[1] if coeffs_a[1] is not None else pack_coefficients(_a))
    function_spaces = _get_block_function_spaces(_a)
    dofmaps = [function_space.dofmap for function_space in function_spaces[0]]
    dofmaps_x0 = [function_space.dofmap for function_space in function_spaces[1]]

    bcs1 = bcs_by_block(function_spaces[1], bcs)
    with BlockVecSubVectorWrapper(b, dofmaps, restriction) as block_b, \
            BlockVecSubVectorReadWrapper(x0, dofmaps_x0, restriction_x0) as block_x0:
        for b_sub, L_sub, a_sub, constant_L, coeff_L, constant_a, coeff_a in zip(
                block_b, _L, _a, c_L[0], c_L[1], c_a[0], c_a[1]):
            dcpp.fem.assemble_vector(b_sub, L_sub, constant_L, coeff_L)
            for a_sub_, bcs1_sub, x0_sub, constant_a_sub, coeff_a_sub in zip(
                    a_sub, bcs1, block_x0, constant_a, coeff_a):
                if x0_sub is None:
                    x0_sub_as_list = []
                else:
                    x0_sub_as_list = [x0_sub]
                dcpp.fem.apply_lifting(
                    b_sub, [a_sub_], [constant_a_sub], [coeff_a_sub], _cpp_dirichletbc([bcs1_sub]),
                    x0_sub_as_list, scale)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    bcs0 = bcs_by_block(function_spaces[0], bcs)
    with BlockVecSubVectorWrapper(b, dofmaps, restriction) as block_b, \
            BlockVecSubVectorReadWrapper(x0, dofmaps_x0, restriction_x0) as block_x0:
        for b_sub, bcs0_sub, x0_sub in zip(block_b, bcs0, block_x0):
            dcpp.fem.set_bc(b_sub, _cpp_dirichletbc(bcs0_sub), x0_sub, scale)
    return b


# -- Matrix assembly ---------------------------------------------------------


class _MatSubMatrixWrapper(object):
    def __init__(self,
                 A: PETSc.Mat, unrestricted_index_sets: typing.Tuple[PETSc.IS],
                 restricted_index_sets: typing.Optional[typing.Tuple[PETSc.IS]] = None,
                 unrestricted_to_restricted: typing.Optional[typing.Tuple[typing.Dict[int, int]]] = None,
                 unrestricted_to_restricted_bs: typing.Optional[typing.Tuple[int]] = None):
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

    def __enter__(self):
        return self._cpp_object.mat()

    def __exit__(self, exception_type, exception_value, traceback):
        self._cpp_object.restore()


class MatSubMatrixWrapper(object):
    def __init__(self,
                 A: PETSc.Mat, dofmaps: typing.Tuple[dcpp.fem.DofMap],
                 restriction: typing.Optional[typing.Tuple[mcpp.fem.DofMapRestriction]] = None):
        assert len(dofmaps) == 2
        if restriction is None:
            index_maps = ((dofmaps[0].index_map, dofmaps[0].index_map_bs),
                          (dofmaps[1].index_map, dofmaps[1].index_map_bs))
            index_sets = (mcpp.la.petsc.create_index_sets([index_maps[0]], [dofmaps[0].index_map_bs])[0],
                          mcpp.la.petsc.create_index_sets([index_maps[1]], [dofmaps[1].index_map_bs])[0])
            self._wrapper = _MatSubMatrixWrapper(A, index_sets)
            self._unrestricted_index_sets = index_sets
            self._restricted_index_sets = None
            self._unrestricted_to_restricted = None
            self._unrestricted_to_restricted_bs = None
        else:
            assert len(restriction) == 2
            assert all([_same_dofmap(dofmaps[i], restriction[i].dofmap) for i in range(2)])
            unrestricted_index_maps = ((dofmaps[0].index_map, dofmaps[0].index_map_bs),
                                       (dofmaps[1].index_map, dofmaps[1].index_map_bs))
            unrestricted_index_sets = (mcpp.la.petsc.create_index_sets([unrestricted_index_maps[0]],
                                                                       [dofmaps[0].index_map_bs])[0],
                                       mcpp.la.petsc.create_index_sets([unrestricted_index_maps[1]],
                                                                       [dofmaps[1].index_map_bs])[0])
            restricted_index_maps = ((restriction[0].index_map, restriction[0].index_map_bs),
                                     (restriction[1].index_map, restriction[1].index_map_bs))
            restricted_index_sets = (mcpp.la.petsc.create_index_sets([restricted_index_maps[0]],
                                                                     [restriction[0].index_map_bs])[0],
                                     mcpp.la.petsc.create_index_sets([restricted_index_maps[1]],
                                                                     [restriction[1].index_map_bs])[0])
            unrestricted_to_restricted = (restriction[0].unrestricted_to_restricted,
                                          restriction[1].unrestricted_to_restricted)
            unrestricted_to_restricted_bs = (restriction[0].index_map_bs,
                                             restriction[1].index_map_bs)
            self._wrapper = _MatSubMatrixWrapper(A, unrestricted_index_sets,
                                                 restricted_index_sets, unrestricted_to_restricted,
                                                 unrestricted_to_restricted_bs)
            self._unrestricted_index_sets = unrestricted_index_sets
            self._restricted_index_sets = restricted_index_sets
            self._unrestricted_to_restricted = unrestricted_to_restricted
            self._unrestricted_to_restricted_bs = unrestricted_to_restricted_bs

    def __enter__(self):
        return self._wrapper.__enter__()

    def __exit__(self, exception_type, exception_value, traceback):
        self._wrapper.__exit__(exception_type, exception_value, traceback)
        self._unrestricted_index_sets[0].destroy()
        self._unrestricted_index_sets[1].destroy()
        if self._restricted_index_sets is not None:
            self._restricted_index_sets[0].destroy()
            self._restricted_index_sets[1].destroy()


class BlockMatSubMatrixWrapper(object):
    def __init__(self,
                 A: PETSc.Mat, dofmaps: typing.Tuple[typing.List[dcpp.fem.DofMap]],
                 restriction: typing.Optional[typing.Tuple[typing.List[mcpp.fem.DofMapRestriction]]] = None):
        self._A = A
        assert len(dofmaps) == 2
        if restriction is None:
            index_maps = ([(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps[0]],
                          [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps[1]])
            index_sets = (mcpp.la.petsc.create_index_sets(index_maps[0], [1] * len(index_maps[0])),
                          mcpp.la.petsc.create_index_sets(index_maps[1], [1] * len(index_maps[1])))
            self._unrestricted_index_sets = index_sets
            self._restricted_index_sets = None
            self._unrestricted_to_restricted = None
            self._unrestricted_to_restricted_bs = None
        else:
            assert len(restriction) == 2
            for i in range(2):
                assert len(dofmaps[i]) == len(restriction[i])
                assert all([_same_dofmap(dofmap, restriction_.dofmap)
                            for (dofmap, restriction_) in zip(dofmaps[i], restriction[i])])
            unrestricted_index_maps = ([(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps[0]],
                                       [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps[1]])
            unrestricted_index_sets = (mcpp.la.petsc.create_index_sets(unrestricted_index_maps[0],
                                                                       [1] * len(unrestricted_index_maps[0])),
                                       mcpp.la.petsc.create_index_sets(unrestricted_index_maps[1],
                                                                       [1] * len(unrestricted_index_maps[1])))
            restricted_index_maps = ([(restriction_.index_map, restriction_.index_map_bs)
                                      for restriction_ in restriction[0]],
                                     [(restriction_.index_map, restriction_.index_map_bs)
                                      for restriction_ in restriction[1]])
            restricted_index_sets = (mcpp.la.petsc.create_index_sets(restricted_index_maps[0],
                                                                     [1] * len(restricted_index_maps[0])),
                                     mcpp.la.petsc.create_index_sets(restricted_index_maps[1],
                                                                     [1] * len(restricted_index_maps[1])))
            unrestricted_to_restricted = ([restriction_.unrestricted_to_restricted for restriction_ in restriction[0]],
                                          [restriction_.unrestricted_to_restricted for restriction_ in restriction[1]])
            unrestricted_to_restricted_bs = ([restriction_.index_map_bs for restriction_ in restriction[0]],
                                             [restriction_.index_map_bs for restriction_ in restriction[1]])
            self._unrestricted_index_sets = unrestricted_index_sets
            self._restricted_index_sets = restricted_index_sets
            self._unrestricted_to_restricted = unrestricted_to_restricted
            self._unrestricted_to_restricted_bs = unrestricted_to_restricted_bs

    def __iter__(self):
        with ExitStack() as wrapper_stack:
            for index0, _ in enumerate(self._unrestricted_index_sets[0]):
                for index1, _ in enumerate(self._unrestricted_index_sets[1]):
                    if self._restricted_index_sets is None:
                        wrapper = _MatSubMatrixWrapper(self._A,
                                                       (self._unrestricted_index_sets[0][index0],
                                                        self._unrestricted_index_sets[1][index1]))
                    else:
                        wrapper = _MatSubMatrixWrapper(self._A,
                                                       (self._unrestricted_index_sets[0][index0],
                                                        self._unrestricted_index_sets[1][index1]),
                                                       (self._restricted_index_sets[0][index0],
                                                        self._restricted_index_sets[1][index1]),
                                                       (self._unrestricted_to_restricted[0][index0],
                                                        self._unrestricted_to_restricted[1][index1]),
                                                       (self._unrestricted_to_restricted_bs[0][index0],
                                                        self._unrestricted_to_restricted_bs[1][index1]))
                    yield (index0, index1, wrapper_stack.enter_context(wrapper))

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        for i in range(2):
            for index_set in self._unrestricted_index_sets[i]:
                index_set.destroy()
        if self._restricted_index_sets is not None:
            for i in range(2):
                for index_set in self._restricted_index_sets[i]:
                    index_set.destroy()


class NestMatSubMatrixWrapper(object):
    def __init__(self,
                 A: PETSc.Mat, dofmaps: typing.Tuple[typing.List[dcpp.fem.DofMap]],
                 restriction: typing.Optional[typing.Tuple[typing.List[mcpp.fem.DofMapRestriction]]] = None):
        self._A = A
        self._dofmaps = dofmaps
        self._restriction = restriction

    def __iter__(self):
        with ExitStack() as wrapper_stack:
            for index0, _ in enumerate(self._dofmaps[0]):
                for index1, _ in enumerate(self._dofmaps[1]):
                    A_sub = self._A.getNestSubMatrix(index0, index1)
                    if self._restriction is None:
                        wrapper_content = A_sub
                    else:
                        wrapper = MatSubMatrixWrapper(A_sub,
                                                      (self._dofmaps[0][index0], self._dofmaps[1][index1]),
                                                      (self._restriction[0][index0], self._restriction[1][index1]))
                        wrapper_content = wrapper_stack.enter_context(wrapper)
                    yield (index0, index1, wrapper_content)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass


@functools.singledispatch
def assemble_matrix(a: Form,
                    bcs: typing.List[DirichletBC] = [],
                    diagonal: float = 1.0,
                    coeffs=Coefficients(None, None),
                    restriction: typing.Optional[typing.Tuple[mcpp.fem.DofMapRestriction]] = None) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    _a = _create_cpp_form(a)
    A = create_matrix(_a, restriction)
    return assemble_matrix(A, _a, bcs, diagonal, coeffs, restriction)


@assemble_matrix.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a: Form,
      bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0,
      coeffs=Coefficients(None, None),
      restriction: typing.Optional[typing.Tuple[mcpp.fem.DofMapRestriction]] = None) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.
    """
    _a = _create_cpp_form(a)
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(_a),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(_a))
    function_spaces = _a.function_spaces
    if restriction is None:
        # Assemble form
        dcpp.fem.petsc.assemble_matrix(A, _a, c[0], c[1], _cpp_dirichletbc(bcs))

        if function_spaces[0].id == function_spaces[1].id:
            # Flush to enable switch from add to set in the matrix
            A.assemble(PETSc.Mat.AssemblyType.FLUSH)

            # Set diagonal
            dcpp.fem.petsc.insert_diagonal(A, function_spaces[0], _cpp_dirichletbc(bcs), diagonal)
    else:
        dofmaps = (function_spaces[0].dofmap, function_spaces[1].dofmap)

        # Assemble form
        with MatSubMatrixWrapper(A, dofmaps, restriction) as A_sub:
            dcpp.fem.petsc.assemble_matrix(A_sub, _a, c[0], c[1], _cpp_dirichletbc(bcs))

        if function_spaces[0].id == function_spaces[1].id:
            # Flush to enable switch from add to set in the matrix
            A.assemble(PETSc.Mat.AssemblyType.FLUSH)

            # Set diagonal
            with MatSubMatrixWrapper(A, dofmaps, restriction) as A_sub:
                dcpp.fem.petsc.insert_diagonal(A_sub, function_spaces[0], _cpp_dirichletbc(bcs), diagonal)
    return A


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_nest(a: typing.List[typing.List[Form]],
                         bcs: typing.List[DirichletBC] = [], mat_types=[],
                         diagonal: float = 1.0,
                         coeffs=Coefficients(None, None),
                         restriction: typing.Optional[typing.Tuple[
                                                      typing.List[mcpp.fem.DofMapRestriction]]] = None) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    _a = _create_cpp_form(a)
    A = create_matrix_nest(_a, restriction, mat_types)
    return assemble_matrix_nest(A, _a, bcs, diagonal, coeffs, restriction)


@assemble_matrix_nest.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a: typing.List[typing.List[Form]],
      bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0,
      coeffs=Coefficients(None, None),
      restriction: typing.Optional[typing.Tuple[
                                   typing.List[mcpp.fem.DofMapRestriction]]] = None) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    _a = _create_cpp_form(a)
    function_spaces = _get_block_function_spaces(_a)
    dofmaps = ([function_space.dofmap for function_space in function_spaces[0]],
               [function_space.dofmap for function_space in function_spaces[1]])

    # Assemble form
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(_a),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(_a))
    with NestMatSubMatrixWrapper(A, dofmaps, restriction) as nest_A:
        for i, j, A_sub in nest_A:
            a_sub = _a[i][j]
            if a_sub is not None:
                const_sub = c[0][i][j]
                coeff_sub = c[1][i][j]
                dcpp.fem.petsc.assemble_matrix(A_sub, a_sub, const_sub, coeff_sub, _cpp_dirichletbc(bcs))

    # Flush to enable switch from add to set in the matrix
    A.assemble(PETSc.Mat.AssemblyType.FLUSH)

    # Set diagonal
    with NestMatSubMatrixWrapper(A, dofmaps, restriction) as nest_A:
        for i, j, A_sub in nest_A:
            if function_spaces[0][i].id == function_spaces[1][j].id:
                a_sub = _a[i][j]
                if a_sub is not None:
                    dcpp.fem.petsc.insert_diagonal(A_sub, function_spaces[0][i], _cpp_dirichletbc(bcs), diagonal)

    return A


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_block(a: typing.List[typing.List[Form]],
                          bcs: typing.List[DirichletBC] = [],
                          diagonal: float = 1.0,
                          coeffs=Coefficients(None, None),
                          restriction: typing.Optional[typing.Tuple[
                                                       typing.List[mcpp.fem.DofMapRestriction]]] = None) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    _a = _create_cpp_form(a)
    A = create_matrix_block(_a, restriction)
    return assemble_matrix_block(A, _a, bcs, diagonal, coeffs, restriction)


@assemble_matrix_block.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a: typing.List[typing.List[Form]],
      bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0,
      coeffs=Coefficients(None, None),
      restriction: typing.Optional[typing.Tuple[
                                   typing.List[mcpp.fem.DofMapRestriction]]] = None) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    _a = _create_cpp_form(a)
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(_a),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(_a))
    function_spaces = _get_block_function_spaces(_a)
    dofmaps = ([function_space.dofmap for function_space in function_spaces[0]],
               [function_space.dofmap for function_space in function_spaces[1]])

    # Assemble form
    with BlockMatSubMatrixWrapper(A, dofmaps, restriction) as block_A:
        for i, j, A_sub in block_A:
            a_sub = _a[i][j]
            if a_sub is not None:
                const_sub = c[0][i][j]
                coeff_sub = c[1][i][j]
                dcpp.fem.petsc.assemble_matrix(A_sub, a_sub, const_sub, coeff_sub, _cpp_dirichletbc(bcs), True)

    # Flush to enable switch from add to set in the matrix
    A.assemble(PETSc.Mat.AssemblyType.FLUSH)

    # Set diagonal
    with BlockMatSubMatrixWrapper(A, dofmaps, restriction) as block_A:
        for i, j, A_sub in block_A:
            if function_spaces[0][i].id == function_spaces[1][j].id:
                a_sub = _a[i][j]
                if a_sub is not None:
                    dcpp.fem.petsc.insert_diagonal(A_sub, function_spaces[0][i], _cpp_dirichletbc(bcs), diagonal)

    return A


# -- Modifiers for Dirichlet conditions ---------------------------------------

def apply_lifting(b: PETSc.Vec,
                  a: typing.List[Form],
                  bcs: typing.List[typing.List[DirichletBC]],
                  x0: typing.Optional[typing.List[PETSc.Vec]] = None,
                  scale: float = 1.0,
                  coeffs=Coefficients(None, None),
                  restriction: typing.Optional[mcpp.fem.DofMapRestriction] = None,
                  restriction_x0: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None) -> None:
    """Modify RHS vector b for lifting of Dirichlet boundary conditions.
    It modifies b such that:

        b <- b - scale * A_j (g_j - x0_j)

    where j is a block (nest) index. For a non-blocked problem j = 0.
    The boundary conditions bcs are on the trial spaces V_j. The forms
    in [a] must have the same test space as L (from which b was built),
    but the trial space may differ. If x0 is not supplied, then it is
    treated as zero.

    Ghost contributions are not accumulated (not sent to owner). Caller
    is responsible for calling VecGhostUpdateBegin/End.
    """
    _a = _create_cpp_form(a)
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(_a),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(_a))
    function_spaces = [form.function_spaces[1] for form in _a]
    dofmaps_x0 = [function_space.dofmap for function_space in function_spaces]
    if restriction is None:
        with b.localForm() as b_local, NestVecSubVectorReadWrapper(x0, dofmaps_x0) as nest_x0:
            if x0 is None:
                x0_as_list = []
            else:
                x0_as_list = [x0_ for x0_ in nest_x0]
            dcpp.fem.apply_lifting(b_local.array_w, _a, c[0], c[1], _cpp_dirichletbc(bcs), x0_as_list, scale)
    else:
        with VecSubVectorWrapper(b, restriction.dofmap, restriction) as b_sub, \
                NestVecSubVectorReadWrapper(x0, dofmaps_x0, restriction_x0) as nest_x0:
            for a_sub, bcs_sub, x0_sub in zip(_a, bcs, nest_x0):
                if x0_sub is None:
                    x0_sub_as_list = []
                else:
                    x0_sub_as_list = [x0_sub]
                dcpp.fem.apply_lifting(
                    b_sub, [a_sub], c[0], c[1], _cpp_dirichletbc([bcs_sub]), x0_sub_as_list, scale)


def apply_lifting_nest(b: PETSc.Vec,
                       a: typing.List[typing.List[Form]],
                       bcs: typing.List[DirichletBC],
                       x0: typing.Optional[PETSc.Vec] = None,
                       scale: float = 1.0,
                       coeffs=Coefficients(None, None),
                       restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None,
                       restriction_x0: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None) -> PETSc.Vec:
    """Modify nested vector for lifting of Dirichlet boundary conditions.

    """
    _a = _create_cpp_form(a)
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(_a),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(_a))
    function_spaces = _get_block_function_spaces(_a)
    dofmaps = [function_space.dofmap for function_space in function_spaces[0]]
    dofmaps_x0 = [function_space.dofmap for function_space in function_spaces[1]]
    bcs1 = bcs_by_block(function_spaces[1], bcs)
    with NestVecSubVectorWrapper(b, dofmaps, restriction) as nest_b, \
            NestVecSubVectorReadWrapper(x0, dofmaps_x0, restriction_x0) as nest_x0:
        for b_sub, a_sub, constants_a, coeffs_a in zip(nest_b, _a, c[0], c[1]):
            for a_sub_, bcs1_sub, x0_sub, constant_a_sub, coeff_a_sub in zip(
                    a_sub, bcs1, nest_x0, constants_a, coeffs_a):
                if x0_sub is None:
                    x0_sub_as_list = []
                else:
                    x0_sub_as_list = [x0_sub]
                dcpp.fem.apply_lifting(
                    b_sub, [a_sub_], [constant_a_sub], [coeff_a_sub], _cpp_dirichletbc([bcs1_sub]),
                    x0_sub_as_list, scale)
    return b


def set_bc(b: typing.Union[PETSc.Vec],
           bcs: typing.List[DirichletBC],
           x0: typing.Optional[PETSc.Vec] = None,
           scale: float = 1.0,
           restriction: typing.Optional[mcpp.fem.DofMapRestriction] = None) -> None:
    """Insert boundary condition values into vector. Only local (owned)
    entries are set, hence communication after calling this function is
    not required unless ghost entries need to be updated to the boundary
    condition value.

    """
    if restriction is None:
        if x0 is not None:
            x0 = x0.array_r
        dcpp.fem.set_bc(b.array_w, _cpp_dirichletbc(bcs), x0, scale)
    else:
        with VecSubVectorWrapper(b, restriction.dofmap, restriction, ghosted=False) as b_sub, \
                VecSubVectorReadWrapper(x0, restriction.dofmap, restriction, ghosted=False) as x0_sub:
            dcpp.fem.set_bc(b_sub, _cpp_dirichletbc(bcs), x0_sub, scale)


def set_bc_nest(b: PETSc.Vec,
                bcs: typing.List[typing.List[DirichletBC]],
                x0: typing.Optional[PETSc.Vec] = None,
                scale: float = 1.0,
                restriction: typing.Optional[typing.List[mcpp.fem.DofMapRestriction]] = None) -> None:
    """Insert boundary condition values into nested vector. Only local (owned)
    entries are set, hence communication after calling this function is
    not required unless the ghost entries need to be updated to the
    boundary condition value.

    """
    if restriction is None:
        dofmaps = [None] * len(b.getNestSubVecs())
    else:
        dofmaps = [restriction_.dofmap for restriction_ in restriction]
    dofmaps_x0 = dofmaps
    restriction_x0 = restriction
    with NestVecSubVectorWrapper(b, dofmaps, restriction, ghosted=False) as nest_b, \
            NestVecSubVectorReadWrapper(x0, dofmaps_x0, restriction_x0, ghosted=False) as nest_x0:
        for b_sub, bcs_sub, x0_sub in zip(nest_b, bcs, nest_x0):
            dcpp.fem.set_bc(b_sub, _cpp_dirichletbc(bcs_sub), x0_sub, scale)

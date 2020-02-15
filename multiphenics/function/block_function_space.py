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

import types
import numpy
import cffi
from ufl.finiteelement import FiniteElementBase
from dolfinx import FunctionSpace, Mesh
import dolfinx.cpp
import dolfinx.fem.dofmap
from dolfinx.jit import ffcx_jit
from multiphenics.cpp import cpp
from multiphenics.function.block_element import BlockElement
from multiphenics.mesh import MeshRestriction

def _compile_dolfinx_element(element, mesh):
    # Compile dofmap and element and create DOLFIN-X objects
    ufc_element, ufc_dofmap_ptr = ffcx_jit(
        element,
        form_compiler_parameters=None,
        mpi_comm=mesh.mpi_comm())

    ffi = cffi.FFI()
    ufc_element = dolfinx.fem.dofmap.make_ufc_finite_element(ffi.cast("uintptr_t", ufc_element))
    dolfinx_element = dolfinx.cpp.fem.FiniteElement(ufc_element)
    ufc_dofmap = dolfinx.fem.dofmap.make_ufc_dofmap(ffi.cast("uintptr_t", ufc_dofmap_ptr))
    dolfinx_dofmap = dolfinx.cpp.fem.create_dofmap(ufc_dofmap, mesh)

    return dolfinx_element, dolfinx_dofmap

BlockFunctionSpace_Base = cpp.function.BlockFunctionSpace

class BlockFunctionSpace(object):
    "Base class for all block function spaces."

    def __init__(self, *args, **kwargs):
        assert len(args) in (1, 2)
        if len(args) == 1:
            assert isinstance(args[0], (list, tuple))
            assert (
                len(kwargs) == 0
                    or
                (len(kwargs) == 1 and "restrict" in kwargs)
            )
            self._init_from_function_spaces(*args, **kwargs)
        elif len(args) == 2:
            if isinstance(args[0], Mesh):
                assert isinstance(args[1], (list, tuple, BlockElement))
                assert (
                    len(kwargs) == 0
                        or
                    (len(kwargs) == 1 and "restrict" in kwargs)
                )
                self._init_from_ufl(*args, **kwargs)
            elif isinstance(args[0], BlockFunctionSpace_Base):
                assert isinstance(args[1], (list, tuple, BlockElement))
                assert len(kwargs) == 0
                self._init_from_cpp(*args, **kwargs)
            else:
                raise AssertionError("Invalid argument to BlockFunctionSpace")
        else:
            raise AssertionError("Invalid argument to BlockFunctionSpace")

    def _init_from_function_spaces(self, function_spaces, restrict=None):
        # Get the common mesh
        assert isinstance(function_spaces[0], FunctionSpace)
        mesh = function_spaces[0].mesh
        for function_space in function_spaces:
            assert isinstance(function_space, FunctionSpace)
            assert function_space.mesh.ufl_domain() == mesh.ufl_domain()
        # Initialize the BlockFunctionSpace_Base
        if restrict is None:
            self._cpp_object = BlockFunctionSpace_Base([function_space._cpp_object for function_space in function_spaces])
        else:
            restrict = self._init_restriction(mesh, restrict)
            assert len(restrict) == len(function_spaces)
            restrict_cpp = [[restrict_e_d for restrict_e_d in restrict_e] for restrict_e in restrict]
            self._cpp_object = BlockFunctionSpace_Base([function_space._cpp_object for function_space in function_spaces], restrict_cpp)

        # Fill in subspaces
        self._init_sub_spaces([function_space.ufl_element() for function_space in function_spaces])

    def _init_from_cpp(self, cppV, elements):
        # Store the BlockFunctionSpace_Base
        self._cpp_object = cppV

        # Fill in subspaces
        self._init_sub_spaces(elements)

    def _init_from_ufl(self, mesh, elements, restrict=None):
        # Compile elements and dofmaps and construct corresponding DOLFIN-X objects
        dolfinx_elements = list()
        dolfinx_dofmaps = list()
        for element in elements:
            assert isinstance(element, FiniteElementBase)
            dolfinx_element, dolfinx_dofmap = _compile_dolfinx_element(element, mesh)
            dolfinx_elements.append(dolfinx_element)
            dolfinx_dofmaps.append(dolfinx_dofmap)

        # Initialize the BlockFunctionSpace_Base
        if restrict is None:
            self._cpp_object = BlockFunctionSpace_Base(mesh, dolfinx_elements, dolfinx_dofmaps)
        else:
            restrict = self._init_restriction(mesh, restrict)
            assert len(restrict) == len(elements)
            restrict_cpp = [[restrict_e_d for restrict_e_d in restrict_e] for restrict_e in restrict]
            self._cpp_object = BlockFunctionSpace_Base(mesh, dolfinx_elements, dolfinx_dofmaps, restrict_cpp)

        # Fill in subspaces
        self._init_sub_spaces(elements)

    @staticmethod
    def _init_restriction(mesh, restrictions):
        assert isinstance(restrictions, (list, tuple))
        all_none = all([restriction is None for restriction in restrictions])
        at_least_one_subdomain = any([isinstance(restriction, types.FunctionType) for restriction in restrictions])
        at_least_one_mesh_restriction = any([isinstance(restriction, MeshRestriction) for restriction in restrictions])
        assert all_none or at_least_one_subdomain or at_least_one_mesh_restriction
        if all_none:
            return [[] for restriction in restrictions]
        elif at_least_one_subdomain:
            assert not at_least_one_mesh_restriction, "Please do not mix functions defining subdomains and MeshRestrictions, rather provide only MeshRestrictions"
            mesh_restrictions = list()
            for subdomain in restrictions:
                mesh_restriction = MeshRestriction(mesh)
                if subdomain is not None:
                    mesh_restriction.mark(subdomain)
                else:
                    mesh_restriction.mark(lambda x: numpy.full(x.shape[1], True))
                mesh_restrictions.append(mesh_restriction)
            return mesh_restrictions
        elif at_least_one_mesh_restriction:
            assert not at_least_one_subdomain, "Please do not mix functions defining subdomains and MeshRestrictions, rather provide only MeshRestrictions"
            return [restriction if isinstance(restriction, MeshRestriction) else [] for restriction in restrictions]
        else:
            raise AssertionError("Invalid arguments provided as BlockFunctionSpace restriction")

    def _init_sub_spaces(self, ufl_sub_elements):
        def extend_sub_function_space(sub_function_space, i):
            # Make sure to preserve a reference to the block function
            def block_function_space(self_):
                return self
            sub_function_space.block_function_space = types.MethodType(block_function_space, sub_function_space)

            # ... and a reference to the block index
            def block_index(self_):
                return i
            sub_function_space.block_index = types.MethodType(block_index, sub_function_space)

            # ... and that these methods are preserved by sub_function_space.sub()
            original_sub = sub_function_space.sub
            def sub(self_, j):
                output = original_sub(j)
                extend_sub_function_space(output, i)
                return output
            sub_function_space.sub = types.MethodType(sub, sub_function_space)

        self._num_sub_spaces = len(ufl_sub_elements)
        self._sub_spaces = list()
        for (i, element_i) in enumerate(ufl_sub_elements):
            # Extend .sub() call with the python layer of FunctionSpace
            sub_function_space = FunctionSpace(None, element_i, self._cpp_object.sub(i))

            # Extend with block function space and block index methods
            extend_sub_function_space(sub_function_space, i)

            # Append
            self._sub_spaces.append(sub_function_space)

        # Finally, fill in ufl_element
        self._ufl_element = BlockElement(*ufl_sub_elements)

    def __str__(self):
        "Pretty-print."
        elements = [str(subspace.ufl_element()) for subspace in self]
        return "<Block function space of dimension %d (%s)>" % \
               (self.dim(), str(elements))

    def ufl_element(self):
        return self._ufl_element

    @property
    def mesh(self):
        return self._cpp_object.mesh

    @property
    def block_dofmap(self):
        return self._cpp_object.block_dofmap

    def tabulate_dof_coordinates(self):
        return self._cpp_object.tabulate_dof_coordinates()

    @property
    def dim(self) -> int:
        return self._cpp_object.dim

    def num_sub_spaces(self):
        "Return the number of sub spaces"
        return self._num_sub_spaces

    def __len__(self):
        "Return the number of sub spaces"
        return self.num_sub_spaces()

    def __getitem__(self, i):
        """
        Return the i-th sub space, *neglecting* restrictions.
        """
        return self.sub(i)

    def sub(self, i):
        """
        Return the i-th sub space, *neglecting* restrictions.
        """
        return self._sub_spaces[i]

    def extract_block_sub_space(self, component, restrict=True):
        """
        Extract block subspace for component, possibly considering restrictions.

        *Arguments*
            component (numpy.array(uint))
               The component.
            restrict (bool)
               Consider or not restrictions

        *Returns*
            _BlockFunctionSpace_
                The block subspace.
        """
        # Transform the argument to a NumPy array
        assert hasattr(component, "__len__")
        component = numpy.asarray(component, dtype=numpy.uintp)

        # Get the cpp version of the BlockFunctionSpace
        cpp_space = self._cpp_object.extract_block_sub_space(component, restrict)

        # Extend with the python layer
        block_sub_element = [self._ufl_element[component_] for component_ in component]
        python_space = BlockFunctionSpace(cpp_space, BlockElement(*block_sub_element))

        # Store the components in the python space
        python_space.is_block_subspace = True
        python_space.sub_components_to_components = dict([(sub_component, int(component_)) for (sub_component, component_) in enumerate(component)])
        python_space.components_to_sub_components = dict([(int(component_), sub_component) for (sub_component, component_) in enumerate(component)])
        python_space.parent_block_function_space = self

        # Return
        return python_space

    def __iter__(self):
        return self._sub_spaces.__iter__()

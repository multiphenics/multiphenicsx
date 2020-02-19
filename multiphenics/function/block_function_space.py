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
from numpy import arange
from dolfinx import FunctionSpace
from multiphenics.cpp import cpp

BlockFunctionSpace_Base = cpp.function.BlockFunctionSpace

class BlockFunctionSpace(object):
    "Class for block function spaces."

    def __init__(self, function_spaces, restrict=None, cpp_object=None):
        # Check function spaces consistency
        assert isinstance(function_spaces, (list, tuple))
        assert isinstance(function_spaces[0], FunctionSpace)
        mesh = function_spaces[0].mesh
        for function_space in function_spaces:
            assert isinstance(function_space, FunctionSpace)
            assert function_space.mesh.ufl_domain() == mesh.ufl_domain()

        # Initialize cpp block function space based on existing cpp_object, if provided
        if cpp_object is not None:
            assert restrict is None, "restrict kwarg is not supposed to be used in combination with cpp_object kwarg"
            self._cpp_object = cpp_object
        else:
            # Provide default argument to restrict kwarg, if not provided, which does not perform any restriction
            if restrict is None:
                restrict = [arange(0, function_space.dofmap.index_map.block_size*(
                            function_space.dofmap.index_map.size_local + function_space.dofmap.index_map.num_ghosts
                            )) for function_space in function_spaces]
            assert len(restrict) == len(function_spaces)
            self._cpp_object = BlockFunctionSpace_Base([function_space._cpp_object
                                                        for function_space in function_spaces], restrict)

        # Store and patch subspaces
        def attach_block_function_space_and_block_index_to_function_space(index, function_space):
            # Make sure to preserve a reference to the block function space
            function_space.block_function_space = types.MethodType(lambda _: self, function_space)

            # ... and a reference to the block index
            function_space.block_index = types.MethodType(lambda _: index, function_space)

            # ... and that these methods are preserved by function_space.sub()
            original_sub = function_space.sub
            def sub(self_, j):
                output = original_sub(j)
                attach_block_function_space_and_block_index_to_function_space(index, output)
                return output
            function_space.sub = types.MethodType(sub, function_space)

        # TODO need to clone because of attach_block_function_space_and_block_index_to_function_space
        #      It would probably be best to remove such patch altogether
        self._sub_spaces = [function_space.clone() for function_space in function_spaces]
        for (index, function_space) in enumerate(self._sub_spaces):
            attach_block_function_space_and_block_index_to_function_space(index, function_space)

        # Finally, fill in ufl_element
        self._ufl_element = [function_space.ufl_element() for function_space in function_spaces]

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
        return len(self._sub_spaces)

    def __len__(self):
        "Return the number of sub spaces"
        return self.num_sub_spaces()

    def __getitem__(self, i):
        """
        Return the i-th sub space, *neglecting* restrictions.
        """
        return self.sub(i)

    def __iter__(self):
        return self._sub_spaces.__iter__()

    def sub(self, i):
        """
        Return the i-th sub space, *neglecting* restrictions.
        """
        return self._sub_spaces[i]

    def extract_block_sub_space(self, component, restrict=True):
        """
        Extract block subspace for component, possibly considering restrictions.

        *Arguments*
            component (array(uint))
               The component.
            restrict (bool)
               Consider or not restrictions

        *Returns*
            _BlockFunctionSpace_
                The block subspace.
        """

        # Get the cpp version of the BlockFunctionSpace
        cpp_space = self._cpp_object.extract_block_sub_space(component, restrict)

        # Extend with the python layer
        sub_spaces = [self._sub_spaces[component_] for component_ in component]
        python_space = BlockFunctionSpace(sub_spaces, cpp_object=cpp_space)

        # Store the components in the python space
        python_space.is_block_subspace = True
        python_space.sub_components_to_components = {sub_component: int(component_)
                                                     for (sub_component, component_) in enumerate(component)}
        python_space.components_to_sub_components = {int(component_): sub_component
                                                     for (sub_component, component_) in enumerate(component)}
        python_space.parent_block_function_space = self

        # Return
        return python_space

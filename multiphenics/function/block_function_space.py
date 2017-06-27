# Copyright (C) 2016-2017 by the multiphenics authors
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
from six import string_types
import numpy
from ufl.finiteelement import FiniteElementBase
import dolfin.cpp
from dolfin import FunctionSpace, Mesh, MeshFunction, MeshFunctionBool
from dolfin.compilemodules.jit import jit
from dolfin.functions.functionspace import _compile_dolfin_element
from multiphenics.function.block_element import BlockElement
from multiphenics.swig import cpp
from multiphenics.mesh import MeshRestriction, SubDomain

class BlockFunctionSpace(cpp.BlockFunctionSpace):
    "Base class for all block function spaces."

    def __init__(self, *args, **kwargs):
        assert len(args) in (1, 2)        
        if len(args) == 1:
            assert isinstance(args[0], (list, tuple, cpp.BlockFunctionSpace))
            if isinstance(args[0], (list, tuple)):
                assert (
                    len(kwargs) is 0 
                        or 
                    (len(kwargs) is 1 and "restrict" in kwargs)
                )
                self._init_from_function_spaces(*args, **kwargs)
            elif isinstance(args[0], cpp.BlockFunctionSpace):
                assert len(kwargs) is 1
                assert "num_sub_spaces" in kwargs
                self._init_from_cpp(*args, **kwargs)
            else:
                raise AssertionError("Invalid argument to BlockFunctionSpace")
        elif len(args) == 2:
            assert isinstance(args[0], Mesh)
            assert isinstance(args[1], (list, tuple, BlockElement))
            assert (
                len(kwargs) is 0 
                    or 
                (len(kwargs) is 1 and "restrict" in kwargs)
            )
            self._init_from_ufl(*args, **kwargs)
            
    def _init_from_function_spaces(self, function_spaces, restrict=None):
        # Get the common mesh                
        assert isinstance(function_spaces[0], FunctionSpace)
        mesh = function_spaces[0].mesh()
        for function_space in function_spaces:
            assert isinstance(function_space, FunctionSpace)
            assert function_space.mesh().ufl_domain() == mesh.ufl_domain()
        # Initialize the cpp.BlockFunctionSpace
        if restrict is None:
            cpp.BlockFunctionSpace.__init__(self, function_spaces)
        else:
            restrict = self._init_restriction(mesh, restrict)
            assert len(restrict) == len(function_spaces)
            cpp.BlockFunctionSpace.__init__(self, function_spaces, restrict)
            
        # Fill in subspaces
        self._init_sub_spaces(len(function_spaces))
    
    def _init_from_cpp(self, cppV, **kwargs):
        # Assign all the members (including 'this' pointer to SWIG wraper)
        # NOTE: This in fact performs assignment of C++ context
        self.__dict__ = cppV.__dict__
        
        # Fill in subspaces
        assert "num_sub_spaces" in kwargs
        self._init_sub_spaces(kwargs["num_sub_spaces"])

    def _init_from_ufl(self, mesh, elements, restrict=None):
        # Compile elements and dofmaps and construct corresponding dolfin objects
        dolfin_elements = list()
        dolfin_dofmaps = list()
        for element in elements:
            assert isinstance(element, FiniteElementBase)
            dolfin_element, dolfin_dofmap = _compile_dolfin_element(element, mesh)
            dolfin_elements.append(dolfin_element)
            dolfin_dofmaps.append(dolfin_dofmap)
            
        # Initialize the cpp.BlockFunctionSpace
        if restrict is None:
            cpp.BlockFunctionSpace.__init__(self, mesh, dolfin_elements, dolfin_dofmaps)
        else:
            restrict = self._init_restriction(mesh, restrict)
            assert len(restrict) == len(elements)
            cpp.BlockFunctionSpace.__init__(self, mesh, dolfin_elements, dolfin_dofmaps, restrict)
        
        # Fill in subspaces
        self._init_sub_spaces(len(elements))
    
    @staticmethod
    def _init_restriction(mesh, subdomains):
        assert isinstance(subdomains, (list, tuple))
        all_none = all([subdomain is None for subdomain in subdomains])
        at_least_one_subdomain = any([isinstance(subdomain, SubDomain) for subdomain in subdomains])
        at_least_one_mesh_function = any([(
                isinstance(subdomain, (list, tuple)) 
                    and 
                all([isinstance(mesh_function, MeshFunctionBool) for mesh_function in subdomain])
            ) for subdomain in subdomains])
        assert all_none or at_least_one_subdomain or at_least_one_mesh_function
        if all_none:
            mesh_functions_for_subdomains = list()
            for subdomain in subdomains:
                empty_mesh_functions_for_current_subdomain = list()
                mesh_functions_for_subdomains.append(empty_mesh_functions_for_current_subdomain)
            return mesh_functions_for_subdomains
        elif at_least_one_subdomain:
            assert not at_least_one_mesh_function, "Please do not mix SubDomains and MeshFunctions, rather provide only MeshFunctions"
            mesh_functions_for_subdomains = list()
            for subdomain in subdomains:
                mesh_functions_for_current_subdomain = MeshRestriction(mesh, subdomain)
                mesh_functions_for_subdomains.append(mesh_functions_for_current_subdomain)
            return mesh_functions_for_subdomains
        elif at_least_one_mesh_function:
            assert not at_least_one_subdomain, "Please do not mix SubDomains and MeshFunctions, rather provide only MeshFunctions"
            mesh_functions_for_subdomains = list()
            for mesh_functions in subdomains:
                if mesh_functions is not None:
                    assert all([isinstance(mesh_function, MeshFunction)] for mesh_function in mesh_functions)
                    mesh_functions_for_subdomains.append(mesh_functions)
                else:
                    empty_mesh_functions_for_current_subdomain = list()
                    mesh_functions_for_subdomains.append(empty_mesh_functions_for_current_subdomain)
            return mesh_functions_for_subdomains
        else:
            raise AssertionError("Invalid arguments provided as BlockFunctionSpace restriction")
    
    def _init_sub_spaces(self, num_sub_spaces):
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
            
        self._num_sub_spaces = num_sub_spaces
        self._sub_spaces = list()
        for i in range(num_sub_spaces):
            # Extend .sub() call with the python layer of FunctionSpace
            sub_function_space = FunctionSpace(cpp.BlockFunctionSpace.sub(self, i))
            
            # Extend with block function space and block index methods
            extend_sub_function_space(sub_function_space, i)
            
            # Append
            self._sub_spaces.append(sub_function_space)
    
    def __str__(self):
        "Pretty-print."
        elements = [str(subspace.ufl_element()) for subspace in self]
        return "<Block function space of dimension %d (%s)>" % \
               (self.block_dofmap().global_dimension(), str(elements))

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
        if not isinstance(i, int):
            raise TypeError("expected an int for 'i'")
        if i >= self.num_sub_spaces():
            raise ValueError("Can only extract SubSpaces with i = 0 ... %d" % \
                  (self.num_sub_spaces() - 1))

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
        cpp_space = cpp.BlockFunctionSpace.extract_block_sub_space(self, component, restrict)

        # Extend with the python layer
        python_space = BlockFunctionSpace(cpp_space, num_sub_spaces=len(component))
        
        # Store the components in the python space
        python_space.is_block_subspace = True
        python_space.sub_components_to_components = dict([(sub_component, int(component_)) for (sub_component, component_) in enumerate(component)])
        python_space.components_to_sub_components = dict([(int(component_), sub_component) for (sub_component, component_) in enumerate(component)])
        python_space.parent_block_function_space = self
        
        # Return
        return python_space
        
    def __iter__(self):
        return self._sub_spaces.__iter__()
        

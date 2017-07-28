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

from multiphenics.swig.multiphenics_compile_extension_module import multiphenics_compile_extension_module

# Storage for additional declarations
additional_declarations = dict()

# Define additional declarations: common macros
additional_declarations["common"] = {
    "pre":
        """
        // The macro for typemap of std::vector<std::vector<std::shared_ptr<TYPE>>> is defined in the following file,
        // and not in the standard typemaps swig include file
        %include "dolfin/swig/fem/pre.i"

        // --- Declare additional typemaps --- //
        TYPEMAPS_STD_VECTOR_OF_POINTERS(BlockFunctionSpace)
        TYPEMAPS_STD_VECTOR_OF_POINTERS(FiniteElement)
        TYPEMAPS_STD_VECTOR_OF_POINTERS(Form)
        IN_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_SHARED_POINTERS(MeshFunction<bool>)
        IN_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_SHARED_POINTERS(DirichletBC)
        MAP_OUT_TYPEMAPS(dolfin::la_index, dolfin::la_index, int, NPY_INT)
        OUT_SET_TYPEMAP_OF_PRIMITIVES(dolfin::la_index, NPY_INT)
        """,
        
    "post":
        """
        """
}

# Define additional declarations: fem module
additional_declarations["fem"] = {
    "pre":
        """
        // --- Templates for BlockFormBase --- //
        // Instantiate Hierarchical template class
        %template (HierarchicalBlockFormBase) dolfin::Hierarchical<dolfin::BlockFormBase>;
        
        // --- Ignores for BlockForm1 --- //
        // Ignore operator(), we will provide it in python to avoid conversions
        %ignore dolfin::BlockForm1::operator();

        // --- Ignores for BlockForm2 --- //
        // Ignore operator(), we will provide it in python to avoid conversions
        %ignore dolfin::BlockForm2::operator();      
        
        // --- Templates for BlockDirichletBC --- //
        // Instantiate Hierarchical template class
        %template (HierarchicalBlockDirichletBC) dolfin::Hierarchical<dolfin::BlockDirichletBC>;  
        """,
        
    "post":
        """
        """
}

# Define additional declarations: function module
additional_declarations["function"] = {
    "pre":
        """
        // --- Renames, ignores and templates for BlockFunction --- //
        // Rename operator[] to sub
        %rename(sub) dolfin::BlockFunction::operator[];
        // Ignore operator=, we will define block_assign instead
        %ignore dolfin::BlockFunction::operator=;
        // Instantiate Hierarchical template class
        %template (HierarchicalBlockFunction) dolfin::Hierarchical<dolfin::BlockFunction>;

        // --- Ignores and templates for BlockFunctionSpace --- //
        // Ignore copy constructor, otherwise it would take precedence
        // over the constructor for list of function spaces
        %ignore dolfin::BlockFunctionSpace::BlockFunctionSpace(const BlockFunctionSpace&);
        // Ignore operator[], since a sub method has been already provided
        %ignore dolfin::BlockFunctionSpace::operator[];
        // Instantiate Hierarchical template class
        %template (HierarchicalBlockFunctionSpace) dolfin::Hierarchical<dolfin::BlockFunctionSpace>;
        """,
        
    "post":
        """
        """
}

# Define additional declarations: la module
additional_declarations["la"] = {
    "pre":
        """
        // --- shared_ptr for GenericBlockVector --- //
        // Mark as shared_ptr
        %shared_ptr(dolfin::GenericBlockVector)
        
        // --- shared_ptr for GenericBlockMatrix --- //
        // Mark as shared_ptr
        %shared_ptr(dolfin::GenericBlockMatrix)
        
        #ifdef HAS_PETSC
        // --- Director and ignores for BlockPETScVector --- //
        // Enable director
        %feature("director") dolfin::BlockPETScVector;
        // Ignore non pythonic assignment operator
        %ignore dolfin::BlockPETScVector::operator=;
        // Ignore Parent's init operator which is causing problems with recent swig
        %ignore dolfin::PETScVector::init;
        // Rename copy to _copy
        %rename(_copy) dolfin::BlockPETScVector::copy;
        #endif
        
        #ifdef HAS_PETSC
        // --- Director and ignores for BlockPETScMatrix --- //
        // Enable director
        %feature("director") dolfin::BlockPETScMatrix;
        // Ignore non pythonic assignment operator
        %ignore dolfin::BlockPETScMatrix::operator=;
        // Rename copy to _copy
        %rename(_copy) dolfin::BlockPETScMatrix::copy;
        #endif
        
        // dolfin::ArrayView is not automatically imported because it is used only by 
        // virtual methods of Parent classes. Import it manually.
        %import(package="", module="dolfin.cpp.common") "dolfin/common/ArrayView.h"
        """,
                
    "post":
        """
        // The macro for as_backend_type of vectors and matrices is defined in dolfin/swig/la/post.i
        // However, this file is not installed with the library. We copy related macros below
        // =================== begin macros =================== //
        %pythoncode %{
        _has_block_type_map = {}
        _as_backend_block_type_map = {}
        %}

        %define AS_BACKEND_BLOCK_TYPE_MACRO(BLOCK_TENSOR_TYPE)
        %inline %{
        bool _has_block_type_ ## BLOCK_TENSOR_TYPE(const std::shared_ptr<dolfin::LinearAlgebraObject> block_tensor)
        { return dolfin::has_type<dolfin::BLOCK_TENSOR_TYPE>(*block_tensor); }

        std::shared_ptr<dolfin::BLOCK_TENSOR_TYPE> _as_backend_block_type_ ## BLOCK_TENSOR_TYPE(const std::shared_ptr<dolfin::LinearAlgebraObject> block_tensor)
        { return dolfin::as_type<dolfin::BLOCK_TENSOR_TYPE>(block_tensor); }
        %}

        %pythoncode %{
        _has_block_type_map[BLOCK_TENSOR_TYPE] = _has_block_type_ ## BLOCK_TENSOR_TYPE
        _as_backend_block_type_map[BLOCK_TENSOR_TYPE] = _as_backend_block_type_ ## BLOCK_TENSOR_TYPE
        %}

        %enddef

        %pythoncode %{
        import dolfin
        
        def get_tensor_type(tensor):
            "Return the concrete subclass of tensor."
            for k, v in _has_block_type_map.items():
                if v(tensor):
                    return k
            else:
                return dolfin.get_tensor_type(tensor)
                
        def has_type(tensor, subclass):
            "Return wether tensor is of the given subclass."
            global _has_block_type_map
            assert _has_block_type_map
            if subclass in _has_block_type_map:
                return bool(_has_block_type_map[subclass](tensor))
            else:
                return dolfin.has_type(tensor, subclass)
                
        def as_backend_type(tensor, subclass=None):
            "Cast tensor to the given subclass, passing the wrong class is an error."
            global _as_backend_block_type_map
            assert _as_backend_block_type_map
            if subclass is None:
                subclass = get_tensor_type(tensor)
            if subclass in _as_backend_block_type_map:
                ret = _as_backend_block_type_map[subclass](tensor)

                # Store the tensor to avoid garbage collection
                ret._org_upcasted_tensor = tensor
                return ret
            else:
                return dolfin.as_backend_type(tensor, subclass)
        %}
        // =================== end macros =================== //

        #ifdef HAS_PETSC
        // --- Backend type for BlockPETScVector --- //
        AS_BACKEND_BLOCK_TYPE_MACRO(BlockPETScVector)
        #endif

        #ifdef HAS_PETSC
        // --- Backend type for BlockPETScMatrix --- //
        AS_BACKEND_BLOCK_TYPE_MACRO(BlockPETScMatrix)
        #endif
        
        // Extend the .copy() method such that it returns the backend type
        // =================== begin macros =================== //
        %define COPY_RETURNS_BACKEND_TYPE(BLOCK_TENSOR_TYPE)
        %extend dolfin::BLOCK_TENSOR_TYPE {
        %pythoncode
        %{
        def copy(self):
            self_copy = self._copy()
            self_copy = as_backend_type(self_copy)
            return self_copy
        %}
        }
        %enddef
        // =================== end macros =================== //

        #ifdef HAS_PETSC
        // --- Extend .copy() for BlockPETScVector --- //
        COPY_RETURNS_BACKEND_TYPE(BlockPETScVector)
        #endif

        #ifdef HAS_PETSC
        // --- Extend .copy() for BlockPETScMatrix --- //
        COPY_RETURNS_BACKEND_TYPE(BlockPETScMatrix)
        #endif
        
        // --- Correctly handle matrix-vector product for BlockPETScMatrix, --- //
        // --- working around the fact that get_tensor_type does not get    --- //
        // --- extended to the parent class because it is a free function   --- //
        // --- rather than a method of the matrix class                     --- //
        %extend dolfin::BlockPETScMatrix {
        %pythoncode
        %{
        def __mul__(self, other):
            if isinstance(other, BlockPETScVector):
                ret = BlockPETScVector()
                self.mult(other, ret)
                return ret
            else:
                return super(BlockPETScMatrix, self).__mul__(other)
        %}
        }
        """
}
for (EigenSolver, Vector) in (
    ("CondensedSLEPcEigenSolver", "dolfin.PETScVector"),
    ("CondensedBlockSLEPcEigenSolver", "BlockPETScVector"), 
):
    additional_declarations["la"]["pre"] += \
        """
        #ifdef HAS_SLEPC
        // --- Renames, ignores and shared_ptr for {0} --- //
        // Ignore the get_eigenvalue and get_eigenpair methods, we will provide it in python
        %ignore dolfin::{0}::get_eigenvalue;
        %ignore dolfin::{0}::get_eigenpair;
        // Mark as shared_ptr
        %shared_ptr(dolfin::{0})
        #endif
        """.format(EigenSolver)
        
    # note that curly braces have been doubled due to str.format()
    additional_declarations["la"]["post"] += \
        """
        // --- get_eigenvalue and get_eigenpair methods for {0} --- //
        #ifdef HAS_SLEPC
        %extend dolfin::{0} {{
        PyObject* _get_eigenvalue(const int i) {{
            double lr, lc;
            self->get_eigenvalue(lr, lc, i);
            return Py_BuildValue("dd", lr, lc);
        }}
        
        PyObject* _get_eigenpair(dolfin::PETScVector& r, dolfin::PETScVector& c, const int i) {{
            double lr, lc;
            self->get_eigenpair(lr, lc, r, c, i);
            return Py_BuildValue("dd", lr, lc);
        }}
        
        %pythoncode %{{
        import dolfin
        
        def get_eigenpair(self, i=0, r_vec=None, c_vec=None):
            r_vec = r_vec or {1}()
            c_vec = c_vec or {1}()
            lr, lc = self._get_eigenpair(r_vec, c_vec, i)
            return lr, lc, r_vec, c_vec

        def get_eigenvalue(self, i=0):
            return self._get_eigenvalue(i)
        %}}
        }}
        #endif
        """.format(EigenSolver, Vector)


# Define additional declarations: mesh module
additional_declarations["mesh"] = {
    "pre":
        """
        // --- Director and renames for CustomizedSubDomain --- //
        %feature("director") dolfin::CustomizedSubDomain;
        // Rename mark method to provide additional checks
        %rename (_mark) dolfin::CustomizedSubDomain::mark;
        """,
        
    "post":
        """
        // --- mark method for CustomizedSubDomain --- //
        %extend dolfin::CustomizedSubDomain {
        %pythoncode
        %{
        import dolfin
        
        def mark(self, *args, **kwargs):
            if len(args) == 2:
                assert isinstance(args[0], \
                            (dolfin.MeshFunctionSizet, dolfin.MeshFunctionInt,
                             dolfin.MeshFunctionDouble, dolfin.MeshFunctionBool))
            if ("check_midpoint" in kwargs):
                args = args + (kwargs["check_midpoint"],)
            self._mark(*args)
        %}
        }
        """
}

# Define additional declarations: nls module
additional_declarations["nls"] = {
    "pre":
        """
        """,
        
    "post":
        """
        """
}

# Collect all additional declarations
all_additional_declarations = {
    "pre":
        additional_declarations["common"]["pre"] +
        additional_declarations["fem"]["pre"] +
        additional_declarations["function"]["pre"] +
        additional_declarations["la"]["pre"] +
        additional_declarations["mesh"]["pre"] +
        additional_declarations["nls"]["pre"],
        
    "post":
        additional_declarations["common"]["post"] +
        additional_declarations["fem"]["post"] +
        additional_declarations["function"]["post"] +
        additional_declarations["la"]["post"] +
        additional_declarations["mesh"]["post"] +
        additional_declarations["nls"]["post"]
}

# Compile extension module
cpp = multiphenics_compile_extension_module(
    # Files are manually sorted to handle dependencies
    "log/log.cpp",
    "fem/BlockDofMap.cpp", 
    "function/BlockFunctionSpace.cpp",
    "fem/BlockFormBase.cpp",
    "fem/BlockForm1.cpp",
    "fem/BlockForm2.cpp",
    "la/BlockInsertMode.cpp",
    "la/GenericBlockVector.cpp",
    "la/GenericBlockMatrix.cpp",
    "la/BlockPETScVector.cpp",
    "la/BlockPETScMatrix.cpp",
    "la/BlockPETScSubMatrix.cpp",
    "la/BlockPETScSubVector.cpp",
    "la/GenericBlockLinearAlgebraFactory.cpp",
    "la/BlockDefaultFactory.cpp",
    "la/BlockPETScFactory.cpp",
    "function/BlockFunction.cpp",
    "fem/BlockAssemblerBase.cpp",
    "fem/BlockAssembler.cpp",
    "fem/BlockDirichletBC.cpp",
    "mesh/SubDomain.cpp",
    "la/CondensedSLEPcEigenSolver.cpp",
    "la/CondensedBlockSLEPcEigenSolver.cpp",
    # Additional keyword arguments
    additional_declarations=all_additional_declarations
)

__all__ = [
    'cpp',
    'multiphenics_compile_extension_module'
]

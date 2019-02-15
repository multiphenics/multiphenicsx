// Copyright (C) 2016-2020 by the multiphenics authors
//
// This file is part of multiphenics.
//
// multiphenics is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// multiphenics is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
//

#ifndef __BLOCK_FUNCTION_SPACE_H
#define __BLOCK_FUNCTION_SPACE_H

#include <dolfin/function/FunctionSpace.h>
#include <multiphenics/fem/BlockDofMap.h>

namespace multiphenics
{
  namespace function
  {
    /// This class represents a finite element function space defined by
    /// a mesh, a vector of finite elements, and a vector of local-to-global mapping of the
    /// degrees of freedom (dofmap).

    class BlockFunctionSpace : public dolfin::common::Variable
    {
    public:
    
      /// Create a block function space from a list of existing function spaces (on the same mesh)
      ///
      /// *Arguments*
      ///     function_spaces (_FunctionSpace_)
      ///         List of existing function spaces.
      explicit BlockFunctionSpace(std::vector<std::shared_ptr<const dolfin::function::FunctionSpace>> function_spaces);
      
      /// Create a block function space from a list of existing function spaces (on the same mesh),
      /// but keeping only a part
      ///
      /// *Arguments*
      ///     function_spaces (_FunctionSpace_)
      ///         List of existing function spaces.
      ///     restrictions (vector (over blocks) of vector (over space dimensions) of _MeshFunction<bool>_)
      ///         The restrictions.
      BlockFunctionSpace(std::vector<std::shared_ptr<const dolfin::function::FunctionSpace>> function_spaces,
                         std::vector<std::vector<std::shared_ptr<const dolfin::mesh::MeshFunction<bool>>>> restrictions);

      /// Create a block function space for given mesh, vector of elements and vector of dofmaps
      /// (shared data)
      ///
      /// *Arguments*
      ///     mesh (_Mesh_)
      ///         The mesh.
      ///     elements (vector of _FiniteElement_)
      ///         The elements.
      ///     dofmaps (vector of _GenericDofMap_)
      ///         The dofmaps.
      BlockFunctionSpace(std::shared_ptr<const dolfin::mesh::Mesh> mesh,
                         std::vector<std::shared_ptr<const dolfin::fem::FiniteElement>> elements,
                         std::vector<std::shared_ptr<const dolfin::fem::GenericDofMap>> dofmaps);
                    
      /// Create a block function space for given mesh, vector of elements and vector of dofmaps
      /// but keeping only a part 
      /// (shared data)
      ///
      /// *Arguments*
      ///     mesh (_Mesh_)
      ///         The mesh.
      ///     elements (vector of _FiniteElement_)
      ///         The elements.
      ///     dofmaps (vector of _GenericDofMap_)
      ///         The dofmaps.
      ///     restrictions (vector (over blocks) of vector (over space dimensions) of _MeshFunction<bool>_)
      ///         The restrictions.
      BlockFunctionSpace(std::shared_ptr<const dolfin::mesh::Mesh> mesh,
                         std::vector<std::shared_ptr<const dolfin::fem::FiniteElement>> elements,
                         std::vector<std::shared_ptr<const dolfin::fem::GenericDofMap>> dofmaps,
                         std::vector<std::vector<std::shared_ptr<const dolfin::mesh::MeshFunction<bool>>>> restrictions);

      /// Copy constructor
      ///
      /// *Arguments*
      ///     V (_BlockFunctionSpace_)
      ///         The object to be copied.
      BlockFunctionSpace(const BlockFunctionSpace& V);

      /// Destructor
      virtual ~BlockFunctionSpace();
      
    private:
                                           
      // Initialize _elements and _dofmaps from function spaces
      void _init_mesh_and_elements_and_dofmaps_from_function_spaces();
      
      // Initialize _function_spaces from elements and dofmaps
      void _init_function_spaces_from_elements_and_dofmaps();
                                 
      // Initialize _block_dofmap from dofmaps and restrictions
      void _init_block_dofmap_from_dofmaps_and_restrictions();

    public:
      
      /// Assignment operator
      ///
      /// *Arguments*
      ///     V (_BlockFunctionSpace_)
      ///         Another block function space.
      const BlockFunctionSpace& operator= (const BlockFunctionSpace& V);

      /// Equality operator
      ///
      /// *Arguments*
      ///     V (_BlockFunctionSpace_)
      ///         Another block function space.
      bool operator== (const BlockFunctionSpace& V) const;

      /// Inequality operator
      ///
      /// *Arguments*
      ///     V (_BlockFunctionSpace_)
      ///         Another block function space.
      bool operator!= (const BlockFunctionSpace& V) const;

      /// Return mesh
      ///
      /// *Returns*
      ///     _Mesh_
      ///         The mesh.
      std::shared_ptr<const dolfin::mesh::Mesh> mesh() const;

      /// Return finite elements
      ///
      /// *Returns*
      ///     vector of _FiniteElement_
      ///         The vector of finite elements.
      std::vector<std::shared_ptr<const dolfin::fem::FiniteElement>> elements() const;
      
      /// Return dofmaps
      ///
      /// *Returns*
      ///     vector of _GenericDofMap_
      ///         The vector of dofmaps.
      std::vector<std::shared_ptr<const dolfin::fem::GenericDofMap>> dofmaps() const;

      /// Return block dofmap
      ///
      /// *Returns*
      ///     _multiphenics::fem::BlockDofMap_
      ///         The block dofmap.
      std::shared_ptr<const multiphenics::fem::BlockDofMap> block_dofmap() const;
      
      /// Return function spaces
      ///
      /// *Returns*
      ///     vector of _FunctionSpace_
      ///         The vector of function spaces.
      std::vector<std::shared_ptr<const dolfin::function::FunctionSpace>> function_spaces() const;

      /// Return dimension of function space
      ///
      /// *Returns*
      ///     std::size_t
      ///         The dimension of the function space.
      std::int64_t dim() const;

      /// Extract subspace for component, *neglecting* restrictions
      ///
      /// *Arguments*
      ///     i (std::size_t)
      ///         Index of the subspace.
      /// *Returns*
      ///     _FunctionSpace_
      ///         The subspace.
      std::shared_ptr<const dolfin::function::FunctionSpace> operator[] (std::size_t i) const;

      /// Extract subspace for component, *neglecting* restrictions
      ///
      /// *Arguments*
      ///     component (std::size_t)
      ///         Index of the subspace.
      /// *Returns*
      ///     _FunctionSpace_
      ///         The subspace.
      std::shared_ptr<const dolfin::function::FunctionSpace> sub(std::size_t component) const;

      /// Extract block subspace for component, possibly considering restrictions
      ///
      /// *Arguments*
      ///     component (std::vector<std::size_t>)
      ///         The component.
      ///
      /// *Returns*
      ///     _BlockFunctionSpace_
      ///         The block subspace.
      std::shared_ptr<BlockFunctionSpace>
      extract_block_sub_space(const std::vector<std::size_t>& component, bool with_restrictions=true) const;

      /// Check whether V is subspace of this, or this itself
      ///
      /// *Arguments*
      ///     V (_FunctionSpace_)
      ///         The space to be tested for inclusion.
      ///
      /// *Returns*
      ///     bool
      ///         True if V is contained or equal to this.
      bool contains(const BlockFunctionSpace& V) const;

      /// Return component w.r.t. to root superspace, i.e.
      ///   W.sub(1).sub(0) == [1, 0].
      ///
      /// *Returns*
      ///     std::vector<std::size_t>
      ///         The component (w.r.t to root superspace).
      std::vector<std::size_t> component() const;

      /// Return informal string representation (pretty-print)
      ///
      /// *Arguments*
      ///     verbose (bool)
      ///         Flag to turn on additional output.
      ///
      /// *Returns*
      ///     std::string
      ///         An informal representation of the function space.
      std::string str(bool verbose) const;
      
      /// Tabulate the coordinates of all dofs on this process. This
      /// function is typically used by preconditioners that require the
      /// spatial coordinates of dofs, for example for re-partitioning or
      /// nullspace computations.
      ///
      /// *Returns*
      ///     std::vector<double>
      ///         The dof coordinates (x0, y0, x1, y1, . . .)
      dolfin::EigenRowArrayXXd tabulate_dof_coordinates() const;

    private:
      // The mesh
      std::shared_ptr<const dolfin::mesh::Mesh> _mesh;

      // The finite elements
      std::vector<std::shared_ptr<const dolfin::fem::FiniteElement>> _elements;

      // The dofmaps
      std::vector<std::shared_ptr<const dolfin::fem::GenericDofMap>> _dofmaps;

      // The restrictions
      std::vector<std::vector<std::shared_ptr<const dolfin::mesh::MeshFunction<bool>>>> _restrictions;
      
      // The subspaces
      std::vector<std::shared_ptr<const dolfin::function::FunctionSpace>> _function_spaces;
      
      // The block dofmap
      std::shared_ptr<multiphenics::fem::BlockDofMap> _block_dofmap;

      // The component w.r.t. to root space
      std::vector<std::size_t> _component;

      // The identifier of root space
      std::size_t _root_space_id;

      // Cache of block subspaces
      typedef std::map<std::vector<std::size_t>,
                       std::shared_ptr<BlockFunctionSpace>> BlockSubpsacesType;
      mutable BlockSubpsacesType _block_subspaces__with_restrictions;
      mutable BlockSubpsacesType _block_subspaces__without_restrictions;

    };
  }
}

#endif

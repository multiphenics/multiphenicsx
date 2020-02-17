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

#include <dolfinx/function/FunctionSpace.h>
#include <multiphenics/fem/BlockDofMap.h>

namespace multiphenics
{
  namespace function
  {
    class BlockFunctionSpace
    {
    public:

      /// Create a block function space from a list of existing function spaces (on the same mesh),
      /// but keep only degrees of freedom in a restriction
      /// @param[in] function_spaces List of existing function spaces
      /// @param[in] restrictions List (of the same size of function_spaces) of list of dofs
      BlockFunctionSpace(std::vector<std::shared_ptr<const dolfinx::function::FunctionSpace>> function_spaces,
                         std::vector<Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>> restrictions);

      /// Copy constructor (deleted)
      BlockFunctionSpace(const BlockFunctionSpace& V) = delete;

      /// Move constructor
      BlockFunctionSpace(BlockFunctionSpace&& V) = default;

      /// Destructor
      ~BlockFunctionSpace() = default;

      /// Assignment operator (deleted)
      BlockFunctionSpace& operator=(const BlockFunctionSpace& V) = delete;

      /// Move assignment operator
      BlockFunctionSpace& operator=(BlockFunctionSpace&& V) = default;

      /// Equality operator
      /// @param[in] V Another block function space.
      bool operator== (const BlockFunctionSpace& V) const;

      /// Inequality operator
      /// @param[in] V Another block function space.
      bool operator!= (const BlockFunctionSpace& V) const;

      /// Return dimension of function space
      /// @return The dimension of the block function space.
      std::int64_t dim() const;

      /// Extract subspace for component, *neglecting* restrictions
      /// @param[in] i Index of the subspace.
      /// @return The subspace.
      std::shared_ptr<const dolfinx::function::FunctionSpace> operator[] (std::size_t i) const;

      /// Extract subspace for component, *neglecting* restrictions
      /// @param[in] i Index of the subspace.
      /// @return The subspace.
      std::shared_ptr<const dolfinx::function::FunctionSpace> sub(std::size_t component) const;

      /// Extract block subspace for component, possibly considering restrictions
      /// @param[in] component Indices of the subspaces.
      /// @return The block subspace.
      std::shared_ptr<BlockFunctionSpace>
      extract_block_sub_space(const std::vector<std::size_t>& component, bool with_restrictions=true) const;

      /// Check whether V is subspace of this, or this itself
      /// @param[in] V The space to be tested for inclusion
      /// @return True if V is contained in or equal to this FunctionSpace
      bool contains(const BlockFunctionSpace& V) const;

      /// Get the component with respect to the root superspace
      /// @return The component with respect to the root superspace , i.e.
      ///         W.sub(1).sub(0) == [1, 0]
      std::vector<std::size_t> component() const;

      /// Tabulate the coordinates of all dofs on this process. This
      /// function is typically used by preconditioners that require the
      /// spatial coordinates of dofs, for example for re-partitioning or
      /// nullspace computations.
      ///
      /// @return The dof coordinates ([x0, y0, z0], [x1, y1, z1], ...)
      Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> tabulate_dof_coordinates() const;

      // The mesh
      std::shared_ptr<const dolfinx::mesh::Mesh> mesh() const;

      // The block dofmap
      std::shared_ptr<multiphenics::fem::BlockDofMap> block_dofmap() const;

      // The subspaces
      std::vector<std::shared_ptr<const dolfinx::function::FunctionSpace>> function_spaces() const;

      // The restrictions
      std::vector<Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>> restrictions() const;

    private:
      // The mesh
      std::shared_ptr<const dolfinx::mesh::Mesh> _mesh;

      // The block dofmap
      std::shared_ptr<multiphenics::fem::BlockDofMap> _block_dofmap;

      // The subspaces
      std::vector<std::shared_ptr<const dolfinx::function::FunctionSpace>> _function_spaces;

      // The restrictions
      std::vector<Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>> _restrictions;

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

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

#ifndef __BLOCK_DOF_MAP_H
#define __BLOCK_DOF_MAP_H

#include <petscvec.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/mesh/MeshFunction.h>

namespace multiphenics
{
  namespace fem
  {
    /// This class handles the mapping of degrees of freedom for block
    /// function spaces, also considering possible restrictions to 
    /// subdomains

    class BlockDofMap : public dolfin::fem::DofMap
    {
    public:

      /// Constructor
      BlockDofMap(std::vector<std::shared_ptr<const dolfin::fem::DofMap>> dofmaps,
                  std::vector<std::vector<std::shared_ptr<const dolfin::mesh::MeshFunction<std::size_t>>>> restrictions,
                  const dolfin::mesh::Mesh& mesh);

    protected:
      
      /// Helper functions for constructor
      BlockDofMap(std::vector<std::shared_ptr<const dolfin::fem::DofMap>> dofmaps,
                  std::vector<std::vector<std::shared_ptr<const dolfin::mesh::MeshFunction<std::size_t>>>> restrictions);
      BlockDofMap(std::vector<std::shared_ptr<const dolfin::fem::DofMap>> dofmaps,
                  std::vector<std::vector<std::shared_ptr<const dolfin::mesh::MeshFunction<std::size_t>>>> restrictions,
                  std::vector<std::shared_ptr<const dolfin::mesh::Mesh>> meshes);
      void _extract_dofs_from_original_dofmaps(
        std::vector<std::shared_ptr<const dolfin::fem::DofMap>> dofmaps,
        std::vector<std::vector<std::shared_ptr<const dolfin::mesh::MeshFunction<std::size_t>>>> restrictions,
        std::vector<std::shared_ptr<const dolfin::mesh::Mesh>> meshes,
        std::vector<std::set<PetscInt>>& owned_dofs,
        std::vector<std::map<PetscInt, bool>>& owned_dofs__to__in_restriction,
        std::vector<std::map<PetscInt, std::set<std::size_t>>>& owned_dofs__to__cell_indices,
        std::vector<std::set<PetscInt>>& unowned_dofs_in_restriction,
        std::vector<std::map<PetscInt, PetscInt>>& unowned_dofs_in_restriction__local_to_global,
        std::vector<std::map<PetscInt, std::set<std::size_t>>>& unowned_dofs_in_restriction__to__cell_indices
      ) const;
      void _assign_owned_dofs_to_block_dofmap(
        std::vector<std::shared_ptr<const dolfin::fem::DofMap>> dofmaps,
        std::vector<std::shared_ptr<const dolfin::mesh::Mesh>> meshes,
        const std::vector<std::set<PetscInt>>& owned_dofs,
        const std::vector<std::map<PetscInt, bool>>& owned_dofs__to__in_restriction,
        const std::vector<std::map<PetscInt, std::set<std::size_t>>>& owned_dofs__to__cell_indices,
        std::int64_t& block_dofmap_local_size,
        std::vector<std::int64_t>& sub_block_dofmap_local_size
      );
      void _prepare_local_to_global_for_unowned_dofs(
        std::vector<std::shared_ptr<const dolfin::fem::DofMap>> dofmaps,
        MPI_Comm comm,
        const std::vector<std::set<PetscInt>>& unowned_dofs_in_restriction,
        const std::vector<std::map<PetscInt, PetscInt>>& unowned_dofs_in_restriction__local_to_global,
        const std::vector<std::map<PetscInt, std::set<std::size_t>>>& unowned_dofs_in_restriction__to__cell_indices,
        std::int64_t block_dofmap_local_size,
        const std::vector<std::int64_t>& sub_block_dofmap_local_size
      );
      void _precompute_views(
        const std::vector<std::shared_ptr<const DofMap>> dofmaps
      );

    public:
      
      /// Copy constructor
      BlockDofMap(const BlockDofMap& dofmap) = delete;
      
    private:
      
      /// Create a view for a component, *considering* restrictions. This is supposed to be used only in this class,
      /// as precomputed views are already available through the view() method.
      BlockDofMap(const BlockDofMap& block_dofmap, std::size_t i);
    
    public:
      
      /// Move constructor
      BlockDofMap(BlockDofMap&& dofmap) = default;
      
      /// Destructor
      virtual ~BlockDofMap() = default;
      
      /// Copy assignment
      BlockDofMap& operator=(const BlockDofMap& dofmap) = delete;

      /// Move assignment
      BlockDofMap& operator=(BlockDofMap&& dofmap) = default;
      
      /// Returns a view of the i-th block
      const BlockDofMap & view(std::size_t i) const;
      
      /// Local-to-global mapping of dofs on a cell
      /// @param[in] cell_index The cell index.
      /// @return  Local-global map for cell (used process-local global
      /// index)
      Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
      cell_dofs(std::size_t cell_index) const;
      
      /// Extract subdofmap component
      /// @param[in] component The component indices
      /// @param[in] mesh The mesh the the dofmap is defined on
      /// @return The dofmap for the component
      dolfin::fem::DofMap
      extract_sub_dofmap(const std::vector<std::size_t>& component,
                         const dolfin::mesh::Mesh& mesh) const;

      /// Create a "collapsed" dofmap (collapses a sub-dofmap)
      /// @param[in] mesh The mesh that the dofmap is defined on
      /// @return The collapsed dofmap
      std::pair<std::unique_ptr<dolfin::fem::DofMap>, std::vector<PetscInt>>
      collapse(const dolfin::mesh::Mesh& mesh) const;

      /// Set dof entries in vector to a specified value. Parallel layout
      /// of vector must be consistent with dof map range. This
      /// function is typically used to construct the null space of a
      /// matrix operator.
      ///
      /// @param[in,out] x The vector to set
      /// @param[in] value The value to set on the vector
      void set(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
               PetscScalar value) const;
               
      // Constructor arguments
      std::vector<std::shared_ptr<const dolfin::fem::DofMap>> dofmaps;
      std::vector<std::vector<std::shared_ptr<const dolfin::mesh::MeshFunction<std::size_t>>>> restrictions;

      // Index Map from local to global
      std::shared_ptr<dolfin::common::IndexMap> index_map;
      
      // Index Map from sub local to sub global
      std::vector<std::shared_ptr<dolfin::common::IndexMap>> sub_index_map;
      
      /// Return informal string representation (pretty-print)
      /// @param[in] verbose Flag to turn on additional output.
      /// @return An informal representation of the function space.
      std::string str(bool verbose) const;
      
      /// Get dofmap array
      Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dof_array() const;
      
      /// Return list of dof indices on this process that belong to mesh
      /// entities of dimension dim
      Eigen::Array<PetscInt, Eigen::Dynamic, 1> dofs(const dolfin::mesh::Mesh& mesh,
                                                     std::size_t dim) const;
      
      /// Accessors
      const std::vector<PetscInt> & block_owned_dofs__local_numbering(std::size_t b) const;
      const std::vector<PetscInt> & block_unowned_dofs__local_numbering(std::size_t b) const;
      const std::vector<PetscInt> & block_owned_dofs__global_numbering(std::size_t b) const;
      const std::vector<PetscInt> & block_unowned_dofs__global_numbering(std::size_t b) const;
      const std::map<PetscInt, PetscInt> & original_to_block(std::size_t b) const;
      const std::map<PetscInt, PetscInt> & block_to_original(std::size_t b) const;
      const std::map<PetscInt, PetscInt> & original_to_sub_block(std::size_t b) const;
      const std::map<PetscInt, PetscInt> & sub_block_to_original(std::size_t b) const;

    protected:
      
      // Cell-local-to-dof map
      std::map<PetscInt, std::vector<PetscInt>> _dofmap;
      std::vector<PetscInt> _empty_vector;
      
      // List of block dofs, for each component, with local numbering
      std::vector<std::vector<PetscInt>> _block_owned_dofs__local;
      std::vector<std::vector<PetscInt>> _block_unowned_dofs__local;
      
      // List of block dofs, for each component, with global numbering
      std::vector<std::vector<PetscInt>> _block_owned_dofs__global;
      std::vector<std::vector<PetscInt>> _block_unowned_dofs__global;
      
      // Local to local (owned and unowned) map from original dofs to block dofs, for each component
      std::vector<std::map<PetscInt, PetscInt>> _original_to_block__local_to_local;
      
      // Local to local (owned and unowned) map from block dofs to original dofs, for each component
      std::vector<std::map<PetscInt, PetscInt>> _block_to_original__local_to_local;
      
      // Local to local (owned and unowned) map from original dofs to block sub dofs, for each component
      std::vector<std::map<PetscInt, PetscInt>> _original_to_sub_block__local_to_local;
      
      // Local to local (owned and unowned) map from block sub dofs to original dofs, for each component
      std::vector<std::map<PetscInt, PetscInt>> _sub_block_to_original__local_to_local;
      
      // Precomputed views
      std::vector<std::shared_ptr<BlockDofMap>> _views;
    };
  }
}

#endif

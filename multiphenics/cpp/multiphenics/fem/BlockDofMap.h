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
                  std::vector<std::vector<std::shared_ptr<const dolfin::mesh::MeshFunction<bool>>>> restrictions,
                  const dolfin::mesh::Mesh& mesh);

    protected:
      
      /// Helper functions for constructor
      BlockDofMap(std::vector<std::shared_ptr<const dolfin::fem::DofMap>> dofmaps,
                  std::vector<std::vector<std::shared_ptr<const dolfin::mesh::MeshFunction<bool>>>> restrictions);
      BlockDofMap(std::vector<std::shared_ptr<const dolfin::fem::DofMap>> dofmaps,
                  std::vector<std::vector<std::shared_ptr<const dolfin::mesh::MeshFunction<bool>>>> restrictions,
                  std::vector<std::shared_ptr<const dolfin::mesh::Mesh>> meshes);
      void _extract_dofs_from_original_dofmaps(
        std::vector<std::shared_ptr<const dolfin::fem::DofMap>> dofmaps,
        std::vector<std::vector<std::shared_ptr<const dolfin::mesh::MeshFunction<bool>>>> restrictions,
        std::vector<std::shared_ptr<const dolfin::mesh::Mesh>> meshes,
        std::vector<std::set<PetscInt>>& owned_dofs,
        std::vector<std::map<PetscInt, bool>>& owned_dofs__to__in_restriction,
        std::vector<std::map<PetscInt, std::set<std::size_t>>>& owned_dofs__to__cell_indices,
        std::vector<std::set<PetscInt>>& unowned_dofs_in_restriction,
        std::vector<std::map<PetscInt, PetscInt>>& unowned_dofs_in_restriction__local_to_global,
        std::vector<std::map<PetscInt, std::set<std::size_t>>>& unowned_dofs_in_restriction__to__cell_indices,
        std::vector<std::set<std::size_t>>& real_dofs
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
      void _store_real_dofs(
        const std::vector<std::shared_ptr<const dolfin::fem::DofMap>> dofmaps,
        const std::vector<std::set<std::size_t>>& real_dofs
      );
      void _precompute_views(
        const std::vector<std::shared_ptr<const DofMap>> dofmaps
      );

    public:
      
      /// Copy constructor
      BlockDofMap(const BlockDofMap& dofmap) = delete;
      
      /// Create a view for a component, *considering* restrictions. This is supposed to be used only in this class,
      /// as precomputed views are already available through the view() method.
      BlockDofMap(const BlockDofMap& block_dofmap, std::size_t i);
    
      /// Move constructor
      BlockDofMap(BlockDofMap&& dofmap) = default;
      
      /// Destructor
      virtual ~BlockDofMap() = default;
      
      /// Assignment
      BlockDofMap& operator=(const BlockDofMap& dofmap) = delete;

      /// Move assignment
      BlockDofMap& operator=(BlockDofMap&& dofmap) = default;
      
      /// Return dofmaps *neglecting* restrictions
      ///
      /// @return   std::vector<const _dolfin::fem::DofMap_>
      ///         The vector of dofmaps *neglecting* restrictions
      std::vector<std::shared_ptr<const dolfin::fem::DofMap>> dofmaps() const;
      
      /// True iff dof map is a view into another map
      ///
      /// @returns bool
      ///         True if the dof map is a sub-dof map (a view into
      ///         another map).
      bool is_view() const;
      
      /// Returns a view of the i-th block
      const BlockDofMap & view(std::size_t i) const;

      /// Return the dimension of the global finite element function
      /// space. Use index_map()->size() to get the local dimension.
      ///
      /// @returns std::int64_t
      ///         The dimension of the global finite element function space.
      std::int64_t global_dimension() const;
      
      /// Return the dimension of the local finite element function
      /// space on a cell
      ///
      /// @param      cell_index (std::size_t)
      ///         Index of cell
      ///
      /// @return     std::size_t
      ///         Dimension of the local finite element function space.
      std::size_t num_element_dofs(std::size_t cell_index) const;
      
      /// Return the maximum dimension of the local finite element
      /// function space
      ///
      /// @return     std::size_t
      ///         Maximum dimension of the local finite element function
      ///         space.
      std::size_t max_element_dofs() const;
      
      /// Return the number of dofs for a given entity dimension.
      /// Note that, in contrast to standard DofMaps, this return the *maximum*
      /// of number of entity dofs, because in case of restrictions entities
      /// of the same dimension at different locations may have variable number
      /// of dofs.
      ///
      /// @param     entity_dim (std::size_t)
      ///         Entity dimension
      ///
      /// @return     std::size_t
      ///         Number of dofs associated with given entity dimension
      virtual std::size_t num_entity_dofs(std::size_t entity_dim) const;

      /// Return the number of dofs for the closure of an entity of given dimension
      /// Note that, in contrast to standard DofMaps, this return the *maximum*
      /// of number of entity dofs, because in case of restrictions entities
      /// of the same dimension at different locations may have variable number
      /// of dofs.
      ///
      /// @param     entity_dim (std::size_t)
      ///         Entity dimension
      ///
      /// @return     std::size_t
      ///         Number of dofs associated with closure of given entity dimension
      virtual std::size_t num_entity_closure_dofs(std::size_t entity_dim) const;

      /// Return the ownership range (dofs in this range are owned by
      /// this process)
      ///
      /// @return   std::array<std::size_t, 2>
      ///         The ownership range.
      std::array<std::int64_t, 2> ownership_range() const;

      /// Return map from all shared nodes to the sharing processes (not
      /// including the current process) that share it.
      ///
      /// @return     std::unordered_map<int, std::vector<int>>
      ///         The map from dofs to list of processes
      const std::unordered_map<int, std::vector<int>>& shared_nodes() const;

      /// Return set of processes that share dofs with this process
      ///
      /// @return     std::set<int>
      ///         The set of processes
      const std::set<int>& neighbours() const;
      
      /// Local-to-global mapping of dofs on a cell
      ///
      /// @param     cell_index (std::size_t)
      ///         The cell index.
      ///
      /// @return         Eigen::Map<const Eigen::Array<PetscInt,
      /// Eigen::Dynamic, 1>>
      ///         Local-to-global mapping of dofs.
      Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
      cell_dofs(std::size_t cell_index) const;
      
      /// Return the dof indices associated with entities of given dimension and
      /// entity indices
      Eigen::Array<PetscInt, Eigen::Dynamic, 1>
      entity_dofs(const dolfin::mesh::Mesh& mesh, std::size_t entity_dim,
                  const std::vector<std::size_t>& entity_indices) const;

      /// Return the dof indices associated with all entities of given dimension
      Eigen::Array<PetscInt, Eigen::Dynamic, 1>
      entity_dofs(const dolfin::mesh::Mesh& mesh, std::size_t entity_dim) const;

      /// Tabulate local-local mapping of dofs on entity (dim, local_entity)
      ///
      /// @param    element_dofs (std::size_t)
      ///         Degrees of freedom on a single element.
      /// @param   entity_dim (std::size_t)
      ///         The entity dimension.
      /// @param    cell_entity_index (std::size_t)
      ///         The local entity index on the cell.
      Eigen::Array<int, Eigen::Dynamic, 1>
      tabulate_entity_dofs(std::size_t entity_dim,
                           std::size_t cell_entity_index) const;

      /// Tabulate local-local closure dofs on entity of cell
      ///
      /// @param   entity_dim (std::size_t)
      ///         The entity dimension.
      /// @param    cell_entity_index (std::size_t)
      ///         The local entity index on the cell.
      /// @return     Eigen::Array<int, Eigen::Dynamic, 1>
      ///         Degrees of freedom on a single element.
      Eigen::Array<int, Eigen::Dynamic, 1>
      tabulate_entity_closure_dofs(std::size_t entity_dim,
                                   std::size_t cell_entity_index) const;

      /// Tabulate globally supported dofs
      Eigen::Array<std::size_t, Eigen::Dynamic, 1> tabulate_global_dofs() const;
      
      /// Extract subdofmap component
      ///
      /// @param     component (std::vector<std::size_t>)
      ///         The component.
      /// @param     mesh (_dolfin::mesh::Mesh_)
      ///         The mesh.
      ///
      /// @return     DofMap
      ///         The subdofmap component.
      std::unique_ptr<dolfin::fem::DofMap>
      extract_sub_dofmap(const std::vector<std::size_t>& component,
                         const dolfin::mesh::Mesh& mesh) const;

      /// Create a "collapsed" dofmap (collapses a sub-dofmap)
      ///
      /// @param     collapsed_map (std::unordered_map<std::size_t, std::size_t>)
      ///         The "collapsed" map.
      /// @param     mesh (_dolfin::mesh::Mesh_)
      ///         The mesh.
      ///
      /// @return    DofMap
      ///         The collapsed dofmap.
      std::pair<std::shared_ptr<dolfin::fem::DofMap>,
                std::unordered_map<std::size_t, std::size_t>>
      collapse(const dolfin::mesh::Mesh& mesh) const;

      /// Return list of dof indices on this process that belong to mesh
      /// entities of dimension dim
      Eigen::Array<PetscInt, Eigen::Dynamic, 1> dofs(const dolfin::mesh::Mesh& mesh,
                                                     std::size_t dim) const;

      /// Set dof entries in vector to a specified value. Parallel layout
      /// of vector must be consistent with dof map range. This
      /// function is typically used to construct the null space of a
      /// matrix operator.
      ///
      /// @param  x (Vec)
      ///         The vector to set.
      /// @param  value (PetscScalar)
      ///         The value to set.
      void set(Vec x, double value) const;

      /// Return the map from local to global (const access)
      std::shared_ptr<const dolfin::common::IndexMap> index_map() const;
      
      /// Return the map from sub local to sub global (const access)
      std::shared_ptr<const dolfin::common::IndexMap> sub_index_map(std::size_t b) const;
      
      /// Return the block size for dof maps with components, typically
      /// used for vector valued functions.
      int block_size() const;

      /// Return informal string representation (pretty-print)
      ///
      /// @param     verbose (bool)
      ///         Flag to turn on additional output.
      ///
      /// @return    std::string
      ///         An informal representation of the function space.
      std::string str(bool verbose) const;
      
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
      
      // Constructor arguments
      std::vector<std::shared_ptr<const dolfin::fem::DofMap>> _constructor_dofmaps;
      std::vector<std::vector<std::shared_ptr<const dolfin::mesh::MeshFunction<bool>>>> _constructor_restrictions;

      // Cell-local-to-dof map
      std::map<PetscInt, std::vector<PetscInt>> _dofmap;
      std::vector<PetscInt> _empty_vector;
      
      // Maximum number of elements associated to a cell in _dofmap
      std::size_t _max_element_dofs;
      
      // Maximum number of dofs associated with each entity dimension
      std::map<std::size_t, std::size_t> _num_entity_dofs;
      
      // Maximum number of dofs associated with closure of an entity for each dimension
      std::map<std::size_t, std::size_t> _num_entity_closure_dofs;
      
      // Real dofs, with local numbering
      std::vector<std::size_t> _real_dofs__local;
      
      // Index Map from local to global
      std::shared_ptr<dolfin::common::IndexMap> _index_map;
      
      // Index Map from sub local to sub global
      std::vector<std::shared_ptr<dolfin::common::IndexMap>> _sub_index_map;
      
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

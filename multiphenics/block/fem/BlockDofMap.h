// Copyright (C) 2016-2017 by the block_ext authors
//
// This file is part of block_ext.
//
// block_ext is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// block_ext is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with block_ext. If not, see <http://www.gnu.org/licenses/>.
//

#ifndef __BLOCK_DOF_MAP_H
#define __BLOCK_DOF_MAP_H

#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/mesh/MeshFunction.h>

namespace dolfin
{

  /// This class handles the mapping of degrees of freedom for block
  /// function spaces, also considering possible restrictions to 
  /// subdomains

  class BlockDofMap : public GenericDofMap
  {
  public:

    /// Constructor
    BlockDofMap(std::vector<std::shared_ptr<const GenericDofMap>> dofmaps,
                std::vector<std::vector<std::shared_ptr<const MeshFunction<bool>>>> restrictions,
                const Mesh& mesh);

  private:
    // Copy constructor
    BlockDofMap(const BlockDofMap& block_dofmap);

  public:
    /// Destructor
    virtual ~BlockDofMap();
    
    /// Return dofmaps *neglecting* restrictions
    ///
    /// *Returns*
    ///     vector of _GenericDofMap_
    ///         The vector of dofmaps *neglecting* restrictions
    const std::vector<std::shared_ptr<const GenericDofMap>> & dofmaps() const;
    
    /// True if dof map is a view into another map (is a sub-dofmap).
    /// BlockDofMap does not allow views, so the value will always be False.
    bool is_view() const;

    /// Return the dimension of the global finite element function
    /// space
    std::size_t global_dimension() const;
    
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
    /// *Arguments*
    ///     entity_dim (std::size_t)
    ///         Entity dimension
    ///
    /// *Returns*
    ///     std::size_t
    ///         Number of dofs associated with closure of an entity of given dimension
    virtual std::size_t num_entity_closure_dofs(std::size_t entity_dim) const;

    /// Return number of facet dofs.
    /// Note that, in contrast to standard DofMaps, this return the *maximum*
    /// of number of entity dofs, because in case of restrictions entities
    /// of the same dimension at different locations may have variable number
    /// of dofs.
    ///
    /// @return     std::size_t
    ///         The number of facet dofs.
    std::size_t num_facet_dofs() const;

    /// Return the ownership range (dofs in this range are owned by
    /// this process)
    std::pair<std::size_t, std::size_t> ownership_range() const;

    /// Return map from nonlocal-dofs (that appear in local dof map)
    /// to owning process
    const std::vector<int>& off_process_owner() const;
    
    /// Return map from all shared nodes to the sharing processes (not
    /// including the current process) that share it.
    ///
    /// @return     std::unordered_map<std::size_t, std::vector<unsigned int>>
    ///         The map from dofs to list of processes
    const std::unordered_map<int, std::vector<int>>& shared_nodes() const;

    /// Return set of processes that share dofs with this process
    ///
    /// @return     std::set<int>
    ///         The set of processes
    const std::set<int>& neighbours() const;
    
    /// Clear any data required to build sub-dofmaps (this is to
    /// reduce memory use)
    void clear_sub_map_data();
    
    /// Local-to-global mapping of dofs on a cell
    ///
    /// @param     cell_index (std::size_t)
    ///         The cell index.
    ///
    /// @return     ArrayView<const dolfin::la_index>
    ///         Local-to-global mapping of dofs.
    ArrayView<const dolfin::la_index> cell_dofs(std::size_t cell_index) const;
    
    /// Return the dof indices associated with entities of given dimension and entity indices
    ///
    /// *Arguments*
    ///     entity_dim (std::size_t)
    ///         Entity dimension.
    ///     entity_indices (std::vector<dolfin::la_index>&)
    ///         Entity indices to get dofs for.
    /// *Returns*
    ///     std::vector<dolfin::la_index>
    ///         Dof indices associated with selected entities.
    std::vector<dolfin::la_index>
      entity_dofs(const Mesh& mesh, std::size_t entity_dim,
                  const std::vector<std::size_t> & entity_indices) const;

    /// Return the dof indices associated with all entities of given dimension
    ///
    /// *Arguments*
    ///     entity_dim (std::size_t)
    ///         Entity dimension.
    /// *Returns*
    ///     std::vector<dolfin::la_index>
    ///         Dof indices associated with selected entities.
    std::vector<dolfin::la_index>
      entity_dofs(const Mesh& mesh, std::size_t entity_dim) const;

    /// Return the dof indices associated with the closure of entities of
    /// given dimension and entity indices
    ///
    /// *Arguments*
    ///     entity_dim (std::size_t)
    ///         Entity dimension.
    ///     entity_indices (std::vector<dolfin::la_index>&)
    ///         Entity indices to get dofs for.
    /// *Returns*
    ///     std::vector<dolfin::la_index>
    ///         Dof indices associated with selected entities and their closure.
    std::vector<dolfin::la_index>
      entity_closure_dofs(const Mesh& mesh, std::size_t entity_dim,
                          const std::vector<std::size_t> & entity_indices) const;

    /// Return the dof indices associated with the closure of all entities of
    /// given dimension
    ///
    /// @param  mesh (Mesh)
    ///         Mesh
    /// @param  entity_dim (std::size_t)
    ///         Entity dimension.
    /// @return  std::vector<dolfin::la_index>
    ///         Dof indices associated with selected entities and their closure.
    std::vector<dolfin::la_index>
      entity_closure_dofs(const Mesh& mesh, std::size_t entity_dim) const;
      
    /// Tabulate local-local facet dofs
    ///
    /// @param    element_dofs (std::size_t)
    ///         Degrees of freedom on a single element.
    /// @param    cell_facet_index (std::size_t)
    ///         The local facet index on the cell.
    void tabulate_facet_dofs(std::vector<std::size_t>& element_dofs,
                             std::size_t cell_facet_index) const;

    /// Tabulate local-local mapping of dofs on entity (dim, local_entity)
    ///
    /// @param    element_dofs (std::size_t)
    ///         Degrees of freedom on a single element.
    /// @param   entity_dim (std::size_t)
    ///         The entity dimension.
    /// @param    cell_entity_index (std::size_t)
    ///         The local entity index on the cell.
    void tabulate_entity_dofs(std::vector<std::size_t>& element_dofs,
                              std::size_t entity_dim, std::size_t cell_entity_index) const;

    /// Tabulate local-local mapping of dofs on closure of entity (dim, local_entity)
    ///
    /// @param   element_dofs (std::size_t)
    ///         Degrees of freedom on a single element.
    /// @param   entity_dim (std::size_t)
    ///         The entity dimension.
    /// @param    cell_entity_index (std::size_t)
    ///         The local entity index on the cell.
    void tabulate_entity_closure_dofs(std::vector<std::size_t>& element_dofs,
                                      std::size_t entity_dim, std::size_t cell_entity_index) const;

    /// Tabulate globally supported dofs
    ///
    /// @param    element_dofs (std::size_t)
    ///         Degrees of freedom.
    void tabulate_global_dofs(std::vector<std::size_t>& element_dofs) const;
    
    /// Create a copy of the dof map
    ///
    /// @return     DofMap
    ///         The Dofmap copy.
    std::shared_ptr<GenericDofMap> copy() const;

    /// Create a copy of the dof map on a new mesh
    ///
    /// @param     new_mesh (_Mesh_)
    ///         The new mesh to create the dof map on.
    ///
    ///  @return    DofMap
    ///         The new Dofmap copy.
    std::shared_ptr<GenericDofMap> create(const Mesh& new_mesh) const;

    /// Extract subdofmap component
    ///
    /// @param     component (std::vector<std::size_t>)
    ///         The component.
    /// @param     mesh (_Mesh_)
    ///         The mesh.
    ///
    /// @return     DofMap
    ///         The subdofmap component.
    std::shared_ptr<GenericDofMap>
      extract_sub_dofmap(const std::vector<std::size_t>& component,
                         const Mesh& mesh) const;

    /// Create a "collapsed" dofmap (collapses a sub-dofmap)
    ///
    /// @param     collapsed_map (std::unordered_map<std::size_t, std::size_t>)
    ///         The "collapsed" map.
    /// @param     mesh (_Mesh_)
    ///         The mesh.
    ///
    /// @return    DofMap
    ///         The collapsed dofmap.
    std::shared_ptr<GenericDofMap>
      collapse(std::unordered_map<std::size_t, std::size_t>&
               collapsed_map, const Mesh& mesh) const;

    /// Return list of dof indices on this process that belong to mesh
    /// entities of dimension dim
    std::vector<dolfin::la_index> dofs(const Mesh& mesh,
                                       std::size_t dim) const;

    std::vector<dolfin::la_index> dofs() const;

    /// Set dof entries in vector to a specified value. Parallel layout
    /// of vector must be consistent with dof map range. This
    /// function is typically used to construct the null space of a
    /// matrix operator.
    ///
    /// @param  x (GenericVector)
    ///         The vector to set.
    /// @param  value (double)
    ///         The value to set.
    void set(GenericVector& x, double value) const;

    /// Return the map from local to global (const access)
    std::shared_ptr<const IndexMap> index_map() const;
    
    /// Return the map from sub local to sub global (const access)
    std::shared_ptr<const IndexMap> sub_index_map(std::size_t b) const;
    
    /// Return the block size for dof maps with components, typically
    /// used for vector valued functions.
    int block_size() const;

    /// Compute the map from local (this process) dof indices to
    /// global dof indices.
    ///
    /// @param     local_to_global_map (_std::vector<std::size_t>_)
    ///         The local-to-global map to fill.
    void tabulate_local_to_global_dofs(std::vector<std::size_t>& local_to_global_map) const;

    /// Return global dof index for a given local (process) dof index
    ///
    /// @param     local_index (int)
    ///         The local local index.
    ///
    /// @return     std::size_t
    ///         The global dof index.
    std::size_t local_to_global_index(int local_index) const;

    /// Return indices of dofs which are owned by other processes
    const std::vector<std::size_t>& local_to_global_unowned() const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;
    
    /// Accessors
    const std::vector<dolfin::la_index> & block_owned_dofs__local_numbering(std::size_t b) const;
    const std::vector<dolfin::la_index> & block_unowned_dofs__local_numbering(std::size_t b) const;
    const std::vector<dolfin::la_index> & block_owned_dofs__global_numbering(std::size_t b) const;
    const std::vector<dolfin::la_index> & block_unowned_dofs__global_numbering(std::size_t b) const;
    const std::map<dolfin::la_index, dolfin::la_index> & original_to_block(std::size_t b) const;
    const std::map<dolfin::la_index, dolfin::la_index> & block_to_original(std::size_t b) const;
    const std::map<dolfin::la_index, dolfin::la_index> & original_to_sub_block(std::size_t b) const;
    const std::map<dolfin::la_index, dolfin::la_index> & sub_block_to_original(std::size_t b) const;

  private:
    
    // Constructor arguments
    std::vector<std::shared_ptr<const GenericDofMap>> _constructor_dofmaps;
    std::vector<std::vector<std::shared_ptr<const MeshFunction<bool>>>> _constructor_restrictions;

    // Cell-local-to-dof map
    std::map<dolfin::la_index, std::vector<dolfin::la_index>> _dofmap;
    std::vector<dolfin::la_index> _empty_vector;
    
    // Maximum number of elements associated to a cell in _dofmap
    std::size_t _max_element_dofs;
    
    // Maximum number of dofs associated with each entity dimension
    std::map<std::size_t, std::size_t> _num_entity_dofs;
    
    // Maximum number of dofs associated with closure of an entity for each dimension
    std::map<std::size_t, std::size_t> _num_entity_closure_dofs;
    
    // Maximum number of facet dofs
    std::size_t _num_facet_dofs;
    
    // Real dofs, with local numbering
    std::vector<std::size_t> _real_dofs__local;
    
    // Index Map from local to global
    std::shared_ptr<IndexMap> _index_map;
    
    // Index Map from sub local to sub global
    std::vector<std::shared_ptr<IndexMap>> _sub_index_map;
    
    // List of block dofs, for each component, with local numbering
    std::vector<std::vector<dolfin::la_index>> _block_owned_dofs__local;
    std::vector<std::vector<dolfin::la_index>> _block_unowned_dofs__local;
    
    // List of block dofs, for each component, with global numbering
    std::vector<std::vector<dolfin::la_index>> _block_owned_dofs__global;
    std::vector<std::vector<dolfin::la_index>> _block_unowned_dofs__global;
    
    // Local to local (owned and unowned) map from original dofs to block dofs, for each component
    std::vector<std::map<dolfin::la_index, dolfin::la_index>> _original_to_block__local_to_local;
    
    // Local to local (owned and unowned) map from block dofs to original dofs (pair of component and local dof)
    std::vector<std::map<dolfin::la_index, dolfin::la_index>> _block_to_original__local_to_local;
    
    // Local to local (owned and unowned) map from original dofs to block sub dofs, for each component
    std::vector<std::map<dolfin::la_index, dolfin::la_index>> _original_to_sub_block__local_to_local;
    
    // Local to local (owned and unowned) map from block sub dofs to original dofs, for each component
    std::vector<std::map<dolfin::la_index, dolfin::la_index>> _sub_block_to_original__local_to_local;
  };

}

#endif

// Copyright (C) 2016-2018 by the multiphenics authors
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

#include <dolfin/fem/DofMap.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <multiphenics/fem/BlockDofMap.h>
#include <multiphenics/log/log.h>

using namespace dolfin;
using namespace multiphenics;

//-----------------------------------------------------------------------------
BlockDofMap::BlockDofMap(std::vector<std::shared_ptr<const GenericDofMap>> dofmaps,
                         std::vector<std::vector<std::shared_ptr<const MeshFunction<bool>>>> restrictions):
  _constructor_dofmaps(dofmaps),
  _constructor_restrictions(restrictions),
  _max_element_dofs(0),
  _block_owned_dofs__local(dofmaps.size()),
  _block_unowned_dofs__local(dofmaps.size()),
  _block_owned_dofs__global(dofmaps.size()),
  _block_unowned_dofs__global(dofmaps.size()),
  _original_to_block__local_to_local(dofmaps.size()),
  _block_to_original__local_to_local(dofmaps.size()),
  _original_to_sub_block__local_to_local(dofmaps.size()),
  _sub_block_to_original__local_to_local(dofmaps.size())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockDofMap::BlockDofMap(std::vector<std::shared_ptr<const GenericDofMap>> dofmaps,
                         std::vector<std::vector<std::shared_ptr<const MeshFunction<bool>>>> restrictions,
                         const Mesh& mesh):
  BlockDofMap(dofmaps, restrictions, std::vector<std::shared_ptr<const dolfin::Mesh>>(dofmaps.size(), std::shared_ptr<const Mesh>(&mesh, [](const Mesh*){})))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockDofMap::BlockDofMap(std::vector<std::shared_ptr<const GenericDofMap>> dofmaps,
                         std::vector<std::vector<std::shared_ptr<const MeshFunction<bool>>>> restrictions,
                         std::vector<std::shared_ptr<const dolfin::Mesh>> meshes):
  BlockDofMap(dofmaps, restrictions)
{
  // A. EXTRACT DOFS FROM ORIGINAL DOFMAPS
  std::vector<std::set<dolfin::la_index>> owned_dofs(dofmaps.size());
  std::vector<std::map<dolfin::la_index, bool>> owned_dofs__to__in_restriction(dofmaps.size());
  std::vector<std::map<dolfin::la_index, std::set<dolfin::la_index>>> owned_dofs__to__cell_indices(dofmaps.size());
  std::vector<std::set<dolfin::la_index>> unowned_dofs_in_restriction(dofmaps.size());
  std::vector<std::map<dolfin::la_index, dolfin::la_index>> unowned_dofs_in_restriction__local_to_global(dofmaps.size());
  std::vector<std::map<dolfin::la_index, std::set<dolfin::la_index>>> unowned_dofs_in_restriction__to__cell_indices(dofmaps.size());
  std::vector<std::set<std::size_t>> real_dofs(dofmaps.size());
  _extract_dofs_from_original_dofmaps(
    dofmaps, restrictions, meshes,
    owned_dofs, owned_dofs__to__in_restriction, owned_dofs__to__cell_indices,
    unowned_dofs_in_restriction, unowned_dofs_in_restriction__local_to_global, unowned_dofs_in_restriction__to__cell_indices,
    real_dofs
  );
  
  // B. ASSIGN OWNED DOFS TO BLOCK DOF MAP
  dolfin::la_index block_dofmap_local_size = 0;
  std::vector<dolfin::la_index> sub_block_dofmap_local_size;
  _assign_owned_dofs_to_block_dofmap(
    dofmaps, meshes,
    owned_dofs, owned_dofs__to__in_restriction, owned_dofs__to__cell_indices,
    block_dofmap_local_size, sub_block_dofmap_local_size
  );
  
  // C. PREPARE LOCAL TO GLOBAL MAP OF BLOCK DOF MAP FOR UNOWNED DOFS
  dolfin_assert(std::all_of(meshes.begin(), meshes.end(), [](std::shared_ptr<const dolfin::Mesh> mesh){return mesh->mpi_comm() == meshes[0]->mpi_comm();}));
  _prepare_local_to_global_for_unowned_dofs(
    dofmaps, meshes[0]->mpi_comm(),
    unowned_dofs_in_restriction, unowned_dofs_in_restriction__local_to_global, unowned_dofs_in_restriction__to__cell_indices,
    block_dofmap_local_size, sub_block_dofmap_local_size
  );
  
  // D. STORE REAL DOFS
  _store_real_dofs(dofmaps, real_dofs);
}
//-----------------------------------------------------------------------------
void BlockDofMap::_extract_dofs_from_original_dofmaps(
  std::vector<std::shared_ptr<const GenericDofMap>> dofmaps,
  std::vector<std::vector<std::shared_ptr<const MeshFunction<bool>>>> restrictions,
  std::vector<std::shared_ptr<const dolfin::Mesh>> meshes,
  std::vector<std::set<dolfin::la_index>>& owned_dofs,
  std::vector<std::map<dolfin::la_index, bool>>& owned_dofs__to__in_restriction,
  std::vector<std::map<dolfin::la_index, std::set<dolfin::la_index>>>& owned_dofs__to__cell_indices,
  std::vector<std::set<dolfin::la_index>>& unowned_dofs_in_restriction,
  std::vector<std::map<dolfin::la_index, dolfin::la_index>>& unowned_dofs_in_restriction__local_to_global,
  std::vector<std::map<dolfin::la_index, std::set<dolfin::la_index>>>& unowned_dofs_in_restriction__to__cell_indices,
  std::vector<std::set<std::size_t>>& real_dofs
) const
{
  // Parts of this code have been adapted from
  //    PeriodicBoundaryComputation::compute_periodic_pairs
  //    DofMap::entity_dofs
  
  dolfin_assert(dofmaps.size() == restrictions.size());
  dolfin_assert(dofmaps.size() == meshes.size());
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
  {
    std::shared_ptr<const DofMap> dofmap = std::dynamic_pointer_cast<const DofMap>(dofmaps[i]);
    const std::vector<std::shared_ptr<const MeshFunction<bool>>>& restriction = restrictions[i];
    const Mesh& mesh = *meshes[i];
    
    // Mesh dimension
    const std::size_t D = mesh.topology().dim();
    
    // Consistency check
    if (restriction.size() != 0 && restriction.size() != D + 1)
    {
      multiphenics_error("BlockDofMap.cpp",
                         "initialize block dof map",
                         "Invalid length of restriction array");
    }
    
    // Arrays used for mapping coordinates
    std::vector<double> x(mesh.geometry().dim());

    // Wrap x (Array view of x)
    Array<double> _x(x.size(), x.data());
    
    // Local size
    const dolfin::la_index dofmap_local_size = dofmap->ownership_range().second - dofmap->ownership_range().first;
    
    // Retrieve Real dofs on the current dofmap
    std::vector<std::size_t> real_dofs_i;
    dofmap->tabulate_global_dofs(real_dofs_i);
    real_dofs[i] = std::set<std::size_t>(real_dofs_i.begin(), real_dofs_i.end());
    
    for (std::size_t d = 0; d <= D; ++d)
    {
      if (restriction.size() == D + 1)
      {
        if (restriction[d]->dim() != d)
        {
          multiphenics_error("BlockDofMap.cpp",
                             "initialize block dof map",
                             "Invalid dimension of restriction mesh function");
        }
      }
      
      // In the case of Real dofs, dofs_per_entity > 0 only when D == d, therefore
      // in the following, when calling
      //    dofmap->tabulate_entity_dofs()       for dimension D
      // we are also obtaining the list of Real dofs, which will be handled separately.
      const std::size_t dofs_per_entity = dofmap->num_entity_dofs(d);
      if (dofs_per_entity > 0)
      {
        mesh.init(d);
        mesh.init(d, D);
        
        std::vector<std::size_t> local_to_local_map(dofs_per_entity);
        for (MeshEntityIterator e(mesh, d); !e.end(); ++e)
        {
          // Check if the mesh entity is in restriction
          bool in_restriction;
          if (restriction[d])
          {
            in_restriction = restriction[d]->operator[](e->index());
          }
          else
          {
            // No restriction provided, keep the whole domain
            in_restriction = true;
          }
          
          // Get ids of all cells connected to the entity
          dolfin_assert(e->num_entities(D) > 0);
          std::set<dolfin::la_index> cell_indices;
          for (std::size_t c(0); c < e->num_entities(D); ++c)
          {
            dolfin::la_index cell_index = e->entities(D)[c];
            cell_indices.insert(cell_index);
          }
                    
          // Get the first cell connected to the entity
          const Cell cell(mesh, *cell_indices.begin());

          // Find local entity number
          std::size_t local_entity_ind = 0;
          for (std::size_t local_i = 0; local_i < cell.num_entities(d); ++local_i)
          {
            if (cell.entities(d)[local_i] == e->index())
            {
              local_entity_ind = local_i;
              break;
            }
          }

          // Get all cell dofs
          const auto cell_dof_list = dofmap->cell_dofs(cell.index());

          // Tabulate local to local map of dofs on local entity
          dofmap->tabulate_entity_dofs(local_to_local_map,
                                       d, local_entity_ind);

          // Fill local dofs for the entity
          for (std::size_t local_dof = 0; local_dof < dofs_per_entity; ++local_dof)
          {
            dolfin::la_index cell_dof = cell_dof_list[local_to_local_map[local_dof]];
            if (real_dofs[i].count(cell_dof) == 0)
            {
              if (cell_dof < dofmap_local_size) 
              {
                owned_dofs[i].insert(cell_dof);
                if (owned_dofs__to__in_restriction[i].count(cell_dof) == 0)
                  owned_dofs__to__in_restriction[i][cell_dof] = in_restriction;
                else
                  owned_dofs__to__in_restriction[i][cell_dof] = owned_dofs__to__in_restriction[i][cell_dof] or in_restriction;
                if (in_restriction)
                  for (auto c : cell_indices)
                    owned_dofs__to__cell_indices[i][cell_dof].insert(c);
              }
              else 
              {
                if (in_restriction)
                {
                  unowned_dofs_in_restriction[i].insert(cell_dof);
                  std::size_t cell_global_dof = dofmap->index_map()->local_to_global(cell_dof);
                  unowned_dofs_in_restriction__local_to_global[i][cell_dof] = cell_global_dof;
                  for (auto c : cell_indices)
                    unowned_dofs_in_restriction__to__cell_indices[i][cell_dof].insert(c);
                }
              }
            }
            else
            {
              // Real dofs will be handled separately below, at the end of the loop over mesh dimensions
            }
          } 
        }
      }
    }
    
    // Handle Real dofs. They are connected to every cell, regardless of restriction (which is ignored).
    for (auto real_dof : real_dofs[i])
    {
      if (real_dof < static_cast<std::size_t>(dofmap_local_size))
      {
        owned_dofs[i].insert(real_dof);
        owned_dofs__to__in_restriction[i][real_dof] = true;
        for (CellIterator c(mesh); !c.end(); ++c)
          owned_dofs__to__cell_indices[i][real_dof].insert(c->index());
      }
      else 
      {
        unowned_dofs_in_restriction[i].insert(real_dof);
        std::size_t real_global_dof = dofmap->index_map()->local_to_global(real_dof);
        unowned_dofs_in_restriction__local_to_global[i][real_dof] = real_global_dof;
        for (CellIterator c(mesh); !c.end(); ++c)
          unowned_dofs_in_restriction__to__cell_indices[i][real_dof].insert(c->index());
      }
    }
  }
}
//-----------------------------------------------------------------------------
void BlockDofMap::_assign_owned_dofs_to_block_dofmap(
  std::vector<std::shared_ptr<const GenericDofMap>> dofmaps,
      std::vector<std::shared_ptr<const dolfin::Mesh>> meshes,
  const std::vector<std::set<dolfin::la_index>>& owned_dofs,
  const std::vector<std::map<dolfin::la_index, bool>>& owned_dofs__to__in_restriction,
  const std::vector<std::map<dolfin::la_index, std::set<dolfin::la_index>>>& owned_dofs__to__cell_indices,
  dolfin::la_index& block_dofmap_local_size,
  std::vector<dolfin::la_index>& sub_block_dofmap_local_size
)
{
  // Fill in private attributes related to local indices of owned dofs
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
  {
    dolfin::la_index block_i_local_size = 0;
    for (auto original_dof : owned_dofs[i])
    {
      bool in_restriction = owned_dofs__to__in_restriction[i].at(original_dof);
      if (in_restriction) 
      {
        _block_owned_dofs__local[i].push_back(block_dofmap_local_size);
        _original_to_block__local_to_local[i][original_dof] = block_dofmap_local_size;
        _block_to_original__local_to_local[i][block_dofmap_local_size] = original_dof;
        _original_to_sub_block__local_to_local[i][original_dof] = block_i_local_size;
        _sub_block_to_original__local_to_local[i][block_i_local_size] = original_dof;
        for (dolfin::la_index cell_index : owned_dofs__to__cell_indices[i].at(original_dof))
          _dofmap[cell_index].push_back(block_dofmap_local_size);
        // Increment counters
        block_dofmap_local_size++;
        block_i_local_size++;
      }
    }
    sub_block_dofmap_local_size.push_back(block_i_local_size);
  }
  
  // Communicator
  dolfin_assert(std::all_of(meshes.begin(), meshes.end(), [](std::shared_ptr<const dolfin::Mesh> mesh){return mesh->mpi_comm() == meshes[0]->mpi_comm();}));
  MPI_Comm comm = meshes[0]->mpi_comm();
  
  // Prepare index map
  _index_map.reset(new IndexMap(comm, block_dofmap_local_size, 1));
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
  {
    auto index_map_i = std::make_shared<IndexMap>(comm, sub_block_dofmap_local_size[i], 1);
    _sub_index_map.push_back(index_map_i);
  }
  
  // Fill in private attributes related to global indices of owned dofs
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
  {
    for (auto local_dof : _block_owned_dofs__local[i])
    {
      _block_owned_dofs__global[i].push_back(_index_map->local_to_global(local_dof));
    }
  }
  
  // Fill in the maximum number of elements associated to a cell in _dofmap
  for (auto & cell_to_dofs: _dofmap)
    if (cell_to_dofs.second.size() > _max_element_dofs)
      _max_element_dofs = cell_to_dofs.second.size();
  _max_element_dofs = MPI::max(comm, _max_element_dofs);
  
  // Fill in 
  // * maximum number of dofs associated with each entity dimension
  // * maximum number of dofs associated with closure of an entity for each dimension
  // * maximum number of facet dofs
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
  {
    std::shared_ptr<const DofMap> dofmap = std::dynamic_pointer_cast<const DofMap>(dofmaps[i]);
    const std::size_t D = meshes[i]->topology().dim();
    for (std::size_t d = 0; d <= D; ++d)
    {
      _num_entity_dofs[d] += dofmap->num_entity_dofs(d);
      _num_entity_closure_dofs[d] += dofmap->num_entity_closure_dofs(d);
    }
    _num_facet_dofs += dofmap->num_facet_dofs();
  }
}
//-----------------------------------------------------------------------------
void BlockDofMap::_prepare_local_to_global_for_unowned_dofs(
  std::vector<std::shared_ptr<const GenericDofMap>> dofmaps,
  MPI_Comm comm,
  const std::vector<std::set<dolfin::la_index>>& unowned_dofs_in_restriction,
  const std::vector<std::map<dolfin::la_index, dolfin::la_index>>& unowned_dofs_in_restriction__local_to_global,
  const std::vector<std::map<dolfin::la_index, std::set<dolfin::la_index>>>& unowned_dofs_in_restriction__to__cell_indices,
  dolfin::la_index block_dofmap_local_size,
  const std::vector<dolfin::la_index>& sub_block_dofmap_local_size
)
{
  // Parts of this code have been adapted from
  //    DofMapBuilder::build
  //    DofMapBuilder::compute_node_reordering
  
  // Now that we know the block_dofmap_local_size, assign local numbering greater than this size
  // to unowned dofs
  dolfin::la_index block_dofmap_unowned_size = 0;
  std::vector<dolfin::la_index> sub_block_dofmap_unowned_size(dofmaps.size());
  std::vector<std::map<dolfin::la_index, dolfin::la_index>> original_to_block__unowned_to_unowned(dofmaps.size());
  std::vector<std::map<dolfin::la_index, dolfin::la_index>> original_to_sub_block__unowned_to_unowned(dofmaps.size());
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
  {
    sub_block_dofmap_unowned_size[i] = 0;
    for (auto original_local_dof : unowned_dofs_in_restriction[i])
    {
      _block_unowned_dofs__local[i].push_back(block_dofmap_local_size + block_dofmap_unowned_size);
      auto original_global_dof = unowned_dofs_in_restriction__local_to_global[i].at(original_local_dof);
      _original_to_block__local_to_local[i][original_local_dof] = block_dofmap_local_size + block_dofmap_unowned_size;
      original_to_block__unowned_to_unowned[i][original_global_dof] = block_dofmap_unowned_size;
      _block_to_original__local_to_local[i][block_dofmap_local_size + block_dofmap_unowned_size] = original_local_dof;
      _original_to_sub_block__local_to_local[i][original_local_dof] = sub_block_dofmap_local_size[i] + sub_block_dofmap_unowned_size[i];
      original_to_sub_block__unowned_to_unowned[i][original_global_dof] = sub_block_dofmap_unowned_size[i];
      _sub_block_to_original__local_to_local[i][sub_block_dofmap_local_size[i] + sub_block_dofmap_unowned_size[i]] = original_local_dof;
      for (dolfin::la_index cell_index : unowned_dofs_in_restriction__to__cell_indices[i].at(original_local_dof))
        _dofmap[cell_index].push_back(block_dofmap_local_size + block_dofmap_unowned_size);
        // Increment counter
      block_dofmap_unowned_size++;
      sub_block_dofmap_unowned_size[i]++;
    }
  }
  
  // Fill in local to global map of unowned dofs
  const std::size_t mpi_rank = MPI::rank(comm);
  const std::size_t mpi_size = MPI::size(comm);
  std::vector<std::vector<std::size_t>> send_buffer(mpi_size);
  std::vector<std::vector<std::size_t>> recv_buffer(mpi_size);
  std::vector<std::size_t> local_to_global_unowned(block_dofmap_unowned_size);
  std::vector<std::vector<std::size_t>> sub_local_to_sub_global_unowned(dofmaps.size());
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
  {
    sub_local_to_sub_global_unowned[i].resize(sub_block_dofmap_unowned_size[i]);
    std::shared_ptr<const DofMap> dofmap = std::dynamic_pointer_cast<const DofMap>(dofmaps[i]);
    int dofmap_block_size = dofmap->index_map()->block_size();
    
    // In order to fill in the (block) local to (block) global map of unowned dofs,
    // we need to proceed as follows:
    // 0. we know the *original* unowned *global* dof.
    // 1. find the *original* owner of the *original* unowned *global* dof. Then, send this *original* 
    //    unowned *global* to its owner.
    // 2. on the *original* owning processor, get the *original* *local* dof. Once this is done,
    //    we can obtain the *block* *local* dof thanks to the original-to-block local-to-local
    //    map that we store as private attribute, and finally obtain the *block* *global*
    //    dof thank to the index map. Then, send this *block* *global* back to the processor
    //    from which we received it.
    // 3. exploit the original-to-block unowned-to-local map to obatin the *block* *local*
    //    dof corresponding to the received *block* *global* dof, and store this in the
    //    (block) local to (block) global map of unowned dofs
    
    // Step 1 - cleanup sending buffer
    for (auto& send_buffer_r: send_buffer)
      send_buffer_r.clear();
    
    // Step 1 - compute
    for (auto original_local_dof : unowned_dofs_in_restriction[i])
    {
      auto original_global_dof = unowned_dofs_in_restriction__local_to_global[i].at(original_local_dof);
      const int index_owner = dofmap->index_map()->global_index_owner(original_global_dof/dofmap_block_size);
      dolfin_assert(index_owner != mpi_rank);
      send_buffer[index_owner].push_back(original_global_dof);
      send_buffer[index_owner].push_back(mpi_rank);
    }
    
    // Step 1 - cleanup receiving buffer
    for (auto& recv_buffer_r: recv_buffer)
      recv_buffer_r.clear();
    
    // Step 1 - communicate
    MPI::all_to_all(comm, send_buffer, recv_buffer);
    
    // Step 2 - cleanup sending buffer
    for (auto& send_buffer_r: send_buffer)
      send_buffer_r.clear();
    
    // Step 2 - compute
    for (std::size_t r = 0; r < mpi_size; ++r)
    {
      for (auto q = recv_buffer[r].begin();
           q != recv_buffer[r].end(); q += 2)
      {
        const std::size_t original_global_dof = *q;
        const std::size_t sender_rank = *(q + 1);

        const std::size_t original_local_dof = original_global_dof - dofmap->ownership_range().first;
        dolfin::la_index block_local_dof = _original_to_block__local_to_local[i].at(original_local_dof);
        dolfin::la_index sub_block_local_dof = _original_to_sub_block__local_to_local[i].at(original_local_dof);
        send_buffer[sender_rank].push_back(_index_map->local_to_global(block_local_dof));
        send_buffer[sender_rank].push_back(_sub_index_map[i]->local_to_global(sub_block_local_dof));
        send_buffer[sender_rank].push_back(original_global_dof);
      }
    }
    
    // Step 2 - cleanup receiving buffer
    for (auto& recv_buffer_r: recv_buffer)
      recv_buffer_r.clear();
    
    // Step 2 - communicate
    MPI::all_to_all(comm, send_buffer, recv_buffer);
    
    // Step 3 - cleanup sending buffer
    for (auto& send_buffer_r: send_buffer)
      send_buffer_r.clear();
    
    // Step 3 - compute
    for (std::size_t r = 0; r < mpi_size; ++r)
    {
      for (auto q = recv_buffer[r].begin();
           q != recv_buffer[r].end(); q += 3)
      {
        const std::size_t block_global_dof = *q;
        const std::size_t sub_block_global_dof = *(q + 1);
        const std::size_t original_global_dof = *(q + 2);
        local_to_global_unowned[original_to_block__unowned_to_unowned[i].at(original_global_dof)] = block_global_dof;
        sub_local_to_sub_global_unowned[i][original_to_sub_block__unowned_to_unowned[i].at(original_global_dof)] = sub_block_global_dof;
      }
    }
  }
  
  _index_map->set_local_to_global(local_to_global_unowned);
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
  {
    _sub_index_map[i]->set_local_to_global(sub_local_to_sub_global_unowned[i]);
  }
  
  // Fill in private attributes related to global indices of unowned dofs
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
  {
    for (auto local_dof : _block_unowned_dofs__local[i])
    {
      _block_unowned_dofs__global[i].push_back(_index_map->local_to_global(local_dof));
    }
  }
}
//-----------------------------------------------------------------------------
void BlockDofMap::_store_real_dofs(
  const std::vector<std::shared_ptr<const GenericDofMap>> dofmaps,
  const std::vector<std::set<std::size_t>>& real_dofs
)
{
  // Fill in private attributes related to Real indices
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
    for (auto real_dof : real_dofs[i])
      _real_dofs__local.push_back(_original_to_block__local_to_local[i].at(real_dof));
}
//-----------------------------------------------------------------------------
BlockDofMap::BlockDofMap(const BlockDofMap& block_dofmap)
{
  _constructor_dofmaps = block_dofmap._constructor_dofmaps;
  _constructor_restrictions = block_dofmap._constructor_restrictions;
  _dofmap = block_dofmap._dofmap;
  _empty_vector = block_dofmap._empty_vector;
  _max_element_dofs = block_dofmap._max_element_dofs;
  _num_entity_dofs = block_dofmap._num_entity_dofs;
  _num_entity_closure_dofs = block_dofmap._num_entity_closure_dofs;
  _num_facet_dofs = block_dofmap._num_facet_dofs;
  _real_dofs__local = block_dofmap._real_dofs__local;
  _index_map = block_dofmap._index_map;
  _sub_index_map = block_dofmap._sub_index_map;
  _block_owned_dofs__local = block_dofmap._block_owned_dofs__local;
  _block_unowned_dofs__local = block_dofmap._block_unowned_dofs__local;
  _block_owned_dofs__global = block_dofmap._block_owned_dofs__global;
  _block_unowned_dofs__global = block_dofmap._block_unowned_dofs__global;
  _original_to_block__local_to_local = block_dofmap._original_to_block__local_to_local;
  _block_to_original__local_to_local = block_dofmap._block_to_original__local_to_local;
  _original_to_sub_block__local_to_local = block_dofmap._original_to_sub_block__local_to_local;
  _sub_block_to_original__local_to_local = block_dofmap._sub_block_to_original__local_to_local;
}
//-----------------------------------------------------------------------------
BlockDofMap::~BlockDofMap()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<const GenericDofMap>> BlockDofMap::dofmaps() const
{
  return _constructor_dofmaps;
}
//-----------------------------------------------------------------------------
bool BlockDofMap::is_view() const
{
    /// BlockDofMap does not allow views, so the value will always be False.
    return false;
}
//-----------------------------------------------------------------------------
std::size_t BlockDofMap::global_dimension() const
{
  return _index_map->size(IndexMap::MapSize::GLOBAL);
}
//-----------------------------------------------------------------------------
std::size_t BlockDofMap::num_element_dofs(std::size_t cell_index) const
{
  return _dofmap.at(cell_index).size();
}
//-----------------------------------------------------------------------------
std::size_t BlockDofMap::max_element_dofs() const
{
  return _max_element_dofs;
}
//-----------------------------------------------------------------------------
std::size_t BlockDofMap::num_entity_dofs(std::size_t entity_dim) const
{
  return _num_entity_dofs.at(entity_dim);
}
//-----------------------------------------------------------------------------
std::size_t BlockDofMap::num_entity_closure_dofs(std::size_t entity_dim) const
{
  return _num_entity_closure_dofs.at(entity_dim);
}
//-----------------------------------------------------------------------------
std::size_t BlockDofMap::num_facet_dofs() const
{
  return _num_facet_dofs;
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> BlockDofMap::ownership_range() const
{
  return _index_map->local_range();
}
//-----------------------------------------------------------------------------
const std::vector<int>&
BlockDofMap::off_process_owner() const
{
  return _index_map->off_process_owner();
}
//-----------------------------------------------------------------------------
const std::unordered_map<int, std::vector<int>>& BlockDofMap::shared_nodes() const
{
  multiphenics_error("BlockDofMap.cpp",
                     "compute shared nodes",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
const std::set<int>& BlockDofMap::neighbours() const
{
  multiphenics_error("BlockDofMap.cpp",
                     "compute neighbours",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
void BlockDofMap::clear_sub_map_data() 
{
  // Nothing to be done, there are no submaps.
}
//-----------------------------------------------------------------------------
Eigen::Map<const Eigen::Array<dolfin::la_index, Eigen::Dynamic, 1>>
BlockDofMap::cell_dofs(std::size_t cell_index) const
{
  if (_dofmap.count(cell_index) > 0)
  {
    const auto & dofmap_cell(_dofmap.at(cell_index)); 
    return Eigen::Map<const Eigen::Array<dolfin::la_index, Eigen::Dynamic, 1>>(dofmap_cell.data(), dofmap_cell.size());
  }
  else
  {
    return Eigen::Map<const Eigen::Array<dolfin::la_index, Eigen::Dynamic, 1>>(_empty_vector.data(), 0);
  }
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index> BlockDofMap::entity_dofs(
    const Mesh& mesh,
    std::size_t entity_dim,
    const std::vector<std::size_t> & entity_indices) const
{
  multiphenics_error("BlockDofMap.cpp",
                     "compute dofs associate to entity indices",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index> BlockDofMap::entity_dofs(
    const Mesh& mesh,
    std::size_t entity_dim) const
{
  multiphenics_error("BlockDofMap.cpp",
                     "compute dofs associate to entity indices",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index> BlockDofMap::entity_closure_dofs(
    const Mesh& mesh,
    std::size_t entity_dim,
    const std::vector<std::size_t> & entity_indices) const
{
  multiphenics_error("BlockDofMap.cpp",
                     "compute closure dofs associate to entity indices",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index> BlockDofMap::entity_closure_dofs(
    const Mesh& mesh,
    std::size_t entity_dim) const
{
  multiphenics_error("BlockDofMap.cpp",
                     "compute closure dofs associate to entity indices",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
void BlockDofMap::tabulate_facet_dofs(std::vector<std::size_t>& element_dofs,
                                 std::size_t cell_facet_index) const
{
  multiphenics_error("BlockDofMap.cpp",
                     "tabulate facet dofs",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
void
BlockDofMap::tabulate_entity_dofs(std::vector<std::size_t>& element_dofs,
                             std::size_t entity_dim, std::size_t cell_entity_index) const
{
  multiphenics_error("BlockDofMap.cpp",
                     "tabulate entity dofs",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
void
BlockDofMap::tabulate_entity_closure_dofs(std::vector<std::size_t>& element_dofs,
                                     std::size_t entity_dim, std::size_t cell_entity_index) const
{
  multiphenics_error("BlockDofMap.cpp",
                     "tabulate entity closure dofs",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
void BlockDofMap::tabulate_global_dofs(std::vector<std::size_t>& element_dofs) const
{
  element_dofs.assign(_real_dofs__local.begin(), _real_dofs__local.end());
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericDofMap> BlockDofMap::copy() const
{
  return std::shared_ptr<GenericDofMap>(new BlockDofMap(*this));
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericDofMap> BlockDofMap::create(const Mesh& new_mesh) const
{
  std::vector<std::shared_ptr<const GenericDofMap>> dofmaps_on_new_mesh;
  for (unsigned int i(0); i < _constructor_dofmaps.size(); ++i)
  {
    std::shared_ptr<const GenericDofMap> dofmap_i = _constructor_dofmaps[i];
    std::shared_ptr<GenericDofMap> dofmap_i_on_new_mesh = dofmap_i->create(new_mesh);
    dofmaps_on_new_mesh.push_back(dofmap_i_on_new_mesh);
  }
  return std::shared_ptr<GenericDofMap>(new BlockDofMap(dofmaps_on_new_mesh, _constructor_restrictions, new_mesh));
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericDofMap>
  BlockDofMap::extract_sub_dofmap(const std::vector<std::size_t>& component,
                             const Mesh& mesh) const
{
  multiphenics_error("BlockDofMap.cpp",
                     "extract sub dofmap",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericDofMap>
  BlockDofMap::collapse(std::unordered_map<std::size_t, std::size_t>&
                   collapsed_map,
                   const Mesh& mesh) const
{
  multiphenics_error("BlockDofMap.cpp",
                     "collapse sub dofmap",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index> BlockDofMap::dofs(const Mesh& mesh,
                                           std::size_t dim) const
{
  multiphenics_error("BlockDofMap.cpp",
                     "obtain list of dofs",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index> BlockDofMap::dofs() const
{
  multiphenics_error("BlockDofMap.cpp",
                     "obtain list of dofs",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
void BlockDofMap::set(GenericVector& x, double value) const
{
  multiphenics_error("BlockDofMap.cpp",
                     "set dof entries of a vector to a value",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
std::shared_ptr<const IndexMap> BlockDofMap::index_map() const
{ 
  return _index_map; 
}
//-----------------------------------------------------------------------------
std::shared_ptr<const IndexMap> BlockDofMap::sub_index_map(std::size_t b) const
{ 
  return _sub_index_map[b]; 
}
//-----------------------------------------------------------------------------
int BlockDofMap::block_size() const
{ 
  return _index_map->block_size(); 
}
//-----------------------------------------------------------------------------
void BlockDofMap::tabulate_local_to_global_dofs(std::vector<std::size_t>& local_to_global_map) const
{
  // This method is exactly the same as
  //    DofMap::tabulate_local_to_global_dofs
  
  const std::size_t bs = _index_map->block_size();
  const std::vector<std::size_t>& local_to_global_unowned
    = _index_map->local_to_global_unowned();
  const std::size_t local_ownership_size
    = _index_map->size(IndexMap::MapSize::OWNED);
  local_to_global_map.resize(_index_map->size(IndexMap::MapSize::ALL));

  const std::size_t global_offset = _index_map->local_range().first;
  for (std::size_t i = 0; i < local_ownership_size; ++i)
    local_to_global_map[i] = i + global_offset;

  for (std::size_t node = 0;
       node < _index_map->local_to_global_unowned().size(); ++node)
  {
    for (std::size_t component = 0; component < bs; ++component)
    {
      local_to_global_map[bs*node + component + local_ownership_size]
        = bs*local_to_global_unowned[node] + component;
    }
  }
}
//-----------------------------------------------------------------------------
std::size_t BlockDofMap::local_to_global_index(int local_index) const
{ 
  return _index_map->local_to_global(local_index);
}
//-----------------------------------------------------------------------------
const std::vector<std::size_t>& BlockDofMap::local_to_global_unowned() const
{
  return _index_map->local_to_global_unowned();
}
//-----------------------------------------------------------------------------
std::string BlockDofMap::str(bool verbose) const
{
  std::stringstream s;
  s << "<BlockDofMap with total global dimension "
    << global_dimension()
    << ">"
    << std::endl;
  return s.str();
}
//-----------------------------------------------------------------------------
const std::vector<dolfin::la_index> & BlockDofMap::block_owned_dofs__local_numbering(std::size_t b) const
{
  return _block_owned_dofs__local[b];
}
//-----------------------------------------------------------------------------
const std::vector<dolfin::la_index> & BlockDofMap::block_unowned_dofs__local_numbering(std::size_t b) const
{
  return _block_unowned_dofs__local[b];
}
//-----------------------------------------------------------------------------
const std::vector<dolfin::la_index> & BlockDofMap::block_owned_dofs__global_numbering(std::size_t b) const
{
  return _block_owned_dofs__global[b];
}
//-----------------------------------------------------------------------------
const std::vector<dolfin::la_index> & BlockDofMap::block_unowned_dofs__global_numbering(std::size_t b) const
{
  return _block_unowned_dofs__global[b];
}
//-----------------------------------------------------------------------------
const std::map<dolfin::la_index, dolfin::la_index> & BlockDofMap::original_to_block(std::size_t b) const
{
  return _original_to_block__local_to_local[b];
}
//-----------------------------------------------------------------------------
const std::map<dolfin::la_index, dolfin::la_index> & BlockDofMap::block_to_original(std::size_t b) const
{
  return _block_to_original__local_to_local[b];
}
//-----------------------------------------------------------------------------
const std::map<dolfin::la_index, dolfin::la_index> & BlockDofMap::original_to_sub_block(std::size_t b) const
{
  return _original_to_sub_block__local_to_local[b];
}
//-----------------------------------------------------------------------------
const std::map<dolfin::la_index, dolfin::la_index> & BlockDofMap::sub_block_to_original(std::size_t b) const
{
  return _sub_block_to_original__local_to_local[b];
}
//-----------------------------------------------------------------------------

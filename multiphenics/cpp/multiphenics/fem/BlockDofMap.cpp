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

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshIterator.h>
#include <multiphenics/fem/BlockDofMap.h>

using namespace multiphenics;
using namespace multiphenics::fem;

using dolfinx::common::IndexMap;
using dolfinx::fem::DofMap;
using dolfinx::mesh::cell_num_entities;
using dolfinx::mesh::Mesh;
using dolfinx::mesh::MeshEntity;
using dolfinx::mesh::MeshFunction;
using dolfinx::mesh::MeshRange;
using dolfinx::mesh::MeshRangeType;

//-----------------------------------------------------------------------------
BlockDofMap::BlockDofMap(std::vector<std::shared_ptr<const DofMap>> dofmaps,
                         std::vector<std::vector<std::shared_ptr<const MeshFunction<std::size_t>>>> restrictions):
  _dofmaps(dofmaps),
  _restrictions(restrictions),
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
BlockDofMap::BlockDofMap(std::vector<std::shared_ptr<const DofMap>> dofmaps,
                         std::vector<std::vector<std::shared_ptr<const MeshFunction<std::size_t>>>> restrictions,
                         const Mesh& mesh):
  BlockDofMap(dofmaps, restrictions, std::vector<std::shared_ptr<const Mesh>>(dofmaps.size(), std::shared_ptr<const Mesh>(&mesh, [](const Mesh*){})))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockDofMap::BlockDofMap(std::vector<std::shared_ptr<const DofMap>> dofmaps,
                         std::vector<std::vector<std::shared_ptr<const MeshFunction<std::size_t>>>> restrictions,
                         std::vector<std::shared_ptr<const Mesh>> meshes):
  BlockDofMap(dofmaps, restrictions)
{
  // A. EXTRACT DOFS FROM ORIGINAL DOFMAPS
  std::vector<std::set<PetscInt>> owned_dofs(dofmaps.size());
  std::vector<std::map<PetscInt, bool>> owned_dofs__to__in_restriction(dofmaps.size());
  std::vector<std::map<PetscInt, std::set<std::size_t>>> owned_dofs__to__cell_indices(dofmaps.size());
  std::vector<std::set<PetscInt>> unowned_dofs_in_restriction(dofmaps.size());
  std::vector<std::map<PetscInt, PetscInt>> unowned_dofs_in_restriction__local_to_global(dofmaps.size());
  std::vector<std::map<PetscInt, std::set<std::size_t>>> unowned_dofs_in_restriction__to__cell_indices(dofmaps.size());
  _extract_dofs_from_original_dofmaps(
    dofmaps, restrictions, meshes,
    owned_dofs, owned_dofs__to__in_restriction, owned_dofs__to__cell_indices,
    unowned_dofs_in_restriction, unowned_dofs_in_restriction__local_to_global, unowned_dofs_in_restriction__to__cell_indices
  );
  
  // B. ASSIGN OWNED DOFS TO BLOCK DOF MAP
  std::int64_t block_dofmap_local_size = 0;
  std::vector<std::int64_t> sub_block_dofmap_local_size;
  _assign_owned_dofs_to_block_dofmap(
    dofmaps, meshes,
    owned_dofs, owned_dofs__to__in_restriction, owned_dofs__to__cell_indices,
    block_dofmap_local_size, sub_block_dofmap_local_size
  );
  
  // C. PREPARE LOCAL TO GLOBAL MAP OF BLOCK DOF MAP FOR UNOWNED DOFS
  assert(std::all_of(meshes.begin(), meshes.end(), [&meshes](std::shared_ptr<const Mesh> mesh){return mesh->mpi_comm() == meshes[0]->mpi_comm();}));
  _prepare_local_to_global_for_unowned_dofs(
    dofmaps, meshes[0]->mpi_comm(),
    unowned_dofs_in_restriction, unowned_dofs_in_restriction__local_to_global, unowned_dofs_in_restriction__to__cell_indices,
    block_dofmap_local_size, sub_block_dofmap_local_size
  );
  
  // D. PRECOMPUTE VIEWS
  _precompute_views(dofmaps);
}
//-----------------------------------------------------------------------------
void BlockDofMap::_extract_dofs_from_original_dofmaps(
  std::vector<std::shared_ptr<const DofMap>> dofmaps,
  std::vector<std::vector<std::shared_ptr<const MeshFunction<std::size_t>>>> restrictions,
  std::vector<std::shared_ptr<const Mesh>> meshes,
  std::vector<std::set<PetscInt>>& owned_dofs,
  std::vector<std::map<PetscInt, bool>>& owned_dofs__to__in_restriction,
  std::vector<std::map<PetscInt, std::set<std::size_t>>>& owned_dofs__to__cell_indices,
  std::vector<std::set<PetscInt>>& unowned_dofs_in_restriction,
  std::vector<std::map<PetscInt, PetscInt>>& unowned_dofs_in_restriction__local_to_global,
  std::vector<std::map<PetscInt, std::set<std::size_t>>>& unowned_dofs_in_restriction__to__cell_indices
) const
{
  assert(dofmaps.size() == restrictions.size());
  assert(dofmaps.size() == meshes.size());
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
  {
    std::shared_ptr<const DofMap> dofmap = std::dynamic_pointer_cast<const DofMap>(dofmaps[i]);
    const std::vector<std::shared_ptr<const MeshFunction<std::size_t>>>& restriction = restrictions[i];
    const Mesh& mesh = *meshes[i];
    
    // Mesh dimension
    const std::size_t D = mesh.topology().dim();
    
    // Consistency check
    if (restriction.size() != 0 && restriction.size() != D + 1)
    {
      throw std::runtime_error("Cannot initialize block dof map. "
                               "Invalid length of restriction array.");
    }
    
    // Local size
    int dofmap_block_size = dofmap->index_map->block_size;
    const std::int64_t dofmap_local_size = dofmap_block_size*(dofmap->index_map->local_range()[1] - dofmap->index_map->local_range()[0]);
    
    // Loop over entity dimension
    for (std::size_t d = 0; d <= D; ++d)
    {
      if (restriction.size() == D + 1)
      {
        if (restriction[d]->dim() != static_cast<int>(d))
        {
          throw std::runtime_error("Cannot initialize block dof map."
                                   "Invalid dimension of restriction mesh function.");
        }
      }
      
      const std::size_t dofs_per_entity = dofmap->element_dof_layout->num_entity_dofs(d);
      if (dofs_per_entity > 0)
      {
        mesh.create_entities(d);
        mesh.create_connectivity(d, D);
        
        for (const auto& e : MeshRange(mesh, d, MeshRangeType::ALL))
        {
          // Check if the mesh entity is in restriction
          bool in_restriction;
          if (restriction.size() > 0)
          {
            in_restriction = (restriction[d]->values()[e.index()] > 0);
          }
          else
          {
            // No restriction provided, keep the whole domain
            in_restriction = true;
          }
          
          // Get ids of all cells connected to the entity
          const std::size_t num_cells
              = mesh.topology().connectivity(d, D)->size(e.index());
          assert(num_cells > 0);
          std::set<std::size_t> cell_indices;
          for (std::size_t c(0); c < num_cells; ++c)
          {
            std::size_t cell_index = e.entities(D)[c];
            cell_indices.insert(cell_index);
          }
          
          // Get the first cell connected to the entity
          const MeshEntity cell(mesh, D, *cell_indices.begin());

          // Find local entity number
          std::size_t local_entity_ind = 0;
          for (int local_i = 0; local_i < cell_num_entities(mesh.cell_type(), d); ++local_i)
          {
            if (cell.entities(d)[local_i] == e.index())
            {
              local_entity_ind = local_i;
              break;
            }
          }

          // Get all cell dofs
          const auto cell_dof_list = dofmap->cell_dofs(cell.index());

          // Tabulate local to local map of dofs on local entity
          const auto local_to_local_map = dofmap->element_dof_layout->entity_dofs(d, local_entity_ind);

          // Fill local dofs for the entity
          for (std::size_t local_dof = 0; local_dof < dofs_per_entity; ++local_dof)
          {
            PetscInt cell_dof = cell_dof_list[local_to_local_map[local_dof]];
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
                std::size_t cell_global_dof = dofmap->index_map->local_to_global(cell_dof/dofmap_block_size)*dofmap_block_size + (cell_dof%dofmap_block_size);
                unowned_dofs_in_restriction__local_to_global[i][cell_dof] = cell_global_dof;
                for (auto c : cell_indices)
                  unowned_dofs_in_restriction__to__cell_indices[i][cell_dof].insert(c);
              }
            }
          }
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void BlockDofMap::_assign_owned_dofs_to_block_dofmap(
  std::vector<std::shared_ptr<const DofMap>> dofmaps,
      std::vector<std::shared_ptr<const Mesh>> meshes,
  const std::vector<std::set<PetscInt>>& owned_dofs,
  const std::vector<std::map<PetscInt, bool>>& owned_dofs__to__in_restriction,
  const std::vector<std::map<PetscInt, std::set<std::size_t>>>& owned_dofs__to__cell_indices,
  std::int64_t& block_dofmap_local_size,
  std::vector<std::int64_t>& sub_block_dofmap_local_size
)
{
  // Fill in private attributes related to local indices of owned dofs
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
  {
    std::int64_t block_i_local_size = 0;
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
        for (std::size_t cell_index : owned_dofs__to__cell_indices[i].at(original_dof))
          _dofmap[cell_index].push_back(block_dofmap_local_size);
        // Increment counters
        block_dofmap_local_size++;
        block_i_local_size++;
      }
    }
    sub_block_dofmap_local_size.push_back(block_i_local_size);
  }
  
  // Communicator
  assert(std::all_of(meshes.begin(), meshes.end(), [&meshes](std::shared_ptr<const Mesh> mesh){return mesh->mpi_comm() == meshes[0]->mpi_comm();}));
  MPI_Comm comm = meshes[0]->mpi_comm();
  
  // Prepare temporary index map, neglecting ghosts
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> empty_ghosts;
  index_map.reset(new IndexMap(comm, block_dofmap_local_size, empty_ghosts, 1));
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
  {
    auto index_map_i = std::make_shared<IndexMap>(comm, sub_block_dofmap_local_size[i], empty_ghosts, 1);
    sub_index_map.push_back(index_map_i);
  }
  
  // Fill in private attributes related to global indices of owned dofs
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
  {
    for (auto local_dof : _block_owned_dofs__local[i])
    {
      _block_owned_dofs__global[i].push_back(index_map->local_to_global(local_dof));
    }
  }
}
//-----------------------------------------------------------------------------
void BlockDofMap::_prepare_local_to_global_for_unowned_dofs(
  std::vector<std::shared_ptr<const DofMap>> dofmaps,
  MPI_Comm comm,
  const std::vector<std::set<PetscInt>>& unowned_dofs_in_restriction,
  const std::vector<std::map<PetscInt, PetscInt>>& unowned_dofs_in_restriction__local_to_global,
  const std::vector<std::map<PetscInt, std::set<std::size_t>>>& unowned_dofs_in_restriction__to__cell_indices,
  std::int64_t block_dofmap_local_size,
  const std::vector<std::int64_t>& sub_block_dofmap_local_size
)
{
  // Parts of this code have been adapted from
  //    DofMapBuilder::build
  //    DofMapBuilder::compute_node_reordering
  
  // Now that we know the block_dofmap_local_size, assign local numbering greater than this size
  // to unowned dofs
  std::int64_t block_dofmap_unowned_size = 0;
  std::vector<std::int64_t> sub_block_dofmap_unowned_size(dofmaps.size());
  std::vector<std::map<PetscInt, PetscInt>> original_to_block__unowned_to_unowned(dofmaps.size());
  std::vector<std::map<PetscInt, PetscInt>> original_to_sub_block__unowned_to_unowned(dofmaps.size());
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
      for (std::size_t cell_index : unowned_dofs_in_restriction__to__cell_indices[i].at(original_local_dof))
        _dofmap[cell_index].push_back(block_dofmap_local_size + block_dofmap_unowned_size);
        // Increment counter
      block_dofmap_unowned_size++;
      sub_block_dofmap_unowned_size[i]++;
    }
  }
  
  // Fill in local to global map of unowned dofs
  const std::size_t mpi_rank = dolfinx::MPI::rank(comm);
  const std::size_t mpi_size = dolfinx::MPI::size(comm);
  std::vector<std::vector<std::size_t>> send_buffer(mpi_size);
  std::vector<std::vector<std::size_t>> recv_buffer(mpi_size);
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> local_to_global_unowned(block_dofmap_unowned_size);
  std::vector<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>, Eigen::aligned_allocator<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>> sub_local_to_sub_global_unowned(dofmaps.size());
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
  {
    sub_local_to_sub_global_unowned[i].resize(sub_block_dofmap_unowned_size[i]);
    std::shared_ptr<const DofMap> dofmap = std::dynamic_pointer_cast<const DofMap>(dofmaps[i]);
    int dofmap_block_size = dofmap->index_map->block_size;
    
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
      const int index_owner = dofmap->index_map->owner(original_global_dof/dofmap_block_size);
      assert(index_owner != mpi_rank);
      send_buffer[index_owner].push_back(original_global_dof);
      send_buffer[index_owner].push_back(mpi_rank);
    }
    
    // Step 1 - cleanup receiving buffer
    for (auto& recv_buffer_r: recv_buffer)
      recv_buffer_r.clear();
    
    // Step 1 - communicate
    dolfinx::MPI::all_to_all(comm, send_buffer, recv_buffer);
    
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

        const std::size_t original_local_dof = original_global_dof - dofmap->index_map->block_size*dofmap->index_map->local_range()[0];
        PetscInt block_local_dof = _original_to_block__local_to_local[i].at(original_local_dof);
        PetscInt sub_block_local_dof = _original_to_sub_block__local_to_local[i].at(original_local_dof);
        send_buffer[sender_rank].push_back(index_map->local_to_global(block_local_dof));
        send_buffer[sender_rank].push_back(sub_index_map[i]->local_to_global(sub_block_local_dof));
        send_buffer[sender_rank].push_back(original_global_dof);
      }
    }
    
    // Step 2 - cleanup receiving buffer
    for (auto& recv_buffer_r: recv_buffer)
      recv_buffer_r.clear();
    
    // Step 2 - communicate
    dolfinx::MPI::all_to_all(comm, send_buffer, recv_buffer);
    
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
  
  // Replace temporary index map with a new one, which now includes ghost local_to_global map
  index_map.reset(new IndexMap(comm, block_dofmap_local_size, local_to_global_unowned, 1));
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
  {
    auto index_map_i = std::make_shared<IndexMap>(comm, sub_block_dofmap_local_size[i], sub_local_to_sub_global_unowned[i], 1);
    sub_index_map.push_back(index_map_i);
  }
  
  // Fill in private attributes related to global indices of unowned dofs
  for (unsigned int i = 0; i < dofmaps.size(); ++i) 
  {
    for (auto local_dof : _block_unowned_dofs__local[i])
    {
      _block_unowned_dofs__global[i].push_back(index_map->local_to_global(local_dof));
    }
  }
}
//-----------------------------------------------------------------------------
BlockDofMap::BlockDofMap(const BlockDofMap& block_dofmap, std::size_t i)
{
  // Get local (owned and unowned) block indices associated to component i
  std::vector<PetscInt> block_i_dofs;
  block_i_dofs.reserve(block_dofmap._block_to_original__local_to_local[i].size());
  for (const auto & parent__block_to_original__iterator: block_dofmap._block_to_original__local_to_local[i])
    block_i_dofs.push_back(parent__block_to_original__iterator.first);
    
  // Intersect parent's _dofmap content with block_i_dofs
  for (const auto & parent__cell_to_dofs: block_dofmap._dofmap)
  {
    const auto cell = parent__cell_to_dofs.first;
    const auto & parent__dofs = parent__cell_to_dofs.second;
    std::set_intersection(
      parent__dofs.begin(), parent__dofs.end(),
      block_i_dofs.begin(), block_i_dofs.end(),
      std::back_inserter(_dofmap[cell])
    );
  }
  
  // Copy index map from local to global from parent
  index_map = block_dofmap.index_map;
  
  // Skip initalizing the rest of the private attributes, as this is the bare minimum
  // required by SparsityPatternBuilder::build()
}
//-----------------------------------------------------------------------------
void BlockDofMap::_precompute_views(
  const std::vector<std::shared_ptr<const DofMap>> dofmaps
)
{
  for (unsigned int i = 0; i < dofmaps.size(); ++i)
    _views.push_back(
      std::make_shared<BlockDofMap>(*this, i)
    );
}
//-----------------------------------------------------------------------------
const BlockDofMap & BlockDofMap::view(std::size_t i) const
{
  return *_views[i];
}
//-----------------------------------------------------------------------------
Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
BlockDofMap::cell_dofs(std::size_t cell_index) const
{
  if (_dofmap.count(cell_index) > 0)
  {
    const auto & dofmap_cell(_dofmap.at(cell_index)); 
    return Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>(dofmap_cell.data(), dofmap_cell.size());
  }
  else
  {
    return Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>(_empty_vector.data(), 0);
  }
}
//-----------------------------------------------------------------------------
std::string BlockDofMap::str(bool verbose) const
{
  std::stringstream s;
  s << "<BlockDofMap with total global dimension "
    << index_map->size_global()
    << ">"
    << std::endl;
  return s.str();
}
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<const dolfinx::fem::DofMap>> BlockDofMap::dofmaps() const
{
  return _dofmaps;
}
//-----------------------------------------------------------------------------
std::vector<std::vector<std::shared_ptr<const dolfinx::mesh::MeshFunction<std::size_t>>>> BlockDofMap::restrictions() const
{
  return _restrictions;
}
//-----------------------------------------------------------------------------
const std::vector<PetscInt> & BlockDofMap::block_owned_dofs__local_numbering(std::size_t b) const
{
  return _block_owned_dofs__local[b];
}
//-----------------------------------------------------------------------------
const std::vector<PetscInt> & BlockDofMap::block_unowned_dofs__local_numbering(std::size_t b) const
{
  return _block_unowned_dofs__local[b];
}
//-----------------------------------------------------------------------------
const std::vector<PetscInt> & BlockDofMap::block_owned_dofs__global_numbering(std::size_t b) const
{
  return _block_owned_dofs__global[b];
}
//-----------------------------------------------------------------------------
const std::vector<PetscInt> & BlockDofMap::block_unowned_dofs__global_numbering(std::size_t b) const
{
  return _block_unowned_dofs__global[b];
}
//-----------------------------------------------------------------------------
const std::map<PetscInt, PetscInt> & BlockDofMap::original_to_block(std::size_t b) const
{
  return _original_to_block__local_to_local[b];
}
//-----------------------------------------------------------------------------
const std::map<PetscInt, PetscInt> & BlockDofMap::block_to_original(std::size_t b) const
{
  return _block_to_original__local_to_local[b];
}
//-----------------------------------------------------------------------------
const std::map<PetscInt, PetscInt> & BlockDofMap::original_to_sub_block(std::size_t b) const
{
  return _original_to_sub_block__local_to_local[b];
}
//-----------------------------------------------------------------------------
const std::map<PetscInt, PetscInt> & BlockDofMap::sub_block_to_original(std::size_t b) const
{
  return _sub_block_to_original__local_to_local[b];
}
//-----------------------------------------------------------------------------

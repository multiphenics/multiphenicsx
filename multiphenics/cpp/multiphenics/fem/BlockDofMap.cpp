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
#include <multiphenics/fem/BlockDofMap.h>

using namespace multiphenics;
using namespace multiphenics::fem;

using dolfinx::common::IndexMap;
using dolfinx::fem::DofMap;

//-----------------------------------------------------------------------------
BlockDofMap::BlockDofMap(std::vector<std::shared_ptr<const DofMap>> dofmaps,
                         std::vector<Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>> restrictions):
  _dofmaps(dofmaps),
  _restrictions(restrictions),
  _original_to_block(dofmaps.size()),
  _block_to_original(dofmaps.size()),
  _original_to_sub_block(dofmaps.size()),
  _sub_block_to_original(dofmaps.size())
{
  // Associate each owned and ghost dof that is in the restriction, i.e. a subset of dofs contained by DOLFINX DofMap,
  // to a numbering with respect to the multiphenics block structure both with offsets due to preceding blocks ("block"),
  // and without offsets ("sub_block")
  _map_owned_dofs(dofmaps, restrictions);
  _map_ghost_dofs(dofmaps, restrictions);

  // Precompute cell dofs
  _precompute_cell_dofs(dofmaps);

  // Precompute views
  _precompute_views(dofmaps);
}
//-----------------------------------------------------------------------------
void BlockDofMap::_map_owned_dofs(std::vector<std::shared_ptr<const DofMap>> dofmaps,
                                  std::vector<Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>> restrictions)
{
  // Compute local block and sub_block indices associated to owned dofs
  std::int32_t block_dofmap_owned_size = 0;
  std::vector<std::int32_t> sub_block_dofmap_owned_size(dofmaps.size());
  for (std::size_t i = 0; i < dofmaps.size(); ++i)
  {
    const auto dofmaps_i_owned_size = dofmaps[i]->index_map->block_size()*dofmaps[i]->index_map->size_local();
    for (Eigen::Index d = 0; d < restrictions[i].rows(); ++d)
    {
      auto original_dof = restrictions[i][d];
      if (original_dof < dofmaps_i_owned_size)
      {
        _original_to_block[i][original_dof] = block_dofmap_owned_size;
        _block_to_original[i][block_dofmap_owned_size] = original_dof;
        _original_to_sub_block[i][original_dof] = sub_block_dofmap_owned_size[i];
        _sub_block_to_original[i][sub_block_dofmap_owned_size[i]] = original_dof;
        // Increment counters
        block_dofmap_owned_size++;
        sub_block_dofmap_owned_size[i]++;
      }
    }
  }

  // Prepare temporary block and sub_block index maps, neglecting ghosts
  assert(std::all_of(dofmaps.begin(), dofmaps.end(), [&dofmaps](std::shared_ptr<const DofMap> dofmap){return dofmap->index_map->mpi_comm() == dofmaps[0]->index_map->mpi_comm();}));
  MPI_Comm comm = dofmaps[0]->index_map->mpi_comm();
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> empty_ghosts;
  index_map.reset(new IndexMap(comm, block_dofmap_owned_size, empty_ghosts, 1));
  for (std::size_t i = 0; i < dofmaps.size(); ++i)
  {
    auto index_map_i = std::make_shared<IndexMap>(comm, sub_block_dofmap_owned_size[i], empty_ghosts, 1);
    sub_index_map.push_back(index_map_i);
  }
}
//-----------------------------------------------------------------------------
void BlockDofMap::_map_ghost_dofs(std::vector<std::shared_ptr<const DofMap>> dofmaps,
                                  std::vector<Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>> restrictions)
{
  // Assign local block and sub_block numbering to ghost dofs
  std::int32_t block_dofmap_ghost_size = 0;
  std::vector<std::int32_t> sub_block_dofmap_ghost_size(dofmaps.size());
  for (std::size_t i = 0; i < dofmaps.size(); ++i)
  {
    const auto dofmaps_i_owned_size = dofmaps[i]->index_map->block_size()*dofmaps[i]->index_map->size_local();
    for (Eigen::Index d = 0; d < restrictions[i].rows(); ++d)
    {
      auto original_local_dof = restrictions[i][d];
      if (original_local_dof >= dofmaps_i_owned_size)
      {
        _original_to_block[i][original_local_dof] = index_map->size_local() + block_dofmap_ghost_size;
        _block_to_original[i][index_map->size_local() + block_dofmap_ghost_size] = original_local_dof;
        _original_to_sub_block[i][original_local_dof] = sub_index_map[i]->size_local() + sub_block_dofmap_ghost_size[i];
        _sub_block_to_original[i][sub_index_map[i]->size_local() + sub_block_dofmap_ghost_size[i]] = original_local_dof;
        // Increment counters
        block_dofmap_ghost_size++;
        sub_block_dofmap_ghost_size[i]++;
      }
    }
  }

  // Fill in local to global map of ghost dofs for both block and sub_block numbering
  assert(std::all_of(dofmaps.begin(), dofmaps.end(), [&dofmaps](std::shared_ptr<const DofMap> dofmap){return dofmap->index_map->mpi_comm() == dofmaps[0]->index_map->mpi_comm();}));
  MPI_Comm comm = dofmaps[0]->index_map->mpi_comm();
  const std::uint32_t mpi_rank = dolfinx::MPI::rank(comm);
  const std::uint32_t mpi_size = dolfinx::MPI::size(comm);
  std::vector<std::vector<std::int64_t>> send_buffer(mpi_size);
  std::vector<std::vector<std::int64_t>> recv_buffer(mpi_size);
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> local_to_global_ghost(block_dofmap_ghost_size);
  std::vector<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>, Eigen::aligned_allocator<Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>> sub_local_to_sub_global_ghost(dofmaps.size());
  for (std::size_t i = 0; i < dofmaps.size(); ++i)
  {
    sub_local_to_sub_global_ghost[i].resize(sub_block_dofmap_ghost_size[i]);
    const auto dofmaps_i_block_size = dofmaps[i]->index_map->block_size();
    const auto dofmaps_i_owned_size = dofmaps_i_block_size*dofmaps[i]->index_map->size_local();
    const auto dofmaps_i_local_range_0 = dofmaps_i_block_size*dofmaps[i]->index_map->local_range()[0];

    // In order to fill in the (block) local to (block) global map of ghost dofs,
    // we need to proceed as follows:
    // 0. we know the *original* ghost *global* dof.
    // 1. find the *original* owner of the *original* ghost *global* dof. Then, send this *original*
    //    ghost *global* to its owner.
    // 2. on the *original* owning processor, get the *original* *local* dof. Once this is done,
    //    we can obtain the *block* *local* dof thanks to the original-to-block local-to-local
    //    map that we store as private attribute, and finally obtain the *block* *global*
    //    dof thank to the index map. Then, send this *block* *global* back to the processor
    //    from which we received it.
    // 3. exploit the original-to-block ghost-to-local map to obatin the *block* *local*
    //    dof corresponding to the received *block* *global* dof, and store this in the
    //    (block) local to (block) global map of ghost dofs

    // Step 1 - cleanup sending buffer
    for (auto& send_buffer_r: send_buffer)
      send_buffer_r.clear();

    // Step 1 - compute
    for (Eigen::Index d = 0; d < restrictions[i].rows(); ++d)
    {
      auto original_local_dof = restrictions[i][d];
      if (original_local_dof >= dofmaps_i_owned_size)
      {
        const auto original_global_dof = dofmaps[i]->index_map->local_to_global(original_local_dof/dofmaps_i_block_size)*dofmaps_i_block_size + (original_local_dof%dofmaps_i_block_size);
        const auto index_owner = dofmaps[i]->index_map->owner(original_global_dof/dofmaps_i_block_size);
        assert(index_owner != mpi_rank);
        send_buffer[index_owner].push_back(original_local_dof);
        send_buffer[index_owner].push_back(original_global_dof);
        send_buffer[index_owner].push_back(mpi_rank);
      }
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
    for (std::uint32_t r = 0; r < mpi_size; ++r)
    {
      for (auto q = recv_buffer[r].begin(); q != recv_buffer[r].end(); q += 3)
      {
        const auto original_local_dof_on_sender = *q;
        const auto original_global_dof = *(q + 1);
        const auto sender_rank = *(q + 2);

        const auto original_local_dof = original_global_dof - dofmaps_i_local_range_0;
        const auto block_local_dof = _original_to_block[i].at(original_local_dof);
        const auto sub_block_local_dof = _original_to_sub_block[i].at(original_local_dof);
        send_buffer[sender_rank].push_back(index_map->local_to_global(block_local_dof));
        send_buffer[sender_rank].push_back(sub_index_map[i]->local_to_global(sub_block_local_dof));
        send_buffer[sender_rank].push_back(original_local_dof_on_sender);
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
    for (std::uint32_t r = 0; r < mpi_size; ++r)
    {
      for (auto q = recv_buffer[r].begin(); q != recv_buffer[r].end(); q += 3)
      {
        const auto block_global_dof = *q;
        const auto sub_block_global_dof = *(q + 1);
        const auto original_local_dof = *(q + 2);

        const auto block_local_dof = _original_to_block[i].at(original_local_dof);
        const auto sub_block_local_dof = _original_to_sub_block[i].at(original_local_dof);
        local_to_global_ghost[block_local_dof - index_map->size_local()] = block_global_dof;
        sub_local_to_sub_global_ghost[i][sub_block_local_dof - sub_index_map[i]->size_local()] = sub_block_global_dof;
      }
    }
  }

  // Replace temporary index map with a new one, which now includes ghost local_to_global map
  index_map.reset(new IndexMap(comm, index_map->size_local(), local_to_global_ghost, 1));
  for (std::size_t i = 0; i < dofmaps.size(); ++i)
  {
    sub_index_map[i].reset(new IndexMap(comm, sub_index_map[i]->size_local(), sub_local_to_sub_global_ghost[i], 1));
  }
}
//-----------------------------------------------------------------------------
void BlockDofMap::_precompute_cell_dofs(std::vector<std::shared_ptr<const dolfinx::fem::DofMap>> dofmaps)
{
  for (std::size_t i = 0; i < dofmaps.size(); ++i)
  {
    const auto & dof_array = dofmaps[i]->dof_array();
    const int cell_dimension = dofmaps[i]->element_dof_layout->num_dofs();
    const int num_cells = dof_array.rows()/cell_dimension;
    for (int cell_index = 0; cell_index < num_cells; ++cell_index)
    {
      const int index = cell_index * cell_dimension;
      auto original_cell_dofs = dof_array.segment(index, cell_dimension);
      std::vector<std::int32_t> block_cell_dofs;
      for (Eigen::Index d = 0; d < original_cell_dofs.rows(); ++d)
      {
        auto original_dof = original_cell_dofs[d];
        if (_original_to_block[i].count(original_dof) > 0)
        {
          block_cell_dofs.push_back(_original_to_block[i][original_dof]);
        }
      }
      if (block_cell_dofs.size() > 0)
      {
        _cell_dofs[cell_index].insert(_cell_dofs[cell_index].end(), block_cell_dofs.begin(), block_cell_dofs.end());
      }
    }
  }
  // Sort the resulting _cell_dofs // TODO is this still needed after removing views? (which use std::set_intersection)
  for (auto & cell_to_dofs: _cell_dofs)
  {
    auto & dofs = cell_to_dofs.second;
    std::sort(dofs.begin(), dofs.end());
  }
}
//-----------------------------------------------------------------------------
void BlockDofMap::_precompute_views(std::vector<std::shared_ptr<const DofMap>> dofmaps)
{
  for (std::size_t i = 0; i < dofmaps.size(); ++i)
    _views.push_back(
      std::make_shared<BlockDofMap>(*this, i)
    );
}
//-----------------------------------------------------------------------------
BlockDofMap::BlockDofMap(const BlockDofMap& block_dofmap, std::size_t i)
{
  // Get local (owned and ghost) block indices associated to component i
  std::vector<std::int32_t> block_i_dofs;
  block_i_dofs.reserve(block_dofmap._block_to_original[i].size());
  for (const auto & parent__block_to_original__iterator: block_dofmap._block_to_original[i])
    block_i_dofs.push_back(parent__block_to_original__iterator.first);

  // Intersect parent's _cell_dofs content with block_i_dofs
  for (const auto & parent__cell_to_dofs: block_dofmap._cell_dofs)
  {
    const auto cell = parent__cell_to_dofs.first;
    const auto & parent__dofs = parent__cell_to_dofs.second;
    std::set_intersection(
      parent__dofs.begin(), parent__dofs.end(),
      block_i_dofs.begin(), block_i_dofs.end(),
      std::back_inserter(_cell_dofs[cell])
    );
  }

  // Copy index map from local to global from parent
  index_map = block_dofmap.index_map;

  // Skip initalizing the rest of the private attributes, as this is the bare minimum
  // required by BlockSparsityPatternBuilder
}
//-----------------------------------------------------------------------------
const BlockDofMap & BlockDofMap::view(std::size_t b) const
{
  return *_views[b];
}
//-----------------------------------------------------------------------------
Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>
BlockDofMap::cell_dofs(int cell_index) const
{
  if (_cell_dofs.count(cell_index) > 0)
  {
    const auto & dofmap_cell(_cell_dofs.at(cell_index));
    return Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>(dofmap_cell.data(), dofmap_cell.size());
  }
  else
  {
    return Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>(_empty_vector.data(), 0);
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
std::vector<Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>> BlockDofMap::restrictions() const
{
  return _restrictions;
}
//-----------------------------------------------------------------------------
const std::map<std::int32_t, std::int32_t> & BlockDofMap::original_to_block(std::size_t b) const
{
  return _original_to_block[b];
}
//-----------------------------------------------------------------------------
const std::map<std::int32_t, std::int32_t> & BlockDofMap::block_to_original(std::size_t b) const
{
  return _block_to_original[b];
}
//-----------------------------------------------------------------------------
const std::map<std::int32_t, std::int32_t> & BlockDofMap::original_to_sub_block(std::size_t b) const
{
  return _original_to_sub_block[b];
}
//-----------------------------------------------------------------------------
const std::map<std::int32_t, std::int32_t> & BlockDofMap::sub_block_to_original(std::size_t b) const
{
  return _sub_block_to_original[b];
}
//-----------------------------------------------------------------------------

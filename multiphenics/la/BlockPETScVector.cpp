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

#include <multiphenics/la/BlockPETScSubVector.h>
#include <multiphenics/la/BlockPETScVector.h>

using namespace dolfin;
using namespace dolfin::la;
using namespace multiphenics;
using namespace multiphenics::la;

//-----------------------------------------------------------------------------
BlockPETScVector::BlockPETScVector() : PETScVector()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockPETScVector::BlockPETScVector(const common::IndexMap& map)
    : BlockPETScVector(map)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockPETScVector::BlockPETScVector(
    MPI_Comm comm, std::array<std::int64_t, 2> range,
    const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& ghost_indices,
    int block_size)
    : PETScVector(comm, range, ghost_indices, block_size)
{
//-----------------------------------------------------------------------------
BlockPETScVector::BlockPETScVector(const BlockPETScVector& x): 
  PETScVector(x),
  _block_dof_map(x._block_dof_map)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockPETScVector::BlockPETScVector(BlockPETScVector&& x):
  PETScVector(x),
  _block_dof_map(x._block_dof_map)
{
  x._block_dof_map = nullptr;
}
//-----------------------------------------------------------------------------
BlockPETScVector::BlockPETScVector(Vec x) : PETScVector(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockPETScVector::~BlockPETScVector()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const BlockPETScVector& BlockPETScVector::operator= (BlockPETScVector&& x)
{
  PETScVector::operator=(x);
  attach_block_dof_map(x._block_dof_map);
  x._block_dof_map = nullptr;
  return *this;
}
//-----------------------------------------------------------------------------
void BlockPETScVector::attach_block_dof_map(std::shared_ptr<const BlockDofMap> block_dof_map) 
{
  _block_dof_map = block_dof_map;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const BlockDofMap> BlockPETScVector::get_block_dof_map() const
{
  dolfin_assert(has_block_dof_map());
  return _block_dof_map;
}
//-----------------------------------------------------------------------------
bool BlockPETScVector::has_block_dof_map() const
{
  return static_cast<bool>(_block_dof_map);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> BlockPETScVector::operator()(std::size_t block_i, BlockInsertMode insert_mode) const
{
  dolfin_assert(has_block_dof_map());
  return std::make_shared<BlockPETScSubVector>(
    *this, 
    _block_dof_map->block_owned_dofs__global_numbering(block_i),
    _block_dof_map->original_to_sub_block(block_i),
    _block_dof_map->original_to_block(block_i),
    _block_dof_map->sub_index_map(block_i),
    _block_dof_map->dofmaps()[block_i]->global_dimension(),
    insert_mode
  );
}

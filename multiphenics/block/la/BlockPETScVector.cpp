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

#ifdef HAS_PETSC

#include <block/la/BlockPETScFactory.h>
#include <block/la/BlockPETScSubVector.h>
#include <block/la/BlockPETScVector.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BlockPETScVector::BlockPETScVector() : PETScVector()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockPETScVector::BlockPETScVector(MPI_Comm comm) : PETScVector(comm)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockPETScVector::BlockPETScVector(MPI_Comm comm, std::size_t N) : PETScVector(comm, N)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockPETScVector::BlockPETScVector(const SparsityPattern& sparsity_pattern) : PETScVector(sparsity_pattern)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockPETScVector::BlockPETScVector(const BlockPETScVector& x): 
  PETScVector(x),
  _block_dof_map(x._block_dof_map)
{
  // Do nothing
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
std::shared_ptr<GenericVector> BlockPETScVector::copy() const
{
  return std::make_shared<BlockPETScVector>(*this);
}
//-----------------------------------------------------------------------------
const BlockPETScVector& BlockPETScVector::operator*= (double a)
{ 
  PETScVector::operator*=(a);
  return *this;
}
//-----------------------------------------------------------------------------
const BlockPETScVector& BlockPETScVector::operator*= (const GenericVector& x)
{
  PETScVector::operator*=(x);
  return *this;
}
//-----------------------------------------------------------------------------
const BlockPETScVector& BlockPETScVector::operator/= (double a)
{
  PETScVector::operator/=(a);
  return *this;
}
//-----------------------------------------------------------------------------
const BlockPETScVector& BlockPETScVector::operator+= (const GenericVector& x)
{
  PETScVector::operator+=(x);
  return *this;
}
//-----------------------------------------------------------------------------
const BlockPETScVector& BlockPETScVector::operator+= (double a)
{
  PETScVector::operator+=(a);
  return *this;
}
//-----------------------------------------------------------------------------
const BlockPETScVector& BlockPETScVector::operator-= (const GenericVector& x)
{
  PETScVector::operator-=(x);
  return *this;
}
//-----------------------------------------------------------------------------
const BlockPETScVector& BlockPETScVector::operator-= (double a)
{
  PETScVector::operator-=(a);
  return *this;
}
//-----------------------------------------------------------------------------
const BlockPETScVector& BlockPETScVector::operator= (const GenericVector& x)
{
  *this = as_type<const BlockPETScVector>(x);
  return *this;
}
//-----------------------------------------------------------------------------
const BlockPETScVector& BlockPETScVector::operator= (double a)
{
  PETScVector::operator=(a);
  return *this;
}
//-----------------------------------------------------------------------------
const BlockPETScVector& BlockPETScVector::operator= (const BlockPETScVector& x)
{
  PETScVector::operator=(x);
  attach_block_dof_map(x._block_dof_map);
  return *this;
}
//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& BlockPETScVector::factory() const
{
  return BlockPETScFactory::instance();
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

#endif

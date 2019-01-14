// Copyright (C) 2016-2019 by the multiphenics authors
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

#ifdef HAS_PETSC

#include <multiphenics/la/BlockPETScFactory.h>
#include <multiphenics/la/BlockPETScVector.h>
#include <multiphenics/la/BlockPETScSubMatrix.h>
#include <multiphenics/la/BlockPETScMatrix.h>

using namespace dolfin;
using namespace multiphenics;

//-----------------------------------------------------------------------------
BlockPETScMatrix::BlockPETScMatrix() : PETScMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockPETScMatrix::BlockPETScMatrix(MPI_Comm comm) : PETScMatrix(comm)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockPETScMatrix::BlockPETScMatrix(Mat A) : PETScMatrix(A)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockPETScMatrix::BlockPETScMatrix(const BlockPETScMatrix& A) : 
  PETScMatrix(A),
  _block_dof_map(A._block_dof_map)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockPETScMatrix::~BlockPETScMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericMatrix> BlockPETScMatrix::copy() const
{
  return std::make_shared<BlockPETScMatrix>(*this);
}
//-----------------------------------------------------------------------------
void BlockPETScMatrix::init_vector(GenericVector& z, std::size_t dim) const
{
  // Call Parent
  PETScMatrix::init_vector(z, dim);
  
  // Attach block dof maps
  BlockPETScVector& _z = as_type<BlockPETScVector>(z);
  _z.attach_block_dof_map(_block_dof_map[dim]);
}
//-----------------------------------------------------------------------------
const BlockPETScMatrix& BlockPETScMatrix::operator*= (double a)
{
  PETScMatrix::operator*=(a);
  return *this;
}
//-----------------------------------------------------------------------------
const BlockPETScMatrix& BlockPETScMatrix::operator/= (double a)
{
  PETScMatrix::operator/=(a);
  return *this;
}
//-----------------------------------------------------------------------------
const GenericMatrix& BlockPETScMatrix::operator= (const GenericMatrix& A)
{
  *this = as_type<const BlockPETScMatrix>(A);
  return *this;
}
//-----------------------------------------------------------------------------
const BlockPETScMatrix& BlockPETScMatrix::operator= (const BlockPETScMatrix& A)
{
  PETScMatrix::operator=(A);
  attach_block_dof_map(A._block_dof_map[0], A._block_dof_map[1]);
  return *this;
}
//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& BlockPETScMatrix::factory() const
{
  return BlockPETScFactory::instance();
}
//-----------------------------------------------------------------------------
void BlockPETScMatrix::attach_block_dof_map(std::shared_ptr<const BlockDofMap> block_dof_map_0, std::shared_ptr<const BlockDofMap> block_dof_map_1) 
{
  dolfin_assert(_block_dof_map.empty());
  _block_dof_map.push_back(block_dof_map_0);
  _block_dof_map.push_back(block_dof_map_1);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const BlockDofMap> BlockPETScMatrix::get_block_dof_map(std::size_t d) const
{
  dolfin_assert(has_block_dof_map(d));
  return _block_dof_map[d];
}
//-----------------------------------------------------------------------------
bool BlockPETScMatrix::has_block_dof_map(std::size_t d) const
{
  return d < _block_dof_map.size() and static_cast<bool>(_block_dof_map[d]);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericMatrix> BlockPETScMatrix::operator()(std::size_t block_i, std::size_t block_j, BlockInsertMode insert_mode) const
{
  dolfin_assert(has_block_dof_map(0));
  dolfin_assert(has_block_dof_map(1));
  return std::make_shared<BlockPETScSubMatrix>(
    *this, 
    _block_dof_map[0]->block_owned_dofs__local_numbering(block_i),
    _block_dof_map[0]->original_to_sub_block(block_i),
    _block_dof_map[0]->block_owned_dofs__global_numbering(block_i),
    _block_dof_map[0]->block_unowned_dofs__global_numbering(block_i),
    _block_dof_map[0]->dofmaps()[block_i]->global_dimension(),
    _block_dof_map[1]->block_owned_dofs__local_numbering(block_j),
    _block_dof_map[1]->original_to_sub_block(block_j),
    _block_dof_map[1]->block_owned_dofs__global_numbering(block_j),
    _block_dof_map[1]->block_unowned_dofs__global_numbering(block_j),
    _block_dof_map[1]->dofmaps()[block_j]->global_dimension(),
    insert_mode
  );
}

#endif

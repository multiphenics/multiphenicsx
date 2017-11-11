// Copyright (C) 2016-2017 by the multiphenics authors
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

#include <dolfin/la/PETScFactory.h>
#include <multiphenics/la/BlockPETScMatrix.h>
#include <multiphenics/la/BlockPETScVector.h>
#include <multiphenics/la/BlockPETScSubMatrix.h>
#include <multiphenics/la/BlockPETScSubVector.h>
#include <multiphenics/la/BlockPETScFactory.h>

using namespace dolfin;
using namespace multiphenics;

// Singleton instance
BlockPETScFactory BlockPETScFactory::factory;

//-----------------------------------------------------------------------------
BlockPETScFactory::BlockPETScFactory()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockPETScFactory::~BlockPETScFactory()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericMatrix> BlockPETScFactory::create_matrix(MPI_Comm comm) const
{
  return std::make_shared<BlockPETScMatrix>(comm);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericMatrix> BlockPETScFactory::create_matrix_with_attached_block_dof_map(
  MPI_Comm comm, std::shared_ptr<const BlockDofMap> block_dof_map_0, std::shared_ptr<const BlockDofMap> block_dof_map_1
) const
{
  auto block_matrix = std::make_shared<BlockPETScMatrix>(comm);
  block_matrix->attach_block_dof_map(block_dof_map_0, block_dof_map_1);
  return block_matrix;
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericMatrix> BlockPETScFactory::wrap_matrix(
  const GenericMatrix & matrix
) const
{
  const PETScMatrix& petsc_matrix = as_type<const PETScMatrix>(matrix);
  auto block_matrix = std::make_shared<BlockPETScMatrix>(petsc_matrix.mat());
  return block_matrix;
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericMatrix> BlockPETScFactory::create_sub_matrix(
  const GenericMatrix & A, std::size_t block_i, std::size_t block_j, BlockInsertMode insert_mode
) const
{
  const BlockPETScMatrix& block_A = as_type<const BlockPETScMatrix>(A);
  return block_A(block_i, block_j, insert_mode);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> BlockPETScFactory::create_vector(MPI_Comm comm) const
{
  return std::make_shared<BlockPETScVector>(comm);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> BlockPETScFactory::create_vector_with_attached_block_dof_map(
  MPI_Comm comm, std::shared_ptr<const BlockDofMap> block_dof_map
) const
{
  auto block_vector = std::make_shared<BlockPETScVector>(comm);
  block_vector->attach_block_dof_map(block_dof_map);
  return block_vector;
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> BlockPETScFactory::wrap_vector(
  const GenericVector & vector
) const
{
  const PETScVector& petsc_vector = as_type<const PETScVector>(vector);
  auto block_vector = std::make_shared<BlockPETScVector>(petsc_vector.vec());
  return block_vector;
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> BlockPETScFactory::create_sub_vector(
  const GenericVector & x, std::size_t block_i, BlockInsertMode insert_mode
) const
{
  const BlockPETScVector& block_x = as_type<const BlockPETScVector>(x);
  return block_x(block_i, insert_mode);
}
//-----------------------------------------------------------------------------
std::shared_ptr<TensorLayout>
BlockPETScFactory::create_layout(MPI_Comm comm, std::size_t rank) const
{
  return PETScFactory::instance().create_layout(comm, rank);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericLinearOperator>
BlockPETScFactory::create_linear_operator(MPI_Comm comm) const
{
  return PETScFactory::instance().create_linear_operator(comm);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericLinearSolver>
BlockPETScFactory::create_lu_solver(MPI_Comm comm, std::string method) const
{
  return PETScFactory::instance().create_lu_solver(comm, method);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericLinearSolver>
BlockPETScFactory::create_krylov_solver(MPI_Comm comm,
                                   std::string method,
                                   std::string preconditioner) const
{
  return PETScFactory::instance().create_krylov_solver(comm, method, preconditioner);
}
//-----------------------------------------------------------------------------
std::map<std::string, std::string> BlockPETScFactory::lu_solver_methods() const
{
  return PETScFactory::instance().lu_solver_methods();
}
//-----------------------------------------------------------------------------
std::map<std::string, std::string> BlockPETScFactory::krylov_solver_methods() const
{
  return PETScFactory::instance().krylov_solver_methods();
}
//-----------------------------------------------------------------------------
std::map<std::string, std::string>
BlockPETScFactory::krylov_solver_preconditioners() const
{
  return PETScFactory::instance().krylov_solver_preconditioners();
}
//-----------------------------------------------------------------------------
BlockPETScFactory& BlockPETScFactory::instance()
{
  return factory;
}

#endif

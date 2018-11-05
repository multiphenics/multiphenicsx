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

#include <dolfin/parameter/GlobalParameters.h>
#include <multiphenics/la/BlockDefaultFactory.h>
#include <multiphenics/la/BlockPETScFactory.h>
#include <multiphenics/log/log.h>

using namespace dolfin;
using namespace dolfin::la;
using namespace multiphenics;
using namespace multiphenics::la;

//-----------------------------------------------------------------------------
BlockDefaultFactory::BlockDefaultFactory()
{
  // Nothing to be done
}
//-----------------------------------------------------------------------------
BlockDefaultFactory::~BlockDefaultFactory()
{
  // Nothing to be done
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericMatrix> BlockDefaultFactory::create_matrix(MPI_Comm comm) const
{
  return factory().create_matrix(comm);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericMatrix> BlockDefaultFactory::create_matrix_with_attached_block_dof_map(
  MPI_Comm comm, std::shared_ptr<const BlockDofMap> block_dof_map_0, std::shared_ptr<const BlockDofMap> block_dof_map_1
) const
{
  return factory().create_matrix_with_attached_block_dof_map(comm, block_dof_map_0, block_dof_map_1);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericMatrix> BlockDefaultFactory::wrap_matrix(
  const GenericMatrix & matrix
) const
{
  return factory().wrap_matrix(matrix);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericMatrix> BlockDefaultFactory::create_sub_matrix(
  const GenericMatrix & A, std::size_t block_i, std::size_t block_j, BlockInsertMode insert_mode
) const
{
  return factory().create_sub_matrix(A, block_i, block_j, insert_mode);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> BlockDefaultFactory::create_vector(MPI_Comm comm) const
{
  return factory().create_vector(comm);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> BlockDefaultFactory::create_vector_with_attached_block_dof_map(
  MPI_Comm comm, std::shared_ptr<const BlockDofMap> block_dof_map
) const
{
  return factory().create_vector_with_attached_block_dof_map(comm, block_dof_map);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> BlockDefaultFactory::wrap_vector(
  const GenericVector & vector
) const
{
  return factory().wrap_vector(vector);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> BlockDefaultFactory::create_sub_vector(
  const GenericVector & x, std::size_t block_i, BlockInsertMode insert_mode
) const
{
  return factory().create_sub_vector(x, block_i, insert_mode);
}
//-----------------------------------------------------------------------------
std::shared_ptr<TensorLayout>
BlockDefaultFactory::create_layout(MPI_Comm comm, std::size_t rank) const
{
  return factory().create_layout(comm, rank);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericLinearOperator>
BlockDefaultFactory::create_linear_operator(MPI_Comm comm) const
{
  return factory().create_linear_operator(comm);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericLinearSolver>
  BlockDefaultFactory::create_lu_solver(MPI_Comm comm, std::string method) const
{
  return factory().create_lu_solver(comm, method);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericLinearSolver>
BlockDefaultFactory::create_krylov_solver(MPI_Comm comm,
                                     std::string method,
                                     std::string preconditioner) const
{
  return factory().create_krylov_solver(comm, method, preconditioner);
}
//-----------------------------------------------------------------------------
std::map<std::string, std::string> BlockDefaultFactory::lu_solver_methods() const
{
  return factory().lu_solver_methods();
}
 //-----------------------------------------------------------------------------
std::map<std::string, std::string>
BlockDefaultFactory::krylov_solver_methods() const
{
  return factory().krylov_solver_methods();
}
//-----------------------------------------------------------------------------
std::map<std::string, std::string>
BlockDefaultFactory::krylov_solver_preconditioners() const
{
  return factory().krylov_solver_preconditioners();
}
//-----------------------------------------------------------------------------
GenericBlockLinearAlgebraFactory& BlockDefaultFactory::factory()
{
  // Fallback
  const std::string default_backend = "PETSc";

  // Get backend from parameter system
  const std::string backend = dolfin::parameters["linear_algebra_backend"];

  // Choose backend
  if (backend == "PETSc")
  {
    #ifdef HAS_PETSC
    return BlockPETScFactory::instance();
    #else
    multiphenics_error("BlockDefaultFactory.cpp",
                       "access block linear algebra backend",
                       "PETSc block linear algebra backend is not available");
    #endif
  }
  else
  {
    multiphenics_error("BlockDefaultFactory.cpp",
                       "access block linear algebra backend",
                       "Invalid block linear algebra backend");
  }
}
//-----------------------------------------------------------------------------

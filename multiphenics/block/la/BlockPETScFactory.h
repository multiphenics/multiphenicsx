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

#ifndef __BLOCK_PETSC_FACTORY_H
#define __BLOCK_PETSC_FACTORY_H

#ifdef HAS_PETSC

#include <block/la/GenericBlockLinearAlgebraFactory.h>

namespace dolfin
{
  
  class BlockPETScFactory : public GenericBlockLinearAlgebraFactory
  {
  public:
    /// Destructor
    virtual ~BlockPETScFactory();

    /// Create empty matrix
    std::shared_ptr<GenericMatrix> create_matrix(MPI_Comm comm) const;
    
    /// Create empty matrix with attached block_dof_map
    std::shared_ptr<GenericMatrix> create_matrix_with_attached_block_dof_map(
      MPI_Comm comm, std::shared_ptr<const BlockDofMap> block_dof_map_0, std::shared_ptr<const BlockDofMap> block_dof_map_1
    ) const;
    
    /// Wrap existing non-block matrix
    std::shared_ptr<GenericMatrix> wrap_matrix(
      std::shared_ptr<const GenericMatrix> matrix
    ) const;
    
    /// Wrap existing non-block matrix and attach block_dof_map
    std::shared_ptr<GenericMatrix> wrap_matrix_and_attach_block_dof_map(
      std::shared_ptr<const GenericMatrix> matrix, std::shared_ptr<const BlockDofMap> block_dof_map_0, std::shared_ptr<const BlockDofMap> block_dof_map_1
    ) const;
    
    /// Create submatrix
    std::shared_ptr<GenericMatrix> create_sub_matrix(
      const GenericMatrix & A, std::size_t block_i, std::size_t block_j, BlockInsertMode insert_mode
    ) const;

    /// Create empty vector
    std::shared_ptr<GenericVector> create_vector(MPI_Comm comm) const;
    
    /// Create empty vector with attached block_dof_map
    std::shared_ptr<GenericVector> create_vector_with_attached_block_dof_map(
      MPI_Comm comm, std::shared_ptr<const BlockDofMap> block_dof_map
    ) const;
    
    /// Wrap existing non-block vector
    std::shared_ptr<GenericVector> wrap_vector(
      std::shared_ptr<const GenericVector> vector
    ) const;
    
    /// Wrap existing non-block vector and attach block_dof_map
    std::shared_ptr<GenericVector> wrap_vector_and_attach_block_dof_map(
      std::shared_ptr<const GenericVector> vector, std::shared_ptr<const BlockDofMap> block_dof_map
    ) const;
    
    /// Create subvector
    std::shared_ptr<GenericVector> create_sub_vector(
      const GenericVector & x, std::size_t block_i, BlockInsertMode insert_mode
    ) const;

    /// Create empty tensor layout
    std::shared_ptr<TensorLayout> create_layout(std::size_t rank) const;

    /// Create empty linear operator
    std::shared_ptr<GenericLinearOperator>
      create_linear_operator(MPI_Comm comm) const;

    /// Create LU solver
    std::shared_ptr<GenericLUSolver> create_lu_solver(MPI_Comm comm,
                                                      std::string method) const;

    /// Create Krylov solver
    std::shared_ptr<GenericLinearSolver>
    create_krylov_solver(MPI_Comm comm,
                         std::string method,
                         std::string preconditioner) const;

    /// Return a list of available LU solver methods
    std::map<std::string, std::string> lu_solver_methods() const;

    /// Return a list of available Krylov solver methods
    std::map<std::string, std::string> krylov_solver_methods() const;

    /// Return a list of available preconditioners
    std::map<std::string, std::string> krylov_solver_preconditioners() const;
    
    /// Return singleton instance
    static BlockPETScFactory& instance();
    
  private:
    
    /// Private constructor
    BlockPETScFactory();
    static BlockPETScFactory factory;
  };
  
}

#endif

#endif

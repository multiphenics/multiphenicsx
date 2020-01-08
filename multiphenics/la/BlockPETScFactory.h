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

#ifndef __BLOCK_PETSC_FACTORY_H
#define __BLOCK_PETSC_FACTORY_H

#ifdef HAS_PETSC

#include <multiphenics/la/GenericBlockLinearAlgebraFactory.h>

namespace multiphenics
{
  
  class BlockPETScFactory : public GenericBlockLinearAlgebraFactory
  {
  public:
    /// Destructor
    virtual ~BlockPETScFactory();

    /// Create empty matrix
    std::shared_ptr<dolfin::GenericMatrix> create_matrix(MPI_Comm comm) const;
    
    /// Create empty matrix with attached block_dof_map
    std::shared_ptr<dolfin::GenericMatrix> create_matrix_with_attached_block_dof_map(
      MPI_Comm comm, std::shared_ptr<const BlockDofMap> block_dof_map_0, std::shared_ptr<const BlockDofMap> block_dof_map_1
    ) const;
    
    /// Wrap existing non-block matrix
    std::shared_ptr<dolfin::GenericMatrix> wrap_matrix(
      const dolfin::GenericMatrix & matrix
    ) const;
    
    /// Create submatrix
    std::shared_ptr<dolfin::GenericMatrix> create_sub_matrix(
      const dolfin::GenericMatrix & A, std::size_t block_i, std::size_t block_j, BlockInsertMode insert_mode
    ) const;

    /// Create empty vector
    std::shared_ptr<dolfin::GenericVector> create_vector(MPI_Comm comm) const;
    
    /// Create empty vector with attached block_dof_map
    std::shared_ptr<dolfin::GenericVector> create_vector_with_attached_block_dof_map(
      MPI_Comm comm, std::shared_ptr<const BlockDofMap> block_dof_map
    ) const;
    
    /// Wrap existing non-block vector
    std::shared_ptr<dolfin::GenericVector> wrap_vector(
      const dolfin::GenericVector & vector
    ) const;
    
    /// Create subvector
    std::shared_ptr<dolfin::GenericVector> create_sub_vector(
      const dolfin::GenericVector & x, std::size_t block_i, BlockInsertMode insert_mode
    ) const;

    /// Create empty tensor layout
    std::shared_ptr<dolfin::TensorLayout> create_layout(MPI_Comm comm,
                                                        std::size_t rank) const;

    /// Create empty linear operator
    std::shared_ptr<dolfin::GenericLinearOperator>
      create_linear_operator(MPI_Comm comm) const;

    /// Create LU solver
    std::shared_ptr<dolfin::GenericLinearSolver> create_lu_solver(MPI_Comm comm,
                                                                  std::string method) const;

    /// Create Krylov solver
    std::shared_ptr<dolfin::GenericLinearSolver>
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

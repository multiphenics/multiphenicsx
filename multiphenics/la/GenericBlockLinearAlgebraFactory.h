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

#ifndef __GENERIC_BLOCK_LINEAR_ALGEBRA_FACTORY_H
#define __GENERIC_BLOCK_LINEAR_ALGEBRA_FACTORY_H

#include <dolfin/la/GenericLinearAlgebraFactory.h>
#include <multiphenics/fem/BlockDofMap.h>
#include <multiphenics/la/BlockInsertMode.h>

namespace dolfin
{

  /// Default block linear algebra factory based on global parameter "linear_algebra_backend"

  class GenericBlockLinearAlgebraFactory: public GenericLinearAlgebraFactory
  {

    public:

    /// Constructor
    GenericBlockLinearAlgebraFactory();

    /// Destructor
    virtual ~GenericBlockLinearAlgebraFactory();
    
    /// Create empty matrix with attached block_dof_map
    virtual std::shared_ptr<GenericMatrix> create_matrix_with_attached_block_dof_map(
      MPI_Comm comm, std::shared_ptr<const BlockDofMap> block_dof_map_0, std::shared_ptr<const BlockDofMap> block_dof_map_1
    ) const = 0;
    
    /// Wrap existing non-block matrix
    virtual std::shared_ptr<GenericMatrix> wrap_matrix(
      std::shared_ptr<const GenericMatrix> matrix
    ) const = 0;
    
    /// Wrap existing non-block matrix and attach block_dof_map
    virtual std::shared_ptr<GenericMatrix> wrap_matrix_and_attach_block_dof_map(
      std::shared_ptr<const GenericMatrix> matrix, std::shared_ptr<const BlockDofMap> block_dof_map_0, std::shared_ptr<const BlockDofMap> block_dof_map_1
    ) const = 0;

    /// Create submatrix
    virtual std::shared_ptr<GenericMatrix> create_sub_matrix(
      const GenericMatrix & A, std::size_t block_i, std::size_t block_j, BlockInsertMode insert_mode
    ) const = 0;

    /// Create empty vector with attached block_dof_map
    virtual std::shared_ptr<GenericVector> create_vector_with_attached_block_dof_map(
      MPI_Comm comm, std::shared_ptr<const BlockDofMap> block_dof_map
    ) const = 0;
    
    /// Wrap existing non-block vector
    virtual std::shared_ptr<GenericVector> wrap_vector(
      std::shared_ptr<const GenericVector> vector
    ) const = 0;
    
    /// Wrap existing non-block vector and attach block_dof_map
    virtual std::shared_ptr<GenericVector> wrap_vector_and_attach_block_dof_map(
      std::shared_ptr<const GenericVector> vector, std::shared_ptr<const BlockDofMap> block_dof_map
    ) const = 0;
    
    /// Create subvector
    virtual std::shared_ptr<GenericVector> create_sub_vector(
      const GenericVector & x, std::size_t block_i, BlockInsertMode insert_mode
    ) const = 0;

  };

}

#endif

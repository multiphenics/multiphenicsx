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

#ifndef __BLOCK_PETSC_MATRIX_H
#define __BLOCK_PETSC_MATRIX_H

#include <dolfin/la/PETScMatrix.h>
#include <multiphenics/fem/BlockDofMap.h>

namespace multiphenics
{
  namespace la
  {
    class BlockPETScSubMatrix;
    
    class BlockPETScMatrix : public dolfin::PETScMatrix
    {
    public:
      /// Create empty matrix (on MPI_COMM_WORLD)
      BlockPETScMatrix();

      /// Create empty matrix
      explicit BlockPETScMatrix(MPI_Comm comm, const SparsityPattern& sparsity_pattern);

      /// Create a wrapper around a PETSc Mat pointer. The Mat object
      /// should have been created, e.g. via PETSc MatCreate.
      explicit BlockPETScMatrix(Mat A);

      /// Copy constructor
      BlockPETScMatrix(const BlockPETScMatrix& A);
      
      /// Move constructor (falls through to base class move constructor)
      BlockPETScMatrix(BlockPETScMatrix&& A) = default;

      /// Destructor
      ~BlockPETScMatrix();
      
      /// Initialize vector to be compatible with the matrix-vector product
      /// y = Ax. In the parallel case, size and layout are both important.
      ///
      /// @param      dim (std::size_t) The dimension (axis): dim = 0 --> z
      ///         = y, dim = 1 --> z = x
      BlockPETScVector init_vector(std::size_t dim) const;

      /// Assignment operator (deleted)
      BlockPETScMatrix& operator=(const BlockPETScMatrix& A) = delete;

      /// Move assignment operator
      BlockPETScMatrix& operator=(BlockPETScMatrix&& A) = default;
      
      //--- Special block functions ---
      
      /// Attach BlockDofMap for submatrix creation
      void attach_block_dof_map(std::shared_ptr<const BlockDofMap> block_dof_map_0, std::shared_ptr<const BlockDofMap> block_dof_map_1);
      
      /// Get BlockDofMap for submatrix creation
      std::shared_ptr<const BlockDofMap> get_block_dof_map(std::size_t d) const;
      
      /// Check if BlockDofMap for submatrix creation has been attached
      bool has_block_dof_map(std::size_t d) const;
      
      /// Block access
      std::shared_ptr<dolfin::GenericMatrix> operator()(std::size_t block_i, std::size_t block_j, BlockInsertMode insert_mode) const;
      
    private:
      std::vector<std::shared_ptr<const BlockDofMap>> _block_dof_map;
    };
  }
}

#endif

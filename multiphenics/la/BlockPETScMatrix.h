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

#ifdef HAS_PETSC

#include <dolfin/la/PETScMatrix.h>
#include <multiphenics/fem/BlockDofMap.h>
#include <multiphenics/la/GenericBlockMatrix.h>

namespace multiphenics
{
  namespace la
  {
    class BlockPETScSubMatrix;
    
    class BlockPETScMatrix : public dolfin::PETScMatrix, public GenericBlockMatrix
    {
    public:
      /// Create empty matrix (on MPI_COMM_WORLD)
      BlockPETScMatrix();

      /// Create empty matrix
      explicit BlockPETScMatrix(MPI_Comm comm);

      /// Create a wrapper around a PETSc Mat pointer. The Mat object
      /// should have been created, e.g. via PETSc MatCreate.
      explicit BlockPETScMatrix(Mat A);

      /// Copy constructor
      BlockPETScMatrix(const BlockPETScMatrix& A);

      /// Destructor
      virtual ~BlockPETScMatrix();
      
      //--- Implementation of the GenericMatrix interface --
      
      /// Return copy of matrix
      virtual std::shared_ptr<dolfin::GenericMatrix> copy() const;
      
      /// Initialize vector z to be compatible with the matrix-vector product
      /// y = Ax. In the parallel case, both size and layout are
      /// important.
      ///
      /// @param z (GenericVector&)
      ///         Vector to initialise
      /// @param  dim (std::size_t)
      ///         The dimension (axis): dim = 0 --> z = y, dim = 1 --> z = x
      virtual void init_vector(dolfin::GenericVector& z, std::size_t dim) const;
      
      /// Multiply matrix by given number
      virtual const BlockPETScMatrix& operator*= (double a);

      /// Divide matrix by given number
      virtual const BlockPETScMatrix& operator/= (double a);

      /// Assignment operator
      virtual const dolfin::GenericMatrix& operator= (const GenericMatrix& A);
      
      //--- Special functions ---

      /// Return linear algebra backend factory
      virtual dolfin::GenericLinearAlgebraFactory& factory() const;
      
      //--- Special PETSc Functions ---
      
      /// Assignment operator
      const BlockPETScMatrix& operator= (const BlockPETScMatrix& A);
      
      //--- Special block functions ---
      
      /// Attach BlockDofMap for submatrix creation
      virtual void attach_block_dof_map(std::shared_ptr<const BlockDofMap> block_dof_map_0, std::shared_ptr<const BlockDofMap> block_dof_map_1);
      
      /// Get BlockDofMap for submatrix creation
      virtual std::shared_ptr<const BlockDofMap> get_block_dof_map(std::size_t d) const;
      
      /// Check if BlockDofMap for submatrix creation has been attached
      virtual bool has_block_dof_map(std::size_t d) const;
      
      /// Block access
      virtual std::shared_ptr<dolfin::GenericMatrix> operator()(std::size_t block_i, std::size_t block_j, BlockInsertMode insert_mode) const;
      
    private:
      std::vector<std::shared_ptr<const BlockDofMap>> _block_dof_map;
      
      // This map stores:
      // * as key: the global row index of rows that have been ident_local'ed in at least a submatrix
      // * as value: the vector of global col indices which should be set to one, instead of zero
      mutable std::map<dolfin::la_index, std::set<dolfin::la_index>> _ident_global_rows_to_global_cols;
      
      // Allow BlockPETScSubMatrix to access _local_zeroed_rows
      friend class BlockPETScSubMatrix;
    };
  }
}

#endif

#endif

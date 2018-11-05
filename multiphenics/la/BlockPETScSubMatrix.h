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

#ifndef __BLOCK_PETSC_SUB_MATRIX_H
#define __BLOCK_PETSC_SUB_MATRIX_H

#include <dolfin/la/PETScMatrix.h>
#include <multiphenics/la/BlockInsertMode.h>
#include <multiphenics/la/BlockPETScMatrix.h>

namespace multiphenics
{
  namespace la
  {
    class BlockPETScMatrix;
    
    /// This is an extension of PETScMatrix to be used while assemblying block forms, that
    /// a) carries out the extraction of a sub matrix
    /// b) in case of restrictions, overrides get/set/add methods to convert original index without restriction to index with restriction
    class BlockPETScSubMatrix : public dolfin::PETScMatrix
    {
    public:
      /// Constructor
      BlockPETScSubMatrix(const dolfin::GenericMatrix & A,
                          const std::vector<dolfin::la_index> & block_owned_dofs_0__local_numbering, 
                          const std::map<dolfin::la_index, dolfin::la_index> & original_to_sub_block_0, 
                          const std::vector<dolfin::la_index> & block_owned_dofs_0__global_numbering,
                          const std::vector<dolfin::la_index> & block_unowned_dofs_0__global_numbering,
                          std::size_t unrestricted_size_0,
                          const std::vector<dolfin::la_index> & block_owned_dofs_1__local_numbering, 
                          const std::map<dolfin::la_index, dolfin::la_index> & original_to_sub_block_1, 
                          const std::vector<dolfin::la_index> & block_owned_dofs_1__global_numbering,
                          const std::vector<dolfin::la_index> & block_unowned_dofs_1__global_numbering, 
                          std::size_t unrestricted_size_1,
                          BlockInsertMode insert_mode);

      /// Destructor
      virtual ~BlockPETScSubMatrix();
      
      //--- Implementation of the GenericMatrix interface --
      
      /// Return number of rows (dim = 0) or columns (dim = 1)
      /// Note that the number returned here is the size of the *unrestricted* submatrix,
      /// and *not* the actual size of the existing (*restricted*) submatrix
      std::size_t size(std::size_t dim) const;

      /// Return number of rows and columns (num_rows, num_cols).
      /// Note that the number returned here is the size of the *unrestricted* submatrix,
      /// and *not* the actual size of the existing (*restricted*) submatrix
      std::pair<std::int64_t, std::int64_t> size() const;
      
      /// Return local range along dimension dim
      virtual std::pair<std::int64_t, std::int64_t> local_range(std::size_t dim) const;
      
      /// Get block of values
      virtual void get(double* block,
                       std::size_t m, const dolfin::la_index* rows,
                       std::size_t n, const dolfin::la_index* cols) const;

      /// Set block of values using global indices
      virtual void set(const double* block,
                       std::size_t m, const dolfin::la_index* rows,
                       std::size_t n, const dolfin::la_index* cols);

      /// Set block of values using local indices
      virtual void set_local(const double* block,
                             std::size_t m, const dolfin::la_index* rows,
                             std::size_t n, const dolfin::la_index* cols);

      /// Add block of values using global indices
      virtual void add(const double* block,
                       std::size_t m, const dolfin::la_index* rows,
                       std::size_t n, const dolfin::la_index* cols);

      /// Add block of values using local indices
      virtual void add_local(const double* block,
                             std::size_t m, const dolfin::la_index* rows,
                             std::size_t n, const dolfin::la_index* cols);
                             
      /// Add multiple of given matrix (AXPY operation)
      virtual void axpy(double a, const dolfin::GenericMatrix& A,
                        bool same_nonzero_pattern);

      /// Return norm of matrix
      double norm(std::string norm_type) const;
                             
      /// Get non-zero values of given row
      virtual void getrow(std::size_t row,
                          std::vector<std::size_t>& columns,
                          std::vector<double>& values) const;

      /// Set values for given row
      virtual void setrow(std::size_t row,
                          const std::vector<std::size_t>& columns,
                          const std::vector<double>& values);
                          
      /// Set given rows (global row indices) to zero
      virtual void zero(std::size_t m, const dolfin::la_index* rows);

      /// Set given rows (local row indices) to zero
      virtual void zero_local(std::size_t m, const dolfin::la_index* rows);

      /// Set given rows (global row indices) to identity matrix
      virtual void ident(std::size_t m, const dolfin::la_index* rows);

      /// Set given rows (local row indices) to identity matrix
      virtual void ident_local(std::size_t m, const dolfin::la_index* rows);
      
      // Matrix-vector product, y = Ax
      virtual void mult(const dolfin::GenericVector& x, dolfin::GenericVector& y) const;

      // Matrix-vector product, y = A^T x
      virtual void transpmult(const dolfin::GenericVector& x, dolfin::GenericVector& y) const;

      /// Get diagonal of a matrix
      virtual void get_diagonal(dolfin::GenericVector& x) const;

      /// Set diagonal of a matrix
      virtual void set_diagonal(const dolfin::GenericVector& x);

      /// Multiply matrix by given number
      virtual const dolfin::PETScMatrix& operator*= (double a);

      /// Divide matrix by given number
      virtual const dolfin::PETScMatrix& operator/= (double a);

      /// Assignment operator
      virtual const dolfin::GenericMatrix& operator= (const dolfin::GenericMatrix& A);

      /// Test if matrix is symmetric
      virtual bool is_symmetric(double tol) const;

      //--- Special PETSc Functions ---

      /// Sets the prefix used by PETSc when searching the options
      /// database
      void set_options_prefix(std::string options_prefix);

      /// Returns the prefix used by PETSc when searching the options
      /// database
      std::string get_options_prefix() const;

      /// Call PETSc function MatSetFromOptions on the PETSc Mat object
      void set_from_options();

      /// Assignment operator
      const dolfin::PETScMatrix& operator= (const dolfin::PETScMatrix& A);

      /// Attach nullspace to matrix (typically used by Krylov solvers
      /// when solving singular systems)
      void set_nullspace(const dolfin::VectorSpaceBasis& nullspace);

      /// Attach near nullspace to matrix (used by preconditioners, such
      /// as smoothed aggregation algerbraic multigrid)
      void set_near_nullspace(const dolfin::VectorSpaceBasis& nullspace);

      /// Dump matrix to PETSc binary format
      void binary_dump(std::string file_name) const;
      
    private:
    
      void to_restricted_submatrix_row_indices(
        const std::vector<dolfin::la_index> & block_unrestricted_submatrix_row_indices, std::vector<dolfin::la_index> & block_restricted_submatrix_row_indices,
        std::vector<bool> * is_row_in_restriction = NULL
      );
      void to_restricted_submatrix_col_indices(
        const std::vector<dolfin::la_index> & block_unrestricted_submatrix_col_indices, std::vector<dolfin::la_index> & block_restricted_submatrix_col_indices,
        std::vector<bool> * is_col_in_restriction = NULL
      );
      void to_restricted_submatrix_indices_and_values(
        const std::vector<dolfin::la_index> & block_unrestricted_submatrix_row_indices, std::vector<dolfin::la_index> & block_restricted_submatrix_row_indices,
        const std::vector<dolfin::la_index> & block_unrestricted_submatrix_col_indices, std::vector<dolfin::la_index> & block_restricted_submatrix_col_indices,
        const std::vector<double> & block_unrestricted_submatrix_values, std::vector<double> & block_restricted_submatrix_values
      );
      
      const BlockPETScMatrix & _global_matrix;
      const std::map<dolfin::la_index, dolfin::la_index> & _original_to_sub_block_0;
      const std::map<dolfin::la_index, dolfin::la_index> & _original_to_sub_block_1;
      const std::size_t _unrestricted_size_0;
      const std::size_t _unrestricted_size_1;
      /*PETSc*/ InsertMode _insert_mode;
      std::vector<IS> _is;
      
      std::vector<dolfin::la_index> _delayed_zero_local;
      std::vector<dolfin::la_index> _delayed_ident_local;
    };
  }
}

#endif

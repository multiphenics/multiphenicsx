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
#include <multiphenics/fem/BlockDofMap.h>
#include <multiphenics/la/BlockInsertMode.h>
#include <multiphenics/la/BlockPETScSubVector.h>

namespace multiphenics
{
  namespace la
  {
    /// This is an extension of PETScMatrix to be used while assemblying block forms, that
    /// a) carries out the extraction of a sub matrix
    /// b) in case of restrictions, overrides get/set/add methods to convert original index without restriction to index with restriction
    class BlockPETScSubMatrix : public dolfin::la::PETScMatrix
    {
    public:
      /// Constructor
      BlockPETScSubMatrix(const dolfin::la::PETScMatrix & A,
                          std::size_t block_i, std::size_t block_j,
                          std::shared_ptr<const multiphenics::fem::BlockDofMap> block_dof_map_0,
                          std::shared_ptr<const multiphenics::fem::BlockDofMap> block_dof_map_1,
                          BlockInsertMode insert_mode);

      /// Destructor
      ~BlockPETScSubMatrix();
      
      /// Copy constructor (deleted)
      BlockPETScSubMatrix(const BlockPETScSubMatrix& A) = delete;

      /// Move constructor (deleted)
      BlockPETScSubMatrix(BlockPETScSubMatrix&& A) = delete;
      
      /// Assignment operator (deleted)
      BlockPETScSubMatrix& operator=(const BlockPETScSubMatrix& A) = delete;

      /// Move assignment operator (deleted)
      BlockPETScSubMatrix& operator=(BlockPETScSubMatrix&& A) = delete;
      
      /// Return number of rows and columns (num_rows, num_cols).
      std::array<std::int64_t, 2> size() const;
      
      /// Return local range along dimension dim
      std::array<std::int64_t, 2> local_range(std::size_t dim) const;
      
      /// Set block of values using global indices
      void set(const PetscScalar* block,
               std::size_t m, const PetscInt* rows,
               std::size_t n, const PetscInt* cols);

      /// Set block of values using local indices
      void set_local(const PetscScalar* block,
                     std::size_t m, const PetscInt* rows,
                     std::size_t n, const PetscInt* cols);

      /// Add block of values using global indices
      void add(const PetscScalar* block,
               std::size_t m, const PetscInt* rows,
               std::size_t n, const PetscInt* cols);

      /// Add block of values using local indices
      void add_local(const PetscScalar* block,
                     std::size_t m, const PetscInt* rows,
                     std::size_t n, const PetscInt* cols);
                             
      /// Return norm of matrix
      double norm(dolfin::la::Norm norm_type) const;
                             
      /// Set all entries to zero and keep any sparse structure
      void zero();
      
      /// Finalize assembly of tensor. The following values are recognized
      /// for the mode parameter:
      /// @param type
      ///   FINAL    - corresponds to PETSc MatAssemblyBegin+End(MAT_FINAL_ASSEMBLY)
      ///   FLUSH  - corresponds to PETSc MatAssemblyBegin+End(MAT_FLUSH_ASSEMBLY)
      void apply(AssemblyType type);
      
      // Matrix-vector product, y = Ax
      void mult(const BlockPETScSubVector& x, BlockPETScSubVector& y) const;

      /// Multiply matrix by a scalar
      void scale(PetscScalar a);

      /// Test if matrix is symmetric
      bool is_symmetric(double tol) const;
      
      /// Test if matrix is hermitian
      bool is_hermitian(double tol) const;
      
      //--- Special PETSc Functions ---

      /// Sets the prefix used by PETSc when searching the options
      /// database
      void set_options_prefix(std::string options_prefix);

      /// Returns the prefix used by PETSc when searching the options
      /// database
      std::string get_options_prefix() const;

      /// Call PETSc function MatSetFromOptions on the PETSc Mat object
      void set_from_options();

      /// Attach nullspace to matrix (typically used by Krylov solvers
      /// when solving singular systems)
      void set_nullspace(const dolfin::la::VectorSpaceBasis& nullspace);

      /// Attach near nullspace to matrix (used by preconditioners, such
      /// as smoothed aggregation algerbraic multigrid)
      void set_near_nullspace(const dolfin::la::VectorSpaceBasis& nullspace);

    private:
      // Hide operations with PETScVector arguments
      using PETScMatrix::mult;
    
      void to_restricted_submatrix_row_indices(
        const std::vector<PetscInt> & block_unrestricted_submatrix_row_indices, std::vector<PetscInt> & block_restricted_submatrix_row_indices,
        std::vector<bool> * is_row_in_restriction = NULL
      );
      void to_restricted_submatrix_col_indices(
        const std::vector<PetscInt> & block_unrestricted_submatrix_col_indices, std::vector<PetscInt> & block_restricted_submatrix_col_indices,
        std::vector<bool> * is_col_in_restriction = NULL
      );
      void to_restricted_submatrix_indices_and_values(
        const std::vector<PetscInt> & block_unrestricted_submatrix_row_indices, std::vector<PetscInt> & block_restricted_submatrix_row_indices,
        const std::vector<PetscInt> & block_unrestricted_submatrix_col_indices, std::vector<PetscInt> & block_restricted_submatrix_col_indices,
        const std::vector<PetscScalar> & block_unrestricted_submatrix_values, std::vector<PetscScalar> & block_restricted_submatrix_values
      );
      
      const PETScMatrix & _global_matrix;
      const std::map<PetscInt, PetscInt> & _original_to_sub_block_0;
      const std::map<PetscInt, PetscInt> & _original_to_sub_block_1;
      /*PETSc*/ InsertMode _insert_mode;
      std::vector<IS> _is;
    };
  }
}

#endif

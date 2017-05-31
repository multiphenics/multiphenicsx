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

#ifdef HAS_PETSC

#include <block/la/BlockPETScSubMatrix.h>
#include <block/log/log.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BlockPETScSubMatrix::BlockPETScSubMatrix(
  const GenericMatrix & A,
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
  BlockInsertMode insert_mode
) : PETScMatrix(A.mpi_comm()), _global_matrix(as_type<const BlockPETScMatrix>(A)),
    _original_to_sub_block_0(original_to_sub_block_0),
    _unrestricted_size_0(unrestricted_size_0),
    _original_to_sub_block_1(original_to_sub_block_1),
    _unrestricted_size_1(unrestricted_size_1)
{
  PetscErrorCode ierr;
  
  // Initialize PETSc insert mode
  if (insert_mode == BlockInsertMode::INSERT_VALUES)
    _insert_mode = /* PETSc */ INSERT_VALUES;
  else if (insert_mode == BlockInsertMode::ADD_VALUES)
    _insert_mode = /* PETSc */ ADD_VALUES;
  else
    block_error("BlockPETScSubMatrix.cpp",
                "initialize sub matrix",
                "Invalid value for insert mode");
  
  // Extract sub matrix
  IS is_0, is_1;
  
  ierr = ISCreateGeneral(A.mpi_comm(), block_owned_dofs_0__local_numbering.size(), block_owned_dofs_0__local_numbering.data(),
                         PETSC_USE_POINTER, &is_0);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISCreateGeneral");
  
  ierr = ISCreateGeneral(A.mpi_comm(), block_owned_dofs_1__local_numbering.size(), block_owned_dofs_1__local_numbering.data(),
                         PETSC_USE_POINTER, &is_1);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISCreateGeneral");
  
  _is.push_back(is_0);
  _is.push_back(is_1);
  
  ierr = MatGetLocalSubMatrix(_global_matrix.mat(), _is[0], _is[1], &this->_matA);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatGetLocalSubMatrix");
  
  // Initialization of local to global PETSc map.
  // Here "global" is inteded with respect to original matrix _global_matrix.mat()
  // (compare to BlockPETScSubVector).
  // --- from PETScMatrix::init --- //
  // Create pointers to PETSc IndexSet for local-to-global map
  ISLocalToGlobalMapping petsc_local_to_global0, petsc_local_to_global1;

  std::vector<PetscInt> _map0, _map1;
  _map0.reserve(block_owned_dofs_0__global_numbering.size() + block_unowned_dofs_0__global_numbering.size());
  _map0.insert(_map0.end(), block_owned_dofs_0__global_numbering.begin(), block_owned_dofs_0__global_numbering.end());
  _map0.insert(_map0.end(), block_unowned_dofs_0__global_numbering.begin(), block_unowned_dofs_0__global_numbering.end());
  _map1.reserve(block_owned_dofs_1__global_numbering.size() + block_unowned_dofs_1__global_numbering.size());
  _map1.insert(_map1.end(), block_owned_dofs_1__global_numbering.begin(), block_owned_dofs_1__global_numbering.end());
  _map1.insert(_map1.end(), block_unowned_dofs_1__global_numbering.begin(), block_unowned_dofs_1__global_numbering.end());
  
  // Create PETSc local-to-global map/index set
  ierr = ISLocalToGlobalMappingCreate(A.mpi_comm(), 1, _map0.size(), _map0.data(),
                                      PETSC_COPY_VALUES, &petsc_local_to_global0);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");
  ierr = ISLocalToGlobalMappingCreate(A.mpi_comm(), 1, _map1.size(), _map1.data(),
                                      PETSC_COPY_VALUES, &petsc_local_to_global1);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");

  // Set matrix local-to-global maps
  ierr = MatSetLocalToGlobalMapping(this->_matA, petsc_local_to_global0,
                                    petsc_local_to_global1);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetLocalToGlobalMapping");

  // Clean up local-to-global maps
  ierr = ISLocalToGlobalMappingDestroy(&petsc_local_to_global0);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");
  ierr = ISLocalToGlobalMappingDestroy(&petsc_local_to_global1);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");
  // --- end --- from PETScMatrix::init --- end --- //
}
//-----------------------------------------------------------------------------
BlockPETScSubMatrix::~BlockPETScSubMatrix()
{
  PetscErrorCode ierr;
  
  // --- preparation for delayed application of zero_local and ident_local --- //
  // Get matrix local-to-global maps, as set by the constructor
  ISLocalToGlobalMapping petsc_local_to_global0, petsc_local_to_global1;
  ierr = MatGetLocalToGlobalMapping(this->_matA, &petsc_local_to_global0,
                                    &petsc_local_to_global1);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatGetLocalToGlobalMapping");
  // Convert submatrix local indices to matrix global indices
  std::vector<dolfin::la_index> delayed_zero_local__matrix_global_row_indices(_delayed_zero_local.size());
  ISLocalToGlobalMappingApply(petsc_local_to_global0, _delayed_zero_local.size(), _delayed_zero_local.data(), delayed_zero_local__matrix_global_row_indices.data());
  std::vector<dolfin::la_index> delayed_zero_local__matrix_global_col_indices(_delayed_zero_local.size());
  ISLocalToGlobalMappingApply(petsc_local_to_global1, _delayed_zero_local.size(), _delayed_zero_local.data(), delayed_zero_local__matrix_global_col_indices.data());
  std::vector<dolfin::la_index> delayed_ident_local__matrix_global_row_indices(_delayed_ident_local.size());
  ISLocalToGlobalMappingApply(petsc_local_to_global0, _delayed_ident_local.size(), _delayed_ident_local.data(), delayed_ident_local__matrix_global_row_indices.data());
  std::vector<dolfin::la_index> delayed_ident_local__matrix_global_col_indices(_delayed_ident_local.size());
  ISLocalToGlobalMappingApply(petsc_local_to_global1, _delayed_ident_local.size(), _delayed_ident_local.data(), delayed_ident_local__matrix_global_col_indices.data());
  // Do not clean up local-to-global maps, this will be taken care of by MatRestoreLocalSubMatrix
  // --- end --- preparation for delayed application of zero_local and ident_local --- end --- //
  
  // --- restore the global matrix --- //
  ierr = MatRestoreLocalSubMatrix(_global_matrix.mat(), _is[0], _is[1], &this->_matA);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatRestoreLocalSubMatrix");
  ierr = ISDestroy(&_is[0]);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISDestroy");
  ierr = ISDestroy(&_is[1]);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISDestroy");
  // --- end --- restore the global matrix --- end --- //
  
  // --- application of zero_local and ident_local on the (restored) global matrix --- //
  // Use flags to detect if any call to MatZeroRows and MatSetValue has been done, because 
  // MatZeroRows needs to be called collectively, while MatSetValue requires a 
  // BeginAssembly/EndAssembly that we do not want to enforce to no avail if 
  // it is not necessary
  int at_least_one_row_to_be_zeroed(0);
  int at_least_one_non_zero_value_set(0);
  // Initialize the flag relative to MatZeroRows
  at_least_one_row_to_be_zeroed = MPI::sum(_global_matrix.mpi_comm(), _delayed_zero_local.size() + _delayed_ident_local.size());
  // First of all, zero the rows
  if (_delayed_zero_local.size() > 0)
  {
    ierr = MatZeroRows(_global_matrix.mat(), delayed_zero_local__matrix_global_row_indices.size(), delayed_zero_local__matrix_global_row_indices.data(), 0.0, NULL, NULL);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatZeroRows");
  }
  if (_delayed_ident_local.size() > 0)
  {
    for (std::size_t i(0); i < _delayed_ident_local.size(); ++i)
    {
      auto matrix_global_row_index = delayed_ident_local__matrix_global_row_indices[i];
      auto matrix_global_col_index = delayed_ident_local__matrix_global_col_indices[i];
      _global_matrix._ident_global_rows_to_global_cols[matrix_global_row_index].insert(matrix_global_col_index);
    }
    ierr = MatZeroRows(_global_matrix.mat(), delayed_ident_local__matrix_global_row_indices.size(), delayed_ident_local__matrix_global_row_indices.data(), 0.0, NULL, NULL);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatZeroRows");
  }
  if (at_least_one_row_to_be_zeroed > 0 && _delayed_zero_local.size() == 0 && _delayed_ident_local.size() == 0)
  {
    // Need to place a dummy call to MatZeroRows to avoid deadlocks in parallel
    ierr = MatZeroRows(_global_matrix.mat(), 0, NULL, 0.0, NULL, NULL);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatZeroRows");
  }
  // Then, check if there were required some non-zero (equal to 1) elements on those rows, and insert them
  if (_delayed_zero_local.size() > 0)
  {
    for (auto matrix_global_row_index : delayed_zero_local__matrix_global_row_indices)
    {
      if (_global_matrix._ident_global_rows_to_global_cols.count(matrix_global_row_index) > 0)
      {
        at_least_one_non_zero_value_set = 1;
        for (auto matrix_global_col_index : _global_matrix._ident_global_rows_to_global_cols[matrix_global_row_index])
        {
          ierr = MatSetValue(_global_matrix.mat(), matrix_global_row_index, matrix_global_col_index, 1.0, INSERT_VALUES);
          if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetValue");
        }
      }
    }
  }
  if (_delayed_ident_local.size() > 0)
  {
    for (auto matrix_global_row_index : delayed_ident_local__matrix_global_row_indices)
    {
      if (_global_matrix._ident_global_rows_to_global_cols.count(matrix_global_row_index) > 0)
      {
        at_least_one_non_zero_value_set = 1;
        for (auto matrix_global_col_index : _global_matrix._ident_global_rows_to_global_cols[matrix_global_row_index])
        {
          ierr = MatSetValue(_global_matrix.mat(), matrix_global_row_index, matrix_global_col_index, 1.0, INSERT_VALUES);
          if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetValue");
        }
      }
    }
  }
  // Call PETSc assembly, only if at least one processor has inserted new values
  at_least_one_non_zero_value_set = MPI::sum(_global_matrix.mpi_comm(), at_least_one_non_zero_value_set);
  if (at_least_one_non_zero_value_set > 0)
  {
    ierr = MatAssemblyBegin(_global_matrix.mat(), MAT_FINAL_ASSEMBLY);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAssemblyBegin");
    ierr = MatAssemblyEnd(_global_matrix.mat(), MAT_FINAL_ASSEMBLY);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAssemblyEnd");    
  }
  // --- end --- application of zero_local and ident_local on the (restored) global matrix --- end --- //
}
//-----------------------------------------------------------------------------
std::size_t BlockPETScSubMatrix::size(std::size_t dim) const
{
  if (dim == 0)
  {
    return _unrestricted_size_0;
  }
  else if (dim == 1)
  {
    return _unrestricted_size_1;
  }
  else
  {
    block_error("BlockPETScSubMatrix.cpp",
                 "access size of sub matrix",
                 "Illegal axis, must be 0 or 1");
  }
}
//-----------------------------------------------------------------------------
std::pair<std::int64_t, std::int64_t> BlockPETScSubMatrix::size() const
{
  return {this->size(0), this->size(1)};
}
//-----------------------------------------------------------------------------
std::pair<std::int64_t, std::int64_t>
BlockPETScSubMatrix::local_range(std::size_t dim) const
{
  block_error("BlockPETScSubMatrix.cpp",
              "access local range for PETSc matrix",
              "This method is not implemented because it would be inconsistent with the overridden size() method");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::get(double* block,
                              std::size_t m, const dolfin::la_index* rows,
                              std::size_t n, const dolfin::la_index* cols) const
{
  block_error("BlockPETScSubMatrix.cpp",
              "get (possibly non local) sub matrix values",
              "This method is not available because there is no guarantee that MatGetValues is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::set(const double* block,
                              std::size_t m, const dolfin::la_index* rows,
                              std::size_t n, const dolfin::la_index* cols)
{
  block_error("BlockPETScSubMatrix.cpp",
              "set (possibly non local) sub matrix values",
              "This method is not available because there is no guarantee that MatSetValues is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::set_local(const double* block,
                                    std::size_t m, const dolfin::la_index* rows,
                                    std::size_t n, const dolfin::la_index* cols)
{
  if (_insert_mode != INSERT_VALUES)
    block_error("BlockPETScSubMatrix.cpp",
                "set local sub matrix values",
                "This method is available only when INSERT_VALUES is chosen as mode in the constructor");
                
  std::vector<la_index> rows_vec(rows, rows + m);
  std::vector<la_index> cols_vec(cols, cols + n);
  std::vector<double> vals_vec(block, block + m*n);
  std::vector<la_index> restricted_rows_vec;
  std::vector<la_index> restricted_cols_vec;
  std::vector<double> restricted_vals_vec;
  to_restricted_submatrix_indices_and_values(rows_vec, restricted_rows_vec, cols_vec, restricted_cols_vec, vals_vec, restricted_vals_vec);
  PETScMatrix::set_local(restricted_vals_vec.data(), restricted_rows_vec.size(), restricted_rows_vec.data(), restricted_cols_vec.size(), restricted_cols_vec.data());
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::add(const double* block,
                              std::size_t m, const dolfin::la_index* rows,
                              std::size_t n, const dolfin::la_index* cols)
{
  block_error("BlockPETScSubMatrix.cpp",
              "add (possibly non local) sub matrix values",
              "This method is not available because there is no guarantee that MatSetValues is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::add_local(const double* block,
                                    std::size_t m, const dolfin::la_index* rows,
                                    std::size_t n, const dolfin::la_index* cols)
{
  if (_insert_mode != ADD_VALUES)
    block_error("BlockPETScSubMatrix.cpp",
                "add local sub matrix values",
                "This method is available only when ADD_VALUES is chosen as mode in the constructor");
                
  std::vector<la_index> rows_vec(rows, rows + m);
  std::vector<la_index> cols_vec(cols, cols + n);
  std::vector<double> vals_vec(block, block + m*n);
  std::vector<la_index> restricted_rows_vec;
  std::vector<la_index> restricted_cols_vec;
  std::vector<double> restricted_vals_vec;
  to_restricted_submatrix_indices_and_values(rows_vec, restricted_rows_vec, cols_vec, restricted_cols_vec, vals_vec, restricted_vals_vec);
  PETScMatrix::add_local(restricted_vals_vec.data(), restricted_rows_vec.size(), restricted_rows_vec.data(), restricted_cols_vec.size(), restricted_cols_vec.data());
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::axpy(double a, const GenericMatrix& A,
                  bool same_nonzero_pattern)
{
  block_error("BlockPETScSubMatrix.cpp",
              "add multiple of given submatrix",
              "This method is not available because there is no guarantee that MatAXPY is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
double BlockPETScSubMatrix::norm(std::string norm_type) const
{
  block_error("BlockPETScSubMatrix.cpp",
              "compute norm of a submatrix",
              "This method is not available because there is no guarantee that MatNorm is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::getrow(std::size_t row,
                                 std::vector<std::size_t>& columns,
                                 std::vector<double>& values) const
{
  block_error("BlockPETScSubMatrix.cpp",
              "get (possibly non local) sub matrix rows",
              "This method is not available because there is no guarantee that MatGetRow/MatRestoreRow are implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::setrow(std::size_t row,
                                 const std::vector<std::size_t>& columns,
                                 const std::vector<double>& values)
{
  block_error("BlockPETScSubMatrix.cpp",
              "set (possibly non local) sub matrix rows",
              "This method is not available because there is no guarantee that MatSetValues is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::zero(std::size_t m, const dolfin::la_index* rows)
{
  block_error("BlockPETScSubMatrix.cpp",
              "zero (possibly non local) sub matrix rows",
              "This method is not available because there is no guarantee that MatZeroRows is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::zero_local(std::size_t m, const dolfin::la_index* rows)
{
  if (_insert_mode != INSERT_VALUES)
    block_error("BlockPETScSubMatrix.cpp",
                "zero local sub matrix values",
                "This method is available only when INSERT_VALUES is chosen as mode in the constructor");
                
  std::vector<la_index> rows_vec(rows, rows + m);
  std::vector<la_index> restricted_rows_vec;
  to_restricted_submatrix_row_indices(rows_vec, restricted_rows_vec);
  // zero_local does not work for PETSc submatrices. Delay its evaluation in the destructor, before restoring the global matrix
  _delayed_zero_local.reserve(_delayed_zero_local.size() + restricted_rows_vec.size());
  _delayed_zero_local.insert(_delayed_zero_local.end(), restricted_rows_vec.begin(), restricted_rows_vec.end());
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::ident(std::size_t m, const dolfin::la_index* rows)
{
  block_error("BlockPETScSubMatrix.cpp",
              "ident (possibly non local) sub matrix rows",
              "This method is not available because there is no guarantee that MatZeroRows is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::ident_local(std::size_t m, const dolfin::la_index* rows)
{
  if (_insert_mode != INSERT_VALUES)
    block_error("BlockPETScSubMatrix.cpp",
                "ident local sub matrix values",
                "This method is available only when INSERT_VALUES is chosen as mode in the constructor");
                
  std::vector<la_index> rows_vec(rows, rows + m);
  std::vector<la_index> restricted_rows_vec;
  to_restricted_submatrix_row_indices(rows_vec, restricted_rows_vec);
  // ident_local does not work for PETSc submatrices. Delay its evaluation in the destructor, before restoring the global matrix
  _delayed_ident_local.reserve(_delayed_ident_local.size() + restricted_rows_vec.size());
  _delayed_ident_local.insert(_delayed_ident_local.end(), restricted_rows_vec.begin(), restricted_rows_vec.end());
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::mult(const GenericVector& x, GenericVector& y) const
{
  block_error("BlockPETScSubMatrix.cpp",
              "perform submatrix-subvector product",
              "This method is not available because there is no guarantee that MatMult is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::transpmult(const GenericVector& x, GenericVector& y) const
{
  block_error("BlockPETScSubMatrix.cpp",
              "perform submatrix-transpose-subvector product",
              "This method is not available because there is no guarantee that MatMultTranspose is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::get_diagonal(GenericVector& x) const
{
  block_error("BlockPETScSubMatrix.cpp",
              "get diagonal submatrix values",
              "This method is not available because there is no guarantee that MatGetDiagonal is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::set_diagonal(const GenericVector& x)
{
  block_error("BlockPETScSubMatrix.cpp",
              "set diagonal submatrix values",
              "This method is not available because there is no guarantee that MatDiagonalSet is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
const PETScMatrix& BlockPETScSubMatrix::operator*= (double a)
{
  block_error("BlockPETScSubMatrix.cpp",
              "scale submatrix",
              "This method is not available because there is no guarantee that MatScale is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
const PETScMatrix& BlockPETScSubMatrix::operator/= (double a)
{
  block_error("BlockPETScSubMatrix.cpp",
              "scale submatrix",
              "This method is not available because there is no guarantee that MatScale is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
const GenericMatrix& BlockPETScSubMatrix::operator= (const GenericMatrix& A)
{
  block_error("BlockPETScSubMatrix.cpp",
              "assign submatrix",
              "This method is not available because submatrices cannot be assigned");
}
//-----------------------------------------------------------------------------
bool BlockPETScSubMatrix::is_symmetric(double tol) const
{
  block_error("BlockPETScSubMatrix.cpp",
              "check submatrix symmetry",
              "This method is not available because there is no guarantee that MatIsSymmetric is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& BlockPETScSubMatrix::factory() const
{
  block_error("BlockPETScSubMatrix.cpp",
              "generate linear algebra factory from submatrix",
              "This method is not available because no factory should be generated from a submatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::set_options_prefix(std::string options_prefix)
{
  block_error("BlockPETScSubMatrix.cpp",
              "set options prefix",
              "This method is not available because there is no guarantee that MatSetOptionsPrefix is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
std::string BlockPETScSubMatrix::get_options_prefix() const
{
  block_error("BlockPETScSubMatrix.cpp",
              "get options prefix",
              "This method is not available because there is no guarantee that MatGetOptionsPrefix is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::set_from_options()
{
  block_error("BlockPETScSubMatrix.cpp",
              "set from options",
              "This method is not available because there is no guarantee that MatSetFromOptions is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
const PETScMatrix& BlockPETScSubMatrix::operator= (const PETScMatrix& A)
{
  block_error("BlockPETScSubMatrix.cpp",
              "assign submatrix",
              "This method is not available because submatrices cannot be assigned");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::set_nullspace(const VectorSpaceBasis& nullspace)
{
  block_error("BlockPETScSubMatrix.cpp",
              "set nullspace",
              "This method is not available because there is no guarantee that MatSetNullSpace is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::set_near_nullspace(const VectorSpaceBasis& nullspace)
{
  block_error("BlockPETScSubMatrix.cpp",
              "set near nullspace",
              "This method is not available because there is no guarantee that MatSetNearNullSpace is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::binary_dump(std::string file_name) const
{
  block_error("BlockPETScSubMatrix.cpp",
              "dump to file",
              "This method is not available because there is no guarantee that MatView is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::to_restricted_submatrix_row_indices(
  const std::vector<dolfin::la_index> & block_unrestricted_submatrix_row_indices, std::vector<dolfin::la_index> & block_restricted_submatrix_row_indices,
  std::vector<bool> * is_row_in_restriction
) 
{
  for (auto block_unrestricted_submatrix_row_index : block_unrestricted_submatrix_row_indices)
    if (_original_to_sub_block_0.count(block_unrestricted_submatrix_row_index) > 0)
    {
      block_restricted_submatrix_row_indices.push_back(
        _original_to_sub_block_0.at(block_unrestricted_submatrix_row_index)
      );
      if (is_row_in_restriction != NULL)
        is_row_in_restriction->push_back(true);
    }
    else
    {
      if (is_row_in_restriction != NULL)
        is_row_in_restriction->push_back(false);
    }
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::to_restricted_submatrix_col_indices(
  const std::vector<dolfin::la_index> & block_unrestricted_submatrix_col_indices, std::vector<dolfin::la_index> & block_restricted_submatrix_col_indices,
  std::vector<bool> * is_col_in_restriction
) 
{
  for (auto block_unrestricted_submatrix_col_index : block_unrestricted_submatrix_col_indices)
    if (_original_to_sub_block_1.count(block_unrestricted_submatrix_col_index) > 0) 
    {
      block_restricted_submatrix_col_indices.push_back(
        _original_to_sub_block_1.at(block_unrestricted_submatrix_col_index)
      );
      if (is_col_in_restriction != NULL)
        is_col_in_restriction->push_back(true);
    }
    else
    {
      if (is_col_in_restriction != NULL)
        is_col_in_restriction->push_back(false);
    }
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::to_restricted_submatrix_indices_and_values(
  const std::vector<dolfin::la_index> & block_unrestricted_submatrix_row_indices, std::vector<dolfin::la_index> & block_restricted_submatrix_row_indices,
  const std::vector<dolfin::la_index> & block_unrestricted_submatrix_col_indices, std::vector<dolfin::la_index> & block_restricted_submatrix_col_indices,
  const std::vector<double> & block_unrestricted_submatrix_values, std::vector<double> & block_restricted_submatrix_values
)
{
  // Extract row indices
  std::vector<bool> is_row_in_restriction;
  to_restricted_submatrix_row_indices(block_unrestricted_submatrix_row_indices, block_restricted_submatrix_row_indices, &is_row_in_restriction);
  
  // Extract col indices
  std::vector<bool> is_col_in_restriction;
  to_restricted_submatrix_col_indices(block_unrestricted_submatrix_col_indices, block_restricted_submatrix_col_indices, &is_col_in_restriction);
  
  // Reserve
  block_restricted_submatrix_values.reserve(block_restricted_submatrix_row_indices.size()*block_restricted_submatrix_col_indices.size());
    
  // Extract values
  std::size_t unrestricted_value_iterator(0);
  for (bool r : is_row_in_restriction)
    for (bool c : is_col_in_restriction)
    {
      if (r and c)
        block_restricted_submatrix_values.push_back(block_unrestricted_submatrix_values[unrestricted_value_iterator]);
      unrestricted_value_iterator++;
    }
  dolfin_assert(unrestricted_value_iterator == block_unrestricted_submatrix_values.size())
  dolfin_assert(block_restricted_submatrix_values.size() == block_restricted_submatrix_row_indices.size()*block_restricted_submatrix_col_indices.size());
}

#endif

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

#include <multiphenics/la/BlockPETScSubMatrix.h>
#include <multiphenics/log/log.h>

using namespace multiphenics;
using namespace multiphenics::la;

using dolfin::la::Norm;
using dolfin::la::petsc_error;
using dolfin::la::PETScMatrix;
using dolfin::la::VectorSpaceBasis;
using multiphenics::fem::BlockDofMap;

//-----------------------------------------------------------------------------
BlockPETScSubMatrix::BlockPETScSubMatrix(
  const PETScMatrix & A,
  std::size_t block_i, std::size_t block_j,
  std::shared_ptr<const BlockDofMap> block_dof_map_0,
  std::shared_ptr<const BlockDofMap> block_dof_map_1,
  BlockInsertMode insert_mode
) : PETScMatrix(), _global_matrix(A),
    _original_to_sub_block_0(block_dof_map_0->original_to_sub_block(block_i)),
    _original_to_sub_block_1(block_dof_map_1->original_to_sub_block(block_j))
{
  PetscErrorCode ierr;
  
  // Initialize PETSc insert mode
  if (insert_mode == BlockInsertMode::INSERT_VALUES)
    _insert_mode = /* PETSc */ INSERT_VALUES;
  else if (insert_mode == BlockInsertMode::ADD_VALUES)
    _insert_mode = /* PETSc */ ADD_VALUES;
  else
    multiphenics_error("BlockPETScSubMatrix.cpp",
                       "initialize sub matrix",
                       "Invalid value for insert mode");
  
  // Extract sub matrix
  IS is_0, is_1;
  
  const auto & block_owned_dofs_0__local_numbering = block_dof_map_0->block_owned_dofs__local_numbering(block_i);
  ierr = ISCreateGeneral(A.mpi_comm(), block_owned_dofs_0__local_numbering.size(), block_owned_dofs_0__local_numbering.data(),
                         PETSC_USE_POINTER, &is_0);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISCreateGeneral");
  
  const auto & block_owned_dofs_1__local_numbering = block_dof_map_1->block_owned_dofs__local_numbering(block_j);
  ierr = ISCreateGeneral(A.mpi_comm(), block_owned_dofs_1__local_numbering.size(), block_owned_dofs_1__local_numbering.data(),
                         PETSC_USE_POINTER, &is_1);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISCreateGeneral");
  
  _is.push_back(is_0);
  _is.push_back(is_1);
  
  ierr = MatDestroy(&this->_matA); // which was automatically created by parent constructor
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatDestroy");
  ierr = MatGetLocalSubMatrix(_global_matrix.mat(), _is[0], _is[1], &this->_matA);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatGetLocalSubMatrix");
  
  // Initialization of local to global PETSc map.
  // Here "global" is inteded with respect to original matrix _global_matrix.mat()
  // (compare to BlockPETScSubVector).
  // --- from PETScMatrix::init --- //
  // Create pointers to PETSc IndexSet for local-to-global map
  ISLocalToGlobalMapping petsc_local_to_global0, petsc_local_to_global1;
  
  const auto & block_owned_dofs_0__global_numbering = block_dof_map_0->block_owned_dofs__global_numbering(block_i);
  const auto & block_unowned_dofs_0__global_numbering = block_dof_map_0->block_unowned_dofs__global_numbering(block_i);
  const auto & block_owned_dofs_1__global_numbering = block_dof_map_1->block_owned_dofs__global_numbering(block_j);
  const auto & block_unowned_dofs_1__global_numbering = block_dof_map_1->block_unowned_dofs__global_numbering(block_j);
  
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
  
  // --- restore the global matrix --- //
  ierr = MatRestoreLocalSubMatrix(_global_matrix.mat(), _is[0], _is[1], &this->_matA);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatRestoreLocalSubMatrix");
  ierr = ISDestroy(&_is[0]);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISDestroy");
  ierr = ISDestroy(&_is[1]);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISDestroy");
  // --- end --- restore the global matrix --- end --- //
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> BlockPETScSubMatrix::size() const
{
  multiphenics_error("BlockPETScSubMatrix.cpp",
                     "access size of PETSc sub matrix",
                     "This method was supposedly never used by the sub matrix interface");
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2>
BlockPETScSubMatrix::local_range(std::size_t dim) const
{
  multiphenics_error("BlockPETScSubMatrix.cpp",
                     "access local range of PETSc sub matrix",
                     "This method was supposedly never used by the sub matrix interface");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::set(const PetscScalar* block,
                              std::size_t m, const PetscInt* rows,
                              std::size_t n, const PetscInt* cols)
{
  multiphenics_error("BlockPETScSubMatrix.cpp",
                     "set (possibly non local) sub matrix values",
                     "This method is not available because there is no guarantee that MatSetValues is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::set_local(const PetscScalar* block,
                                    std::size_t m, const PetscInt* rows,
                                    std::size_t n, const PetscInt* cols)
{
  if (_insert_mode != INSERT_VALUES)
    multiphenics_error("BlockPETScSubMatrix.cpp",
                       "set local sub matrix values",
                       "This method is available only when INSERT_VALUES is chosen as mode in the constructor");
                
  std::vector<PetscInt> rows_vec(rows, rows + m);
  std::vector<PetscInt> cols_vec(cols, cols + n);
  std::vector<PetscScalar> vals_vec(block, block + m*n);
  std::vector<PetscInt> restricted_rows_vec;
  std::vector<PetscInt> restricted_cols_vec;
  std::vector<PetscScalar> restricted_vals_vec;
  to_restricted_submatrix_indices_and_values(rows_vec, restricted_rows_vec, cols_vec, restricted_cols_vec, vals_vec, restricted_vals_vec);
  PETScMatrix::set_local(restricted_vals_vec.data(), restricted_rows_vec.size(), restricted_rows_vec.data(), restricted_cols_vec.size(), restricted_cols_vec.data());
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::add(const PetscScalar* block,
                              std::size_t m, const PetscInt* rows,
                              std::size_t n, const PetscInt* cols)
{
  multiphenics_error("BlockPETScSubMatrix.cpp",
                     "add (possibly non local) sub matrix values",
                     "This method is not available because there is no guarantee that MatSetValues is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::add_local(const PetscScalar* block,
                                    std::size_t m, const PetscInt* rows,
                                    std::size_t n, const PetscInt* cols)
{
  if (_insert_mode != ADD_VALUES)
    multiphenics_error("BlockPETScSubMatrix.cpp",
                       "add local sub matrix values",
                       "This method is available only when ADD_VALUES is chosen as mode in the constructor");
                
  std::vector<PetscInt> rows_vec(rows, rows + m);
  std::vector<PetscInt> cols_vec(cols, cols + n);
  std::vector<PetscScalar> vals_vec(block, block + m*n);
  std::vector<PetscInt> restricted_rows_vec;
  std::vector<PetscInt> restricted_cols_vec;
  std::vector<PetscScalar> restricted_vals_vec;
  to_restricted_submatrix_indices_and_values(rows_vec, restricted_rows_vec, cols_vec, restricted_cols_vec, vals_vec, restricted_vals_vec);
  PETScMatrix::add_local(restricted_vals_vec.data(), restricted_rows_vec.size(), restricted_rows_vec.data(), restricted_cols_vec.size(), restricted_cols_vec.data());
}
//-----------------------------------------------------------------------------
double BlockPETScSubMatrix::norm(Norm norm_type) const
{
  multiphenics_error("BlockPETScSubMatrix.cpp",
                     "compute norm of a submatrix",
                     "This method is not available because there is no guarantee that MatNorm is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::zero()
{
  multiphenics_error("BlockPETScSubMatrix.cpp",
                     "zero sub matrix rows",
                     "This method is not available because there is no guarantee that MatZeroRows is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::apply(AssemblyType type)
{
  multiphenics_error("BlockPETScSubMatrix.cpp",
                     "apply sub matrix changes",
                     "This method is not available because there is no need to apply sub matrix changes, as they are applied on destruction");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::mult(const BlockPETScSubVector& x, BlockPETScSubVector& y) const
{
  multiphenics_error("BlockPETScSubMatrix.cpp",
                     "perform submatrix-subvector product",
                     "This method is not available because there is no guarantee that MatMult is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::scale(PetscScalar a)
{
  multiphenics_error("BlockPETScSubMatrix.cpp",
                     "scale submatrix",
                     "This method is not available because there is no guarantee that MatScale is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
bool BlockPETScSubMatrix::is_symmetric(double tol) const
{
  multiphenics_error("BlockPETScSubMatrix.cpp",
                     "check submatrix symmetry",
                     "This method is not available because there is no guarantee that MatIsSymmetric is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
bool BlockPETScSubMatrix::is_hermitian(double tol) const
{
  multiphenics_error("BlockPETScSubMatrix.cpp",
                     "check if submatrix is hermitian",
                     "This method is not available because there is no guarantee that MatIsHermitian is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::set_options_prefix(std::string options_prefix)
{
  multiphenics_error("BlockPETScSubMatrix.cpp",
                     "set options prefix",
                     "This method is not available because there is no guarantee that MatSetOptionsPrefix is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
std::string BlockPETScSubMatrix::get_options_prefix() const
{
  multiphenics_error("BlockPETScSubMatrix.cpp",
                     "get options prefix",
                     "This method is not available because there is no guarantee that MatGetOptionsPrefix is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::set_from_options()
{
  multiphenics_error("BlockPETScSubMatrix.cpp",
                     "set from options",
                     "This method is not available because there is no guarantee that MatSetFromOptions is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::set_nullspace(const VectorSpaceBasis& nullspace)
{
  multiphenics_error("BlockPETScSubMatrix.cpp",
                     "set nullspace",
                     "This method is not available because there is no guarantee that MatSetNullSpace is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::set_near_nullspace(const VectorSpaceBasis& nullspace)
{
  multiphenics_error("BlockPETScSubMatrix.cpp",
                     "set near nullspace",
                     "This method is not available because there is no guarantee that MatSetNearNullSpace is implemented by PETSc LocalSubMatrix");
}
//-----------------------------------------------------------------------------
void BlockPETScSubMatrix::to_restricted_submatrix_row_indices(
  const std::vector<PetscInt> & block_unrestricted_submatrix_row_indices, std::vector<PetscInt> & block_restricted_submatrix_row_indices,
  std::vector<bool> * is_row_in_restriction
) 
{
  assert(block_restricted_submatrix_row_indices.size() == 0);
  if (is_row_in_restriction != NULL)
      assert(is_row_in_restriction->size() == 0);
  
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
  const std::vector<PetscInt> & block_unrestricted_submatrix_col_indices, std::vector<PetscInt> & block_restricted_submatrix_col_indices,
  std::vector<bool> * is_col_in_restriction
) 
{
  assert(block_restricted_submatrix_col_indices.size() == 0);
  if (is_col_in_restriction != NULL)
      assert(is_col_in_restriction->size() == 0);
  
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
  const std::vector<PetscInt> & block_unrestricted_submatrix_row_indices, std::vector<PetscInt> & block_restricted_submatrix_row_indices,
  const std::vector<PetscInt> & block_unrestricted_submatrix_col_indices, std::vector<PetscInt> & block_restricted_submatrix_col_indices,
  const std::vector<PetscScalar> & block_unrestricted_submatrix_values, std::vector<PetscScalar> & block_restricted_submatrix_values
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
  assert(unrestricted_value_iterator == block_unrestricted_submatrix_values.size());
  assert(block_restricted_submatrix_values.size() == block_restricted_submatrix_row_indices.size()*block_restricted_submatrix_col_indices.size());
}

// Copyright (C) 2016-2021 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <cassert>
#include <vector>
#include <dolfinx/la/PETScVector.h>  // for dolfinx::la::petsc_error
#include <multiphenicsx/la/PETScMatrix.h>

using dolfinx::la::petsc_error;
using multiphenicsx::la::MatSubMatrixWrapper;

//-----------------------------------------------------------------------------
MatSubMatrixWrapper::MatSubMatrixWrapper(
  Mat A,
  std::array<IS, 2> index_sets)
  : _global_matrix(A), _is(index_sets)
{
  PetscErrorCode ierr;

  // Get communicator from matrix object
  MPI_Comm comm = MPI_COMM_NULL;
  ierr = PetscObjectGetComm((PetscObject) A, &comm);
  if (ierr != 0) petsc_error(ierr, __FILE__, "PetscObjectGetComm");

  // Sub matrix inherits block size of the index sets. Check that they
  // are consistent with the ones of the global matrix.
  std::vector<PetscInt> bs_A(2);
  ierr = MatGetBlockSizes(A, &bs_A[0], &bs_A[1]);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatGetBlockSizes");
  std::vector<PetscInt> bs_is(2);
  ierr = ISGetBlockSize(_is[0], &bs_is[0]);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISGetBlockSize");
  ierr = ISGetBlockSize(_is[1], &bs_is[1]);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISGetBlockSize");
  assert(bs_A[0] == bs_is[0]);
  assert(bs_A[1] == bs_is[1]);

  // Extract sub matrix
  ierr = MatGetLocalSubMatrix(A, _is[0], _is[1], &_sub_matrix);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatGetLocalSubMatrix");
}
//-----------------------------------------------------------------------------
MatSubMatrixWrapper::MatSubMatrixWrapper(
  Mat A,
  std::array<IS, 2> unrestricted_index_sets,
  std::array<IS, 2> restricted_index_sets,
  std::array<std::map<std::int32_t, std::int32_t>, 2> unrestricted_to_restricted,
  std::array<int, 2> unrestricted_to_restricted_bs)
  : MatSubMatrixWrapper(A, restricted_index_sets)
{
  PetscErrorCode ierr;

  // Initialization of custom local to global PETSc map.
  // In order not to change the assembly routines, here "local" is intended
  // with respect to the *unrestricted* index sets (which where generated using
  // the index map that will be passed to the assembly routines). Instead,
  // "global" is intended with respect to the *restricted* index sets for
  // entries in the restriction, while it is set to -1 (i.e., values corresponding
  // to those indices will be discarded) for entries not in the restriction.

  // Get sub matrix (i.e., index sets) block sizes
  std::vector<PetscInt> bs(2);
  ierr = MatGetBlockSizes(_sub_matrix, &bs[0], &bs[1]);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatGetBlockSizes");

  // Compare sub matrix block sizes with unrestricted_to_restricted_bs:
  // they should either be the same (typically the case of restricted matrices or restricted
  // nest matrices) or unrestricted_to_restricted_bs may be larger than the sub matrix block
  // sizes (typically the case of restricted block matrices, because bs is forced to one).
  assert(bs[0] == unrestricted_to_restricted_bs[0] || (bs[0] == 1 && unrestricted_to_restricted_bs[0] > 1));
  assert(bs[1] == unrestricted_to_restricted_bs[1] || (bs[1] == 1 && unrestricted_to_restricted_bs[1] > 1));
  std::vector<PetscInt> unrestricted_to_restricted_correction(2);
  for (std::size_t i = 0; i < 2; ++i)
  {
    if (bs[i] == unrestricted_to_restricted_bs[i])
    {
      unrestricted_to_restricted_correction[i] = 1;
    }
    else
    {
      assert(bs[i] == 1);
      unrestricted_to_restricted_correction[i] = unrestricted_to_restricted_bs[i];
    }
  }

  // Get matrix local-to-global map
  std::array<ISLocalToGlobalMapping, 2> petsc_local_to_global_matrix;
  ierr = MatGetLocalToGlobalMapping(A, &petsc_local_to_global_matrix[0], &petsc_local_to_global_matrix[1]);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatGetLocalToGlobalMapping");

  // Allocate data for submatrix local-to-global maps in an STL vector
  std::array<std::vector<PetscInt>, 2> stl_local_to_global_submatrix;
  for (std::size_t i = 0; i < 2; ++i)
  {
    PetscInt unrestricted_is_size;
    ierr = ISBlockGetLocalSize(unrestricted_index_sets[i], &unrestricted_is_size);
    if (ierr != 0) petsc_error(ierr, __FILE__, "ISGetLocalSize");
    stl_local_to_global_submatrix[i].resize(unrestricted_is_size);

    const PetscInt *restricted_indices;
    ierr = ISBlockGetIndices(restricted_index_sets[i], &restricted_indices);
    if (ierr != 0) petsc_error(ierr, __FILE__, "ISGetIndices");

    std::vector<PetscInt> restricted_local_index(1);
    std::vector<PetscInt> restricted_global_index(1);

    for (PetscInt unrestricted_index = 0; unrestricted_index < unrestricted_is_size; unrestricted_index++)
    {
      if (unrestricted_to_restricted[i].count(unrestricted_index/unrestricted_to_restricted_correction[i]) > 0)
      {
        restricted_local_index[0] = restricted_indices[
          unrestricted_to_restricted_correction[i]*unrestricted_to_restricted[i].at(
            unrestricted_index/unrestricted_to_restricted_correction[i]) +
              unrestricted_index%unrestricted_to_restricted_correction[i]];
        ISLocalToGlobalMappingApplyBlock(petsc_local_to_global_matrix[i], restricted_local_index.size(),
                                         restricted_local_index.data(), restricted_global_index.data());
        stl_local_to_global_submatrix[i][unrestricted_index] = restricted_global_index[0];
      }
      else
      {
        stl_local_to_global_submatrix[i][unrestricted_index] = -1;
      }
    }

    ierr = ISBlockRestoreIndices(restricted_index_sets[i], &restricted_indices);
    if (ierr != 0) petsc_error(ierr, __FILE__, "ISRestoreIndices");
  }

  // Get communicator from submatrix object
  MPI_Comm comm = MPI_COMM_NULL;
  ierr = PetscObjectGetComm((PetscObject) _sub_matrix, &comm);
  if (ierr != 0) petsc_error(ierr, __FILE__, "PetscObjectGetComm");

  // Create submatrix local-to-global maps as index set
  std::array<ISLocalToGlobalMapping, 2> petsc_local_to_global_submatrix;
  for (std::size_t i = 0; i < 2; ++i)
  {
    ierr = ISLocalToGlobalMappingCreate(comm, bs[i], stl_local_to_global_submatrix[i].size(),
                                        stl_local_to_global_submatrix[i].data(), PETSC_COPY_VALUES,
                                        &petsc_local_to_global_submatrix[i]);
    if (ierr != 0) petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");
  }

  // Set submatrix local-to-global maps
  ierr = MatSetLocalToGlobalMapping(_sub_matrix, petsc_local_to_global_submatrix[0],
                                    petsc_local_to_global_submatrix[1]);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetLocalToGlobalMapping");

  // Clean up submatrix local-to-global maps
  for (std::size_t i = 0; i < 2; ++i)
  {
    ierr = ISLocalToGlobalMappingDestroy(&petsc_local_to_global_submatrix[i]);
    if (ierr != 0) petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");
  }
}
//-----------------------------------------------------------------------------
MatSubMatrixWrapper::~MatSubMatrixWrapper()
{
  // Sub matrix should have been restored before destroying object
  assert(!_sub_matrix);
  assert(!_is[0]);
  assert(!_is[1]);
}
//-----------------------------------------------------------------------------
void MatSubMatrixWrapper::restore()
{
  // Restore the global matrix
  PetscErrorCode ierr;
  assert(_sub_matrix);
  ierr = MatRestoreLocalSubMatrix(_global_matrix, _is[0], _is[1], &_sub_matrix);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatRestoreLocalSubMatrix");

  // Clear pointers
  _sub_matrix = nullptr;
  _is.fill(nullptr);
}
//-----------------------------------------------------------------------------
Mat MatSubMatrixWrapper::mat() const
{
  return _sub_matrix;
}
//-----------------------------------------------------------------------------

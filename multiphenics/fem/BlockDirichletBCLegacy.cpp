// Copyright (C) 2016-2019 by the multiphenics authors
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

#include <multiphenics/fem/BlockDirichletBCLegacy.h>
#include <multiphenics/fem/DirichletBCLegacy.h>
#include <multiphenics/la/BlockInsertMode.h>
#include <multiphenics/la/BlockPETScSubMatrix.h>
#include <multiphenics/la/BlockPETScSubVector.h>

using namespace multiphenics;
using namespace multiphenics::fem;

using dolfin::fem::DirichletBC;
using dolfin::la::petsc_error;
using dolfin::la::PETScMatrix;
using dolfin::la::PETScVector;
using multiphenics::la::BlockInsertMode;
using multiphenics::la::BlockPETScSubMatrix;
using multiphenics::la::BlockPETScSubVector;

//-----------------------------------------------------------------------------
void BlockDirichletBCLegacy::apply(const BlockDirichletBC& bcs, PETScMatrix& A, PetscScalar diag)
{
  for (std::size_t I(0); I < bcs.size(); ++I)
  {
    PetscErrorCode ierr;
    // MatZeroRowsLocal does not work with local indices to submatrices A_{IJ}, only works on matrix A, so we cannot
    // delegate application of BCs to the legacy DirichletBCLegacy::apply(bcs[I], A_{IJ}, diag*\delta_{IJ}). Furthermore,
    // since only row indices are required (i.e., local-to-global maps for rows), it is not necessary to loop overy J.
    // Thus, for simplicity extract only diagonal A_{II} blocks.
    std::shared_ptr<BlockPETScSubMatrix> A_II = std::make_shared<BlockPETScSubMatrix>(A, I, I, bcs.block_function_space()->block_dofmap(), bcs.block_function_space()->block_dofmap(), BlockInsertMode::INSERT_VALUES);
    // Get matrix local-to-global maps, set by the constructor of *A_II
    ISLocalToGlobalMapping petsc_local_to_global0, petsc_local_to_global1__unused;
    ierr = MatGetLocalToGlobalMapping(A_II->mat(), &petsc_local_to_global0,
                                      &petsc_local_to_global1__unused);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatGetLocalToGlobalMapping");
    // Get all dof indices stored in the I-th bcs
    std::vector<PetscInt> bc_indices_I;
    for (auto bc: bcs[I])
    {
      const auto bc_indices_eigen = bc->dof_indices();
      std::vector<PetscInt> bc_indices;
      bc_indices.insert(bc_indices.end(), bc_indices_eigen.data(), bc_indices_eigen.data() + bc_indices_eigen.size());
      std::vector<PetscInt> restricted_bc_indices;
      A_II->to_restricted_submatrix_row_indices(bc_indices, restricted_bc_indices);
      bc_indices_I.insert(bc_indices_I.end(), restricted_bc_indices.begin(), restricted_bc_indices.end());
    }
    // Convert submatrix local indices to matrix global indices
    std::vector<PetscInt> matrix_global_row_indices(bc_indices_I.size());
    ISLocalToGlobalMappingApply(petsc_local_to_global0, bc_indices_I.size(), bc_indices_I.data(), matrix_global_row_indices.data());
    // Clean up A_II, as its goal was only to use the internal setup of local-to-global maps. Note that there is no need
    // to clean up local-to-global maps allocated here, as this will be taken care of by the internal call to MatRestoreLocalSubMatrix
    // carried out by ~BlockPETScSubMatrix.
    A_II.reset();
    // Keep nonzero structure after calling MatZeroRows
    ierr = MatSetOption(A.mat(), MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetOption");
    // Call MatZeroRows. Note that this will clear {A_{IJ} for all J}, even though local-to-global maps were extracted
    // from A_{II} only, because we are now clearing the global matrix A (i.e., global rows).
    std::size_t MatZeroRows_called = dolfin::MPI::sum(A.mpi_comm(), bc_indices_I.size());
    if (bc_indices_I.size() > 0)
    {
      ierr = MatZeroRows(A.mat(), matrix_global_row_indices.size(), matrix_global_row_indices.data(), diag, NULL, NULL);
      if (ierr != 0) petsc_error(ierr, __FILE__, "MatZeroRows");
    }
    // Need to place a dummy call to MatZeroRows to avoid deadlocks in parallel
    // if any call to MatZeroRows has been done by other processes, but no Dirichlet
    // indices are found on the current one.
    if (MatZeroRows_called > 0 && bc_indices_I.size() == 0)
    {
      ierr = MatZeroRows(A.mat(), 0, NULL, diag, NULL, NULL);
      if (ierr != 0) petsc_error(ierr, __FILE__, "MatZeroRows");
    }
  }
}
//-----------------------------------------------------------------------------
void BlockDirichletBCLegacy::apply(const BlockDirichletBC& bcs, PETScVector& b)
{
  for (std::size_t I(0); I < bcs.size(); ++I)
  {
    // DirichletBCLegacy::apply operates on the plain Vec object, thus neglecting restrictions.
    // We partially replicate here the implementation of DirichletBCLegacy::apply in order to make use of the PETScVector interface,
    // which we have patched to handle the restriction of subvectors, as we do in multiphenics::fem::block_assemble.
    std::vector<PetscInt> unrestricted_rows;
    std::vector<PetscScalar> unrestricted_bc_values;
    BlockDirichletBCLegacy::_apply(bcs[I], unrestricted_rows, unrestricted_bc_values);
    std::shared_ptr<PETScVector> b_I = std::make_shared<BlockPETScSubVector>(b, I, bcs.block_function_space()->block_dofmap(), BlockInsertMode::INSERT_VALUES);
    b_I->set_local(unrestricted_bc_values.data(), unrestricted_bc_values.size(), unrestricted_rows.data());
  }
}
//-----------------------------------------------------------------------------
void BlockDirichletBCLegacy::apply(const BlockDirichletBC& bcs, PETScVector& b, const PETScVector& x)
{
  for (std::size_t I(0); I < bcs.size(); ++I)
  {
    std::vector<PetscInt> unrestricted_rows;
    std::vector<PetscScalar> unrestricted_bc_values;
    BlockDirichletBCLegacy::_apply(bcs[I], unrestricted_rows, unrestricted_bc_values);
    std::shared_ptr<PETScVector> x_I = std::make_shared<BlockPETScSubVector>(x, I, bcs.block_function_space()->block_dofmap(), BlockInsertMode::INSERT_VALUES);
    std::vector<PetscScalar> unrestricted_x_values(unrestricted_rows.size());
    x_I->get_local(unrestricted_x_values.data(), unrestricted_x_values.size(), unrestricted_rows.data());
    for (std::size_t i(0); i < unrestricted_x_values.size(); ++i)
      unrestricted_bc_values[i] = unrestricted_x_values[i] - unrestricted_bc_values[i];
    std::shared_ptr<PETScVector> b_I = std::make_shared<BlockPETScSubVector>(b, I, bcs.block_function_space()->block_dofmap(), BlockInsertMode::INSERT_VALUES);
    b_I->set_local(unrestricted_bc_values.data(), unrestricted_bc_values.size(), unrestricted_rows.data());
  }
}
//-----------------------------------------------------------------------------
void BlockDirichletBCLegacy::_apply(std::vector<std::shared_ptr<const DirichletBC>> bcs,
                                    std::vector<PetscInt>& unrestricted_rows,
                                    std::vector<PetscScalar>& unrestricted_values)
{
  for (auto bc: bcs)
  {
    const auto index_map = bc->function_space()->dofmap()->index_map();
    Eigen::Array<PetscInt, Eigen::Dynamic, 1> bc_indices;
    Eigen::Array<PetscScalar, Eigen::Dynamic, 1> bc_values;
    std::tie(bc_indices, bc_values) = bc->bcs();
    std::vector<PetscInt> unrestricted_rows_bc;
    std::vector<PetscScalar> unrestricted_values_bc;
    unrestricted_rows_bc.reserve(bc_indices.size());   // actual size might be less than this,
    unrestricted_values_bc.reserve(bc_indices.size()); // because ghost nodes are skipped
    for (Eigen::Index i = 0; i < bc_indices.size(); ++i)
    {
      if (bc_indices[i] < index_map->block_size()*index_map->size_local()) // skip indices associated to ghosts
      {
        unrestricted_rows_bc.push_back(bc_indices[i]);
        unrestricted_values_bc.push_back(bc_values[i]);
      }
    }
    unrestricted_rows.insert(unrestricted_rows.end(), unrestricted_rows_bc.begin(), unrestricted_rows_bc.end());
    unrestricted_values.insert(unrestricted_values.end(), unrestricted_values_bc.begin(), unrestricted_values_bc.end());
  }
}
//-----------------------------------------------------------------------------

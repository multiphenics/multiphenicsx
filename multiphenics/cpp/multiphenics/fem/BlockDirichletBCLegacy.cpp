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
#include <multiphenics/la/BlockPETScSubVectorReadWrapper.h>
#include <multiphenics/la/BlockPETScSubVectorWrapper.h>

using namespace multiphenics;
using namespace multiphenics::fem;

using dolfin::fem::DirichletBC;
using dolfin::la::petsc_error;
using multiphenics::la::BlockPETScSubVectorReadWrapper;
using multiphenics::la::BlockPETScSubVectorWrapper;

//-----------------------------------------------------------------------------
void BlockDirichletBCLegacy::apply(const BlockDirichletBC& bcs, Mat A, PetscScalar diag)
{
  PetscErrorCode ierr;
  // Keep nonzero structure after calling MatZeroRows
  ierr = MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetOption");

  // Submatrices A_{IJ} do not support MatZeroRowsLocal, so we cannot delegate application of BCs to the legacy 
  // DirichletBCLegacy::apply(bcs[I], A_{IJ}, diag*\delta_{IJ}). 
  // We will thus operator directly on the global matrix A.  
  const auto block_dofmap = bcs.block_function_space()->block_dofmap();
  const auto block_index_map = block_dofmap->index_map;
  for (std::size_t I(0); I < bcs.size(); ++I)
  {
    // Extract global indices associated to bcs[I]
    std::vector<PetscInt> block_global_indices;
    const auto & original_to_block = block_dofmap->original_to_block(I);
    for (auto bc: bcs[I])
    {
      const auto original_local_indices = bc->dof_indices();
      for (Eigen::Index o = 0; o < original_local_indices.size(); ++o)
      {
        if (original_to_block.count(original_local_indices[o]) > 0) // skip all dofs which have been removed by restriction
        {
          block_global_indices.push_back(
            block_index_map->local_to_global(original_to_block.at(original_local_indices[o]))
          );
        }
      }
    }
    // Call MatZeroRows. Note that this will clear {A_{IJ} for all J != I}, because we are now clearing the 
    // global matrix A.
    MPI_Comm mpi_comm = MPI_COMM_NULL;
    ierr = PetscObjectGetComm((PetscObject) A, &mpi_comm);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PetscObjectGetComm");
    std::size_t MatZeroRows_called = dolfin::MPI::sum(mpi_comm, block_global_indices.size());
    if (block_global_indices.size() > 0)
    {
      ierr = MatZeroRows(A, block_global_indices.size(), block_global_indices.data(), diag, NULL, NULL);
      if (ierr != 0) petsc_error(ierr, __FILE__, "MatZeroRows");
    }
    // Need to place a dummy call to MatZeroRows to avoid deadlocks in parallel
    // if any call to MatZeroRows has been done by other processes, but no Dirichlet
    // indices are found on the current one.
    if (MatZeroRows_called > 0 && block_global_indices.size() == 0)
    {
      ierr = MatZeroRows(A, 0, NULL, diag, NULL, NULL);
      if (ierr != 0) petsc_error(ierr, __FILE__, "MatZeroRows");
    }
  }
}
//-----------------------------------------------------------------------------
void BlockDirichletBCLegacy::apply(const BlockDirichletBC& bcs, Vec b)
{
  // This method is adapted from
  //    dolfin::fem::set_bc in dolfin/fem/assembler.cpp
  
  for (std::size_t I(0); I < bcs.size(); ++I)
  {
    BlockPETScSubVectorWrapper b_I(b, I, bcs.block_function_space()->block_dofmap(), INSERT_VALUES);
    for (auto bc: bcs[I])
    {
      assert(bc);
      bc->set(b_I.content);
    }
  }
  // Finalize assembly of global tensor
  PetscErrorCode ierr;
  ierr = VecGhostUpdateBegin(b, INSERT_VALUES, SCATTER_FORWARD);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostUpdateBegin");
  ierr = VecGhostUpdateEnd(b, INSERT_VALUES, SCATTER_FORWARD);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostUpdateEnd");
}
//-----------------------------------------------------------------------------
void BlockDirichletBCLegacy::apply(const BlockDirichletBC& bcs, Vec b, const Vec x)
{
  // This method is adapted from
  //    dolfin::fem::set_bc in dolfin/fem/assembler.cpp
  
  for (std::size_t I(0); I < bcs.size(); ++I)
  {
    BlockPETScSubVectorWrapper b_I(b, I, bcs.block_function_space()->block_dofmap(), INSERT_VALUES);
    BlockPETScSubVectorReadWrapper x_I(x, I, bcs.block_function_space()->block_dofmap());
    for (auto bc: bcs[I])
    {
      assert(bc);
      bc->set(b_I.content, x_I.content, -1.);
    }
  }
  // Finalize assembly of global tensor
  PetscErrorCode ierr;
  ierr = VecGhostUpdateBegin(b, INSERT_VALUES, SCATTER_FORWARD);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostUpdateBegin");
  ierr = VecGhostUpdateEnd(b, INSERT_VALUES, SCATTER_FORWARD);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostUpdateEnd");
}
//-----------------------------------------------------------------------------

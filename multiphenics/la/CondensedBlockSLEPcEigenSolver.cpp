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

#ifdef HAS_SLEPC

#include <multiphenics/la/CondensedBlockSLEPcEigenSolver.h>

using namespace dolfin;
using namespace multiphenics;

//-----------------------------------------------------------------------------
CondensedBlockSLEPcEigenSolver::CondensedBlockSLEPcEigenSolver(MPI_Comm comm):
  CondensedSLEPcEigenSolver(comm)
{
}
//-----------------------------------------------------------------------------
CondensedBlockSLEPcEigenSolver::CondensedBlockSLEPcEigenSolver(EPS eps) :
  CondensedSLEPcEigenSolver(eps)
{
}
//-----------------------------------------------------------------------------
CondensedBlockSLEPcEigenSolver::CondensedBlockSLEPcEigenSolver(std::shared_ptr<const BlockPETScMatrix> A,
                                                               std::shared_ptr<const BlockDirichletBC> block_bcs)
  : CondensedBlockSLEPcEigenSolver(A, nullptr, block_bcs)
{
  // Do nothing (handled by other constructor)
}
//-----------------------------------------------------------------------------
CondensedBlockSLEPcEigenSolver::CondensedBlockSLEPcEigenSolver(MPI_Comm comm, std::shared_ptr<const BlockPETScMatrix> A,
                                                               std::shared_ptr<const BlockDirichletBC> block_bcs)
  : CondensedBlockSLEPcEigenSolver(comm, A, nullptr, block_bcs)
{
  // Do nothing (handled by other constructor)
}
//-----------------------------------------------------------------------------
CondensedBlockSLEPcEigenSolver::CondensedBlockSLEPcEigenSolver(std::shared_ptr<const BlockPETScMatrix> A,
                                                               std::shared_ptr<const BlockPETScMatrix> B,
                                                               std::shared_ptr<const BlockDirichletBC> block_bcs)
  : CondensedSLEPcEigenSolver(A->mpi_comm())
{
  set_boundary_conditions(block_bcs);
  set_operators(A, B);
}
//-----------------------------------------------------------------------------
CondensedBlockSLEPcEigenSolver::CondensedBlockSLEPcEigenSolver(MPI_Comm comm,
                                                               std::shared_ptr<const BlockPETScMatrix> A,
                                                               std::shared_ptr<const BlockPETScMatrix> B,
                                                               std::shared_ptr<const BlockDirichletBC> block_bcs)
  : CondensedSLEPcEigenSolver(comm)
{
  set_boundary_conditions(block_bcs);
  set_operators(A, B);
}
//-----------------------------------------------------------------------------
CondensedBlockSLEPcEigenSolver::~CondensedBlockSLEPcEigenSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void CondensedBlockSLEPcEigenSolver::set_boundary_conditions(std::shared_ptr<const BlockDirichletBC> block_bcs)
{
  // Get dofmap and related quantities
  auto comm = block_bcs->block_function_space()->mesh()->mpi_comm();
  auto block_dofmap = block_bcs->block_function_space()->block_dofmap();
  auto local_range = block_dofmap->ownership_range();
  
  // List all constrained local dofs
  std::set<la_index> constrained_local_dofs;
  BlockDirichletBC::Map bc_local_indices_to_values;
  block_bcs->get_boundary_values(bc_local_indices_to_values);
  for (auto & bc_local_index_to_value : bc_local_indices_to_values)
    constrained_local_dofs.insert(block_dofmap->local_to_global_index(bc_local_index_to_value.first));
  
  // List all unconstrained dofs
  std::vector<la_index> local_dofs(local_range.second - local_range.first);
  std::iota(local_dofs.begin(), local_dofs.end(), local_range.first);
  std::vector<la_index> unconstrained_local_dofs;
  std::set_difference(local_dofs.begin(), local_dofs.end(), 
                      constrained_local_dofs.begin(), constrained_local_dofs.end(), 
                      std::inserter(unconstrained_local_dofs, unconstrained_local_dofs.begin()));
                      
  // Generate IS accordingly
  PetscErrorCode ierr;
  ierr = ISCreateGeneral(comm, unconstrained_local_dofs.size(), unconstrained_local_dofs.data(),
                         PETSC_COPY_VALUES, &_is);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISCreateGeneral");
}
//-----------------------------------------------------------------------------
void CondensedBlockSLEPcEigenSolver::get_eigenpair(double& lr, double& lc,
                                                   GenericVector& r, GenericVector& c,
                                                   std::size_t i) const
{
  CondensedSLEPcEigenSolver::get_eigenpair(lr, lc, r, c, i);
}
//-----------------------------------------------------------------------------
void CondensedBlockSLEPcEigenSolver::get_eigenpair(double& lr, double& lc,
                                                   PETScVector& r, PETScVector& c,
                                                   std::size_t i) const
{
  CondensedSLEPcEigenSolver::get_eigenpair(lr, lc, r, c, i);
}
//-----------------------------------------------------------------------------

#endif

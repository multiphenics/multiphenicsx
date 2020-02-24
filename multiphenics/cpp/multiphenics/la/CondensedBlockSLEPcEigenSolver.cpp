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

#ifdef HAS_SLEPC

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <multiphenics/la/CondensedBlockSLEPcEigenSolver.h>

using namespace multiphenics;
using namespace multiphenics::la;

using dolfinx::la::petsc_error;
using multiphenics::fem::BlockDirichletBC;

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
CondensedBlockSLEPcEigenSolver::~CondensedBlockSLEPcEigenSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void CondensedBlockSLEPcEigenSolver::set_boundary_conditions(std::shared_ptr<const BlockDirichletBC> block_bcs)
{
  // Get dofmap and related quantities
  const auto comm = block_bcs->block_function_space()->mesh()->mpi_comm();
  const auto block_dofmap = block_bcs->block_function_space()->block_dofmap();
  assert(block_dofmap->index_map->block_size() == 1);
  const auto block_index_map = block_dofmap->index_map;
  auto local_range = block_index_map->local_range();

  // List all constrained local dofs
  std::set<PetscInt> constrained_local_dofs;
  for (std::size_t I(0); I < block_bcs->size(); ++I)
  {
    const auto & original_to_block = block_dofmap->original_to_block(I);
    for (auto bc: block_bcs->operator[](I))
    {
      const auto original_local_indices = bc->dofs_owned().col(0);
      for (Eigen::Index o = 0; o < original_local_indices.size(); ++o)
      {
        if (original_to_block.count(original_local_indices[o]) > 0) // skip all dofs which have been removed by restriction
        {
          constrained_local_dofs.insert(
            block_index_map->local_to_global(original_to_block.at(original_local_indices[o]))
          );
        }
      }
    }
  }

  // List all unconstrained dofs
  std::vector<PetscInt> local_dofs(local_range[1] - local_range[0]);
  std::iota(local_dofs.begin(), local_dofs.end(), local_range[0]);
  std::vector<PetscInt> unconstrained_local_dofs;
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

#endif

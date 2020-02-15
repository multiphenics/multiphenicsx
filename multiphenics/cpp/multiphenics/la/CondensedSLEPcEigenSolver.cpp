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
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/la/utils.h>
#include <multiphenics/la/CondensedSLEPcEigenSolver.h>

using dolfinx::fem::DirichletBC;
using dolfinx::la::CondensedSLEPcEigenSolver;
using dolfinx::la::petsc_error;

//-----------------------------------------------------------------------------
CondensedSLEPcEigenSolver::CondensedSLEPcEigenSolver(MPI_Comm comm):
  SLEPcEigenSolver(comm)
{
}
//-----------------------------------------------------------------------------
CondensedSLEPcEigenSolver::CondensedSLEPcEigenSolver(EPS eps) :
  SLEPcEigenSolver(eps)
{
}
//-----------------------------------------------------------------------------
CondensedSLEPcEigenSolver::~CondensedSLEPcEigenSolver()
{
  PetscErrorCode ierr;
  ierr = ISDestroy(&_is);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISDestroy");

  if (_condensed_A) {
    ierr = MatDestroy(&_condensed_A);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatDestroy");
  }

  if (_condensed_B) {
    ierr = MatDestroy(&_condensed_B);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatDestroy");
  }
}
//-----------------------------------------------------------------------------
void CondensedSLEPcEigenSolver::set_boundary_conditions(std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  // Get dofmap and related quantities
  auto comm = bcs[0]->function_space()->mesh()->mpi_comm();
  auto dofmap = bcs[0]->function_space()->dofmap();
  #ifdef DEBUG
  for (auto & bc : bcs)
  {
    assert(comm == bc->function_space()->mesh()->mpi_comm());
    assert(dofmap == bc->function_space()->dofmap());
  }
  #endif
  auto local_range = dofmap->index_map->local_range();
  int dofmap_block_size = dofmap->index_map->block_size;

  // List all constrained local dofs
  std::set<PetscInt> constrained_local_dofs;
  for (auto & bc : bcs)
  {
    const auto bc_local_dofs = bc->dofs_owned().col(0);
    for (Eigen::Index i = 0; i < bc_local_dofs.size(); ++i)
    {
      constrained_local_dofs.insert(dofmap->index_map->local_to_global(bc_local_dofs[i]/dofmap_block_size)*dofmap_block_size + (bc_local_dofs[i]%dofmap_block_size));
    }
  }

  // List all unconstrained dofs
  std::vector<PetscInt> local_dofs(dofmap_block_size*(local_range[1] - local_range[0]));
  std::iota(local_dofs.begin(), local_dofs.end(), dofmap_block_size*local_range[0]);
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
void CondensedSLEPcEigenSolver::set_operators(const Mat A, const Mat B)
{
  _A = A;
  _condensed_A = _condense_matrix(A);
  if (B)
  {
    _B = B;
    _condensed_B = _condense_matrix(B);
    EPSSetOperators(this->eps(), _condensed_A, _condensed_B);
  }
  else
  {
    EPSSetOperators(this->eps(), _condensed_A, NULL);
  }
}
//-----------------------------------------------------------------------------
Mat CondensedSLEPcEigenSolver::_condense_matrix(const Mat mat) const
{
  PetscErrorCode ierr;

  Mat condensed_mat;
  ierr = MatCreateSubMatrix(mat, _is, _is, MAT_INITIAL_MATRIX, &condensed_mat);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatCreateSubMatrix");

  return condensed_mat;
}
//-----------------------------------------------------------------------------
void CondensedSLEPcEigenSolver::get_eigenpair(double& lr, double& lc,
                                              Vec r, Vec c,
                                              std::size_t i) const
{
  const PetscInt ii = static_cast<PetscInt>(i);

  // Get number of computed eigenvectors/values
  PetscInt num_computed_eigenvalues;
  EPSGetConverged(this->eps(), &num_computed_eigenvalues);

  if (ii < num_computed_eigenvalues)
  {
    PetscErrorCode ierr;

    // Condense input vectors
    Vec condensed_r_vec;
    ierr = VecGetSubVector(r, _is, &condensed_r_vec);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetSubVector");
    Vec condensed_c_vec;
    ierr = VecGetSubVector(c, _is, &condensed_c_vec);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetSubVector");

    // Get eigen pairs (as in Parent)
    EPSGetEigenpair(this->eps(), ii, &lr, &lc, condensed_r_vec, condensed_c_vec);

    // Restore input vectors
    ierr = VecRestoreSubVector(r, _is, &condensed_r_vec);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecRestoreSubVector");
    ierr = VecRestoreSubVector(c, _is, &condensed_c_vec);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecRestoreSubVector");
  }
  else
  {
    throw std::runtime_error("Cannot extract eigenpair from SLEPc eigenvalue solver. "
                             "Requested eigenpair has not been computed");
  }
}
//-----------------------------------------------------------------------------

#endif

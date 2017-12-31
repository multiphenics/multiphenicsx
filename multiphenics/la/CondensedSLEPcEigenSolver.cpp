// Copyright (C) 2016-2018 by the multiphenics authors
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

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <multiphenics/log/log.h>
#include <multiphenics/la/CondensedSLEPcEigenSolver.h>

using namespace dolfin;

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
CondensedSLEPcEigenSolver::CondensedSLEPcEigenSolver(std::shared_ptr<const PETScMatrix> A,
                                                     std::vector<std::shared_ptr<const DirichletBC>> bcs)
  : CondensedSLEPcEigenSolver(A, nullptr, bcs)
{
  // Do nothing (handled by other constructor)
}
//-----------------------------------------------------------------------------
CondensedSLEPcEigenSolver::CondensedSLEPcEigenSolver(MPI_Comm comm, std::shared_ptr<const PETScMatrix> A,
                                                     std::vector<std::shared_ptr<const DirichletBC>> bcs)
  : CondensedSLEPcEigenSolver(comm, A, nullptr, bcs)
{
  // Do nothing (handled by other constructor)
}
//-----------------------------------------------------------------------------
CondensedSLEPcEigenSolver::CondensedSLEPcEigenSolver(std::shared_ptr<const PETScMatrix> A,
                                                     std::shared_ptr<const PETScMatrix> B,
                                                     std::vector<std::shared_ptr<const DirichletBC>> bcs)
  : SLEPcEigenSolver(A->mpi_comm())
{
  set_boundary_conditions(bcs);
  set_operators(A, B);
}
//-----------------------------------------------------------------------------
CondensedSLEPcEigenSolver::CondensedSLEPcEigenSolver(MPI_Comm comm,
                                                     std::shared_ptr<const PETScMatrix> A,
                                                     std::shared_ptr<const PETScMatrix> B,
                                                     std::vector<std::shared_ptr<const DirichletBC>> bcs)
  : SLEPcEigenSolver(comm)
{
  set_boundary_conditions(bcs);
  set_operators(A, B);
}
//-----------------------------------------------------------------------------
CondensedSLEPcEigenSolver::~CondensedSLEPcEigenSolver()
{
  PetscErrorCode ierr;
  ierr = ISDestroy(&_is);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISDestroy");
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
    dolfin_assert(comm == bc->function_space()->mesh()->mpi_comm());
    dolfin_assert(dofmap == bc->function_space()->dofmap());
  }
  #endif
  auto local_range = dofmap->ownership_range();
  
  // List all constrained local dofs
  std::set<la_index> constrained_local_dofs;
  for (auto & bc : bcs)
  {
    DirichletBC::Map bc_local_indices_to_values;
    bc->get_boundary_values(bc_local_indices_to_values);
    for (auto & bc_local_index_to_value : bc_local_indices_to_values)
      constrained_local_dofs.insert(dofmap->local_to_global_index(bc_local_index_to_value.first));
  }
  
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
void CondensedSLEPcEigenSolver::set_operators(std::shared_ptr<const PETScMatrix> A,
                                              std::shared_ptr<const PETScMatrix> B)
{
  _A = A;
  _condensed_A = _condense_matrix(A);
  if (B)
  {
    _B = B;
    _condensed_B = _condense_matrix(B);
    EPSSetOperators(this->eps(), _condensed_A->mat(), _condensed_B->mat());
  }
  else
  {
    EPSSetOperators(this->eps(), _condensed_A->mat(), NULL);
  }
}
//-----------------------------------------------------------------------------
std::shared_ptr<const PETScMatrix> CondensedSLEPcEigenSolver::_condense_matrix(std::shared_ptr<const PETScMatrix> mat)
{
  PetscErrorCode ierr;
  
  Mat condensed_mat;
  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 7
  ierr = MatGetSubMatrix(mat->mat(), _is, _is, MAT_INITIAL_MATRIX, &condensed_mat);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatGetSubMatrix");
  #else
  ierr = MatCreateSubMatrix(mat->mat(), _is, _is, MAT_INITIAL_MATRIX, &condensed_mat);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatCreateSubMatrix");
  #endif
  
  return std::make_shared<const PETScMatrix>(condensed_mat);
}
//-----------------------------------------------------------------------------
void CondensedSLEPcEigenSolver::get_eigenpair(double& lr, double& lc,
                                              GenericVector& r, GenericVector& c,
                                              std::size_t i) const
{
  PETScVector& _r = as_type<PETScVector>(r);
  PETScVector& _c = as_type<PETScVector>(c);
  get_eigenpair(lr, lc, _r, _c, i);
}
//-----------------------------------------------------------------------------
void CondensedSLEPcEigenSolver::get_eigenpair(double& lr, double& lc,
                                              PETScVector& r, PETScVector& c,
                                              std::size_t i) const
{
  const PetscInt ii = static_cast<PetscInt>(i);

  // Get number of computed eigenvectors/values
  PetscInt num_computed_eigenvalues;
  EPSGetConverged(this->eps(), &num_computed_eigenvalues);

  if (ii < num_computed_eigenvalues)
  {
    PetscErrorCode ierr;
    
    // Initialize input vectors, if needed
    _A->init_vector(r, 0);
    _A->init_vector(c, 0);
    
    // Condense input vectors
    Vec condensed_r_vec;
    ierr = VecCreate(r.mpi_comm(), &condensed_r_vec);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecCreate");
    ierr = VecGetSubVector(r.vec(), _is, &condensed_r_vec);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetSubVector");
    Vec condensed_c_vec;
    ierr = VecCreate(c.mpi_comm(), &condensed_c_vec);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecCreate");
    ierr = VecGetSubVector(c.vec(), _is, &condensed_c_vec);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetSubVector");
    
    // Get eigen pairs (as in Parent)
    EPSGetEigenpair(this->eps(), ii, &lr, &lc, condensed_r_vec, condensed_c_vec);
    
    // Restore input vectors
    ierr = VecRestoreSubVector(r.vec(), _is, &condensed_r_vec);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecRestoreSubVector");
    ierr = VecRestoreSubVector(c.vec(), _is, &condensed_c_vec);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecRestoreSubVector");
  }
  else
  {
    multiphenics_error("SLEPcEigenSolver.cpp",
                       "extract eigenpair from SLEPc eigenvalue solver",
                       "Requested eigenpair has not been computed");
  }
}
//-----------------------------------------------------------------------------

#endif

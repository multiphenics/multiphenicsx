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

#include <multiphenics/fem/DirichletBCLegacy.h>
#include <multiphenics/la/BlockInsertMode.h>
#include <multiphenics/la/BlockPETScSubMatrix.h>
#include <multiphenics/la/BlockPETScSubVector.h>

using namespace dolfin;
using namespace dolfin::fem;

using dolfin::la::PETScMatrix;
using dolfin::la::PETScVector;

//-----------------------------------------------------------------------------
void DirichletBCLegacy::apply(std::vector<std::shared_ptr<const DirichletBC>> bcs, PETScMatrix& A, PetscScalar diag)
{
  for (auto bc: bcs)
  {
    const auto rows = bc->dof_indices();
    MatZeroRowsLocal(A.mat(), rows.size(), rows.data(), diag, NULL, NULL);
  }
}
//-----------------------------------------------------------------------------
void DirichletBCLegacy::apply(std::vector<std::shared_ptr<const DirichletBC>> bcs, PETScVector& b)
{
  // This method is adapted from
  //    dolfin::fem::set_bc in dolfin/fem/assemble_vector_impl.cpp
  
  PetscInt b_local_size;
  VecGetLocalSize(b.vec(), &b_local_size);
  PetscScalar* b_values;
  VecGetArray(b.vec(), &b_values);
  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b_vec(b_values, b_local_size);
  
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> bc_indices;
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> bc_values;
  
  for (auto bc: bcs)
  {
    std::tie(bc_indices, bc_values) = bc->bcs();
    for (Eigen::Index i = 0; i < bc_indices.size(); ++i)
      if (bc_indices[i] < b_local_size) // skip indices associated to ghosts
        b_vec[bc_indices[i]] = bc_values[i];
  }
  
  VecRestoreArray(b.vec(), &b_values);
}
//-----------------------------------------------------------------------------
void DirichletBCLegacy::apply(std::vector<std::shared_ptr<const DirichletBC>> bcs, PETScVector& b, const PETScVector& x)
{
  // This method is inspired by
  //    dolfin::fem::set_bc in dolfin/fem/assemble_vector_impl.cpp
  
  PetscInt b_local_size;
  VecGetLocalSize(b.vec(), &b_local_size);
  PetscInt x_local_size;
  VecGetLocalSize(x.vec(), &x_local_size);
  assert(x_local_size == local_size);
  PetscScalar* b_values;
  VecGetArray(b.vec(), &b_values);
  PetscScalar* x_values;
  VecGetArray(x.vec(), &x_values);
  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b_vec(b_values, b_local_size);
  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> x_vec(x_values, x_local_size);
  
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> bc_indices;
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> bc_values;
  
  for (auto bc: bcs)
  {
    std::tie(bc_indices, bc_values) = bc->bcs();
    for (Eigen::Index i = 0; i < bc_indices.size(); ++i)
      if (bc_indices[i] < b_local_size) // skip indices associated to ghosts
        b_vec[bc_indices[i]] = x_vec[bc_indices[i]] - bc_values[i];
  }
  
  VecRestoreArray(b.vec(), &b_values);
  VecRestoreArray(x.vec(), &x_values);
}
//-----------------------------------------------------------------------------

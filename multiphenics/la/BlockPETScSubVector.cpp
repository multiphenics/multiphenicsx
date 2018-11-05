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

#include <multiphenics/la/BlockPETScSubVector.h>
#include <multiphenics/log/log.h>

using namespace multiphenics;
using namespace multiphenics::la;

using dolfin::la::Norm;
using dolfin::la::petsc_error;
using dolfin::la::PETScVector;
using multiphenics::fem::BlockDofMap;

//-----------------------------------------------------------------------------
BlockPETScSubVector::BlockPETScSubVector(
  const PETScVector & x,
  std::size_t block_i,
  std::shared_ptr<const BlockDofMap> block_dof_map,
  BlockInsertMode insert_mode
) : PETScVector(), _global_vector(x),
    _original_to_sub_block(block_dof_map->original_to_sub_block(block_i)),
    _original_to_block(block_dof_map->original_to_block(block_i))
{
  PetscErrorCode ierr;
  
  // Initialize PETSc insert mode
  if (insert_mode == BlockInsertMode::INSERT_VALUES)
    _insert_mode = /* PETSc */ INSERT_VALUES;
  else if (insert_mode == BlockInsertMode::ADD_VALUES)
    _insert_mode = /* PETSc */ ADD_VALUES;
  else
    multiphenics_error("BlockPETScSubVector.cpp",
                       "initialize sub vector",
                       "Invalid value for insert mode");
  
  // Extract sub vector
  const auto & block_owned_dofs__global_numbering = block_dof_map->block_owned_dofs__global_numbering(block_i);
  ierr = ISCreateGeneral(_global_vector.mpi_comm(), block_owned_dofs__global_numbering.size(), block_owned_dofs__global_numbering.data(),
                         PETSC_USE_POINTER, &_is);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISCreateGeneral");
  
  ierr = VecCreate(_global_vector.mpi_comm(), &_x);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecCreate");
  
  // We cannot use
  //VecGetSubVector(_global_vector.vec(), _is, &_x);
  // because of the hardcoded INSERT_VALUES mode in VecRestoreSubVector
  // --- from VecGetSubVector --- //
  PetscInt n, N;
  VecType vec_type;
  ierr = ISGetLocalSize(_is, &n);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISGetLocalSize");
  ierr = ISGetSize(_is, &N);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISGetSize");
  ierr = VecSetSizes(_x, n, N);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetSizes");
  ierr = VecGetType(_global_vector.vec(), &vec_type);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecGetType");
  ierr = VecSetType(_x, vec_type);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecSetType");
  ierr = VecScatterCreate(_global_vector.vec(), _is, _x, NULL, &_scatter);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterCreate");
  ierr = VecScatterBegin(_scatter, _global_vector.vec(), _x, _insert_mode, SCATTER_FORWARD);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterBegin");
  ierr = VecScatterEnd(_scatter, _global_vector.vec(), _x, _insert_mode, SCATTER_FORWARD);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterEnd");
  // --- end --- from VecGetSubVector --- end --- //
  
  // Moreover, if the insert mode is ADD_VALUES we must clear out the subvector
  // in order not to add twice values already stored in global_vector
  if (_insert_mode == ADD_VALUES)
    VecZeroEntries(_x);
  
  // Initialization of local to global PETSc map.
  // Here "global" is inteded with respect to the subvector _x
  // (compare to BlockPETScSubMatrix).
  // --- from PETScVector::_init and PETScMatrix::init --- //
  // Create pointers to PETSc IndexSet for local-to-global map
  auto sub_index_map = block_dof_map->sub_index_map(block_i);
  ISLocalToGlobalMapping petsc_local_to_global;
  assert(sub_index_map->block_size() == 1);
  std::vector<PetscInt> map(sub_index_map->size_local() + sub_index_map->num_ghosts());
  for (std::size_t i = 0; i < map.size(); ++i)
    map[i] = sub_index_map->local_to_global(i);
  
  // Create PETSc local-to-global map/index set
  ierr = ISLocalToGlobalMappingCreate(_global_vector.mpi_comm(), 1, map.size(), map.data(),
                                      PETSC_COPY_VALUES, &petsc_local_to_global);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");

  // Set matrix local-to-global maps
  ierr = VecSetLocalToGlobalMapping(_x, petsc_local_to_global);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetLocalToGlobalMapping");

  // Clean up local-to-global maps
  ierr = ISLocalToGlobalMappingDestroy(&petsc_local_to_global);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");
  // --- end --- from PETScVector::_init and PETScMatrix::init --- end --- //
  
  // Reset DOLFIN storage
  this->PETScVector::operator=(PETScVector(_x));
}
//-----------------------------------------------------------------------------
BlockPETScSubVector::~BlockPETScSubVector()
{
  PetscErrorCode ierr;
  
  // Communicate ghosted values
  ierr = VecAssemblyBegin(_x);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecAssemblyBegin");
  ierr = VecAssemblyEnd(_x);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecAssemblyEnd");
  
  // We cannot use
  //VecRestoreSubVector(_global_vector.vec(), _is, &_x);
  // because of the hardcoded INSERT_VALUES mode
  // --- from VecRestoreSubVector --- //
  ierr = VecScatterBegin(_scatter, _x, _global_vector.vec(), _insert_mode, SCATTER_REVERSE);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterBegin");
  ierr = VecScatterEnd(_scatter, _x, _global_vector.vec(), _insert_mode, SCATTER_REVERSE);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterEnd");
  // Also destroy scatter
  ierr = VecScatterDestroy(&_scatter);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecScatterDestroy");
  // Also destroy IS
  ierr = ISDestroy(&_is);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISDestroy");
  // Finally, destroy vector container
  ierr = VecDestroy(&_x);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecDestroy");
  // --- end --- from VecRestoreSubVector --- end --- //
}
//-----------------------------------------------------------------------------
std::int64_t BlockPETScSubVector::size() const
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "access size of PETSc sub vector",
                     "This method was supposedly never used by the sub vector interface");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::set(PetscScalar a)
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "set local sub vector values",
                     "This method is not available because there is no guarantee that VecSet is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::shift(PetscScalar a)
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "shift local sub vector values",
                     "This method is not available because there is no guarantee that VecShift is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::scale(PetscScalar a)
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "scale local sub vector values",
                     "This method is not available because there is no guarantee that VecScale is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::mult(const BlockPETScSubVector& x)
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "multiply by other sub vector",
                     "This method is not available because there is no guarantee that VecPointwiseMult is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::apply()
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "apply local sub vector values",
                     "This method is not available because there is no need to apply sub vector changes, as they are applied on destruction");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::apply_ghosts()
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "apply local sub vector ghost values",
                     "This method is not available because there is no need to apply sub vector changes, as they are applied on destruction");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::update_ghosts()
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "updated local sub vector ghost values",
                     "This method is not available because there is no need to apply sub vector changes, as they are applied on destruction");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::get_local(std::vector<PetscScalar>& values) const
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "get local sub vector values",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::set_local(const std::vector<PetscScalar>& values)
{
  if (_insert_mode != INSERT_VALUES)
    multiphenics_error("BlockPETScSubVector.cpp",
                       "set local sub vector values",
                       "This method is available only when INSERT_VALUES is chosen as mode in the constructor");
                
  const auto _local_range = local_range();
  const std::int64_t local_size = _local_range[1] - _local_range[0];
  std::vector<PetscScalar> restricted_values(local_size);
  for (auto & it: _original_to_sub_block)
  {
    auto original_index = it.first;
    auto sub_block_local_index = it.second;
    if (sub_block_local_index < local_size)
      restricted_values[sub_block_local_index] = values[original_index];
  }
  PETScVector::set_local(restricted_values);
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::add_local(const std::vector<PetscScalar>& values)
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "add local sub vector values",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::get_local(PetscScalar* block, std::size_t m,
                                    const PetscInt* rows) const 
{
  // The sub vector is not ghosted because using VecMPISetGhost cannot be used in the constructor
  // as it clears out the storage. For this reason, in order to allow getting unowned rows,
  // we query the global vector (which is ghosted), rather then the sub vector.
  
  if (_insert_mode != INSERT_VALUES)
    multiphenics_error("BlockPETScSubVector.cpp",
                       "get local sub vector values",
                       "This method is available only when INSERT_VALUES is chosen as mode in the constructor");
  
  std::vector<PetscInt> rows_vec(rows, rows + m);
  std::vector<PetscInt> restricted_rows_vec__global_vector;
  std::vector<bool> is_in_restriction;
  to_restricted_vector_indices(rows_vec, restricted_rows_vec__global_vector, &is_in_restriction);
  
  std::vector<PetscScalar> restricted_vals_vec(restricted_rows_vec__global_vector.size());
  _global_vector.get_local(restricted_vals_vec.data(), restricted_rows_vec__global_vector.size(), restricted_rows_vec__global_vector.data());

  std::vector<PetscScalar> vals_vec(block, block + m);
  from_restricted_subvector_values(vals_vec, restricted_vals_vec, is_in_restriction);
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::set(const PetscScalar* block, std::size_t m,
                              const PetscInt* rows)
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "set (possibly non local) sub vector values",
                     "This method is not available because there is no guarantee that VecSetValues is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::set_local(const PetscScalar* block, std::size_t m,
                                    const PetscInt* rows)
{
  if (_insert_mode != INSERT_VALUES)
    multiphenics_error("BlockPETScSubVector.cpp",
                       "set local sub vector values",
                       "This method is available only when INSERT_VALUES is chosen as mode in the constructor");
                
  std::vector<PetscInt> rows_vec(rows, rows + m);
  std::vector<PetscScalar> vals_vec(block, block + m);
  std::vector<PetscInt> restricted_rows_vec;
  std::vector<PetscScalar> restricted_vals_vec;
  to_restricted_subvector_indices_and_values(rows_vec, restricted_rows_vec, vals_vec, restricted_vals_vec);
  PETScVector::set_local(restricted_vals_vec.data(), restricted_rows_vec.size(), restricted_rows_vec.data());
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::add(const PetscScalar* block, std::size_t m,
                              const PetscInt* rows)
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "add (possibly non local) sub vector values",
                     "This method is not available because there is no guarantee that VecSetValues is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::add_local(const PetscScalar* block, std::size_t m,
                                    const PetscInt* rows)
{
  if (_insert_mode != ADD_VALUES)
    multiphenics_error("BlockPETScSubVector.cpp",
                       "add local sub vector values",
                       "This method is available only when ADD_VALUES is chosen as mode in the constructor");
                
  std::vector<PetscInt> rows_vec(rows, rows + m);
  std::vector<PetscScalar> vals_vec(block, block + m);
  std::vector<PetscInt> restricted_rows_vec;
  std::vector<PetscScalar> restricted_vals_vec;
  to_restricted_subvector_indices_and_values(rows_vec, restricted_rows_vec, vals_vec, restricted_vals_vec);
  PETScVector::add_local(restricted_vals_vec.data(), restricted_rows_vec.size(), restricted_rows_vec.data());
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::gather(BlockPETScSubVector& y, const std::vector<PetscInt>& indices) const
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "gather sub vector values",
                     "This method was supposedly never used by block interface, and its implementation requires some more work");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::axpy(double a, const BlockPETScSubVector& y)
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "add multiple of a given sub vector",
                     "This method is not available because there is no guarantee that VecAXPY is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::abs()
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "replace all entries in the sub vector by their absolute values",
                     "This method is not available because there is no guarantee that VecAbs is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
PetscScalar BlockPETScSubVector::dot(const BlockPETScSubVector& y) const
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "compute inner product with given sub vector",
                     "This method is not available because there is no guarantee that VecDot is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
PetscReal BlockPETScSubVector::norm(Norm norm_type) const
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "compute norm of sub vector",
                     "This method is not available because there is no guarantee that VecNorm is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
PetscReal BlockPETScSubVector::normalize()
  {
  multiphenics_error("BlockPETScSubVector.cpp",
                     "normalize sub vector",
                     "This method is not available because there is no guarantee that VecNormalize is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
std::pair<double, PetscInt> BlockPETScSubVector::min() const
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "compute minimum value of sub vector",
                     "This method is not available because there is no guarantee that VecMin is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
std::pair<double, PetscInt> BlockPETScSubVector::max() const
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "compute maximum value of sub vector",
                     "This method is not available because there is no guarantee that VecMax is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
PetscScalar BlockPETScSubVector::sum() const
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "compute sum of values of sub vector",
                     "This method is not available because there is no guarantee that VecSum is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::set_options_prefix(std::string options_prefix)
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "set options prefix",
                     "This method is not available because there is no guarantee that VecSetOptionsPrefix is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
std::string BlockPETScSubVector::get_options_prefix() const
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "get options prefix",
                     "This method is not available because there is no guarantee that VecGetOptionsPrefix is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::set_from_options()
{
  multiphenics_error("BlockPETScSubVector.cpp",
                     "set from options",
                     "This method is not available because there is no guarantee that VecSetFromOptions is implemented by PETSc SubVector");
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::to_restricted_subvector_indices(
  const std::vector<PetscInt> & block_unrestricted_subvector_indices, std::vector<PetscInt> & block_restricted_subvector_indices,
  std::vector<bool> * is_in_restriction
) const
{
  assert(block_restricted_subvector_indices.size() == 0);
  if (is_in_restriction != NULL)
      assert(is_in_restriction->size() == 0);
  
  for (auto block_unrestricted_subvector_index : block_unrestricted_subvector_indices)
    if (_original_to_sub_block.count(block_unrestricted_subvector_index) > 0) 
    {
      block_restricted_subvector_indices.push_back(
        _original_to_sub_block.at(block_unrestricted_subvector_index)
      );
      if (is_in_restriction != NULL)
        is_in_restriction->push_back(true);
    }
    else
    {
      if (is_in_restriction != NULL)
        is_in_restriction->push_back(false);
    }
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::to_restricted_vector_indices(
  const std::vector<PetscInt> & block_unrestricted_subvector_indices, std::vector<PetscInt> & block_restricted_vector_indices,
  std::vector<bool> * is_in_restriction
) const
{
  assert(block_restricted_vector_indices.size() == 0);
  if (is_in_restriction != NULL)
      assert(is_in_restriction->size() == 0);
  
  for (auto block_unrestricted_subvector_index : block_unrestricted_subvector_indices)
    if (_original_to_block.count(block_unrestricted_subvector_index) > 0) 
    {
      block_restricted_vector_indices.push_back(
        _original_to_block.at(block_unrestricted_subvector_index)
      );
      if (is_in_restriction != NULL)
        is_in_restriction->push_back(true);
    }
    else
    {
      if (is_in_restriction != NULL)
        is_in_restriction->push_back(false);
    }
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::to_restricted_subvector_indices_and_values(
  const std::vector<PetscInt> & block_unrestricted_subvector_indices, std::vector<PetscInt> & block_restricted_subvector_indices,
  const std::vector<PetscScalar> & block_unrestricted_subvector_values, std::vector<PetscScalar> & block_restricted_subvector_values
) const
{
  assert(block_unrestricted_subvector_indices.size() == block_unrestricted_subvector_values.size());
  assert(block_restricted_subvector_indices.size() == 0);
  assert(block_restricted_subvector_values.size() == 0);
  
  // Extract indices
  std::vector<bool> is_in_restriction;
  to_restricted_subvector_indices(block_unrestricted_subvector_indices, block_restricted_subvector_indices, &is_in_restriction);
  
  // Resize
  block_restricted_subvector_values.resize(block_restricted_subvector_indices.size());
  
  // Extract values
  std::size_t restricted_value_iterator(0);
  for (std::size_t unrestricted_value_iterator(0); unrestricted_value_iterator < block_unrestricted_subvector_values.size(); ++unrestricted_value_iterator)
  {
    if (is_in_restriction[unrestricted_value_iterator])
    {
      block_restricted_subvector_values[restricted_value_iterator] = block_unrestricted_subvector_values[unrestricted_value_iterator];
      restricted_value_iterator++;
    }
  }
  assert(restricted_value_iterator == block_restricted_subvector_values.size());
}
//-----------------------------------------------------------------------------
void BlockPETScSubVector::from_restricted_subvector_values(
  std::vector<PetscScalar> & block_unrestricted_subvector_values, const std::vector<PetscScalar> & block_restricted_subvector_values,
  const std::vector<bool> & is_in_restriction
) const
{
  assert(block_unrestricted_subvector_values.size() == is_in_restriction.size());
  
  // Set values
  std::size_t restricted_value_iterator(0);
  for (std::size_t unrestricted_value_iterator(0); unrestricted_value_iterator < block_unrestricted_subvector_values.size(); ++unrestricted_value_iterator)
  {
    if (is_in_restriction[unrestricted_value_iterator])
    {
      block_unrestricted_subvector_values[unrestricted_value_iterator] = block_restricted_subvector_values[restricted_value_iterator];
      ++restricted_value_iterator;
    }
  }
  assert(restricted_value_iterator == block_restricted_subvector_values.size());
}

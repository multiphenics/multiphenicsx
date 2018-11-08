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

#include <dolfin/common/Timer.h>
#include <dolfin/fem/assemble_matrix_impl.h>
#include <dolfin/fem/assemble_vector_impl.h>
#include <dolfin/fem/SparsityPatternBuilder.h>
#include <multiphenics/fem/block_assemble.h>
#include <multiphenics/fem/BlockDofMap.h>
#include <multiphenics/la/BlockInsertMode.h>
#include <multiphenics/la/BlockPETScSubMatrix.h>
#include <multiphenics/la/BlockPETScSubVector.h>

using namespace dolfin;
using namespace dolfin::fem;
using namespace multiphenics;
using namespace multiphenics::fem;

using dolfin::common::IndexMap;
using dolfin::common::Timer;
using dolfin::la::SparsityPattern;
using dolfin::la::PETScMatrix;
using dolfin::la::PETScVector;
using dolfin::mesh::Mesh;
using multiphenics::la::BlockInsertMode;
using multiphenics::la::BlockPETScSubMatrix;
using multiphenics::la::BlockPETScSubVector;

//-----------------------------------------------------------------------------
std::shared_ptr<PETScVector> multiphenics::fem::block_assemble(const BlockForm1& L)
{
  std::shared_ptr<PETScVector> b = init_vector(L);
  block_assemble(*b, L);
  return b;
}
//-----------------------------------------------------------------------------
void multiphenics::fem::block_assemble(PETScVector& b, const BlockForm1& L)
{
  // This method is adapted from
  //    dolfin::fem::_assemble_vector in dolfin/fem/assembler.cpp
  //    dolfin::fem::assemble_ghosted in dolfin/fem/assemble_vector_impl.cpp

  // Get ghosted form
  Vec b_ghosted;
  VecGhostGetLocalForm(b.vec(), &b_ghosted);
  // Get array corresponding to ghosted form
  PetscInt b_ghosted_size = 0;
  VecGetSize(b_ghosted, &b_ghosted_size);
  PetscScalar* b_ghosted_array;
  VecGetArray(b_ghosted, &b_ghosted_array);
  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b_ghosted_eigen(b_ghosted_array, b_ghosted_size);
  // Assemble using standard assembler
  for (unsigned int i(0); i < L.block_size(0); ++i)
  {
    // Note that we cannot call either
    // * dolfin::fem::assemble_ghosted because PETSc sub vectors (i) are not ghosted, and (ii) trying to call VecGetSize on them
    //   results in segmentation fault, nor
    // * dolfin::fem::assemble_local because that function calls native VecGetArray/VecRestoreArray rather than the PETScVector interface,
    //   which we have patched to handle the restriction of subvectors.
    // We thus call directly dolfin::fem::assemble_eigen on a temporarily allocated Eigen::Array. Note that this might be less efficient
    // than the implementation in dolfin::fem::assemble_local, which reuses PETSc storage. Unfortunately, vector sizes (due to restriction)
    // may not match.
    const auto index_map_i = L.block_function_spaces()[0]->operator[](i)->dofmap()->index_map();
    std::size_t unrestricted_ghosted_size = index_map_i->block_size()*(index_map_i->size_local() + index_map_i->num_ghosts());
    std::vector<PetscScalar> unrestricted_ghosted_values(unrestricted_ghosted_size);
    Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> unrestricted_ghosted_values_eigen(unrestricted_ghosted_values.data(), unrestricted_ghosted_values.size());
    assemble_eigen(unrestricted_ghosted_values_eigen, L(i), {}, {});
    // Exploit a temporary BlockPETScSubVector to handle restrictions
    std::shared_ptr<BlockPETScSubVector> b_i = std::make_shared<BlockPETScSubVector>(b, i, L.block_function_spaces()[0]->block_dofmap(), BlockInsertMode::ADD_VALUES);
    std::vector<PetscInt> unrestricted_ghosted_rows(unrestricted_ghosted_size);
    std::iota(unrestricted_ghosted_rows.begin(), unrestricted_ghosted_rows.end(), 0);
    std::vector<PetscInt> restricted_ghosted_rows__global_vector;
    std::vector<bool> is_in_restriction;
    b_i->to_restricted_vector_indices(unrestricted_ghosted_rows, restricted_ghosted_rows__global_vector, &is_in_restriction);
    for (std::size_t k_value = 0, k_global_row = 0; k_value < unrestricted_ghosted_size; ++k_value)
    {
      if (is_in_restriction[k_value])
      {
        b_ghosted_eigen[restricted_ghosted_rows__global_vector[k_global_row]] = unrestricted_ghosted_values[k_value];
        k_global_row++;
      }
    }
  }
  // Restore ghosted array
  VecRestoreArray(b_ghosted, &b_ghosted_array);
  // Restore ghosted form
  VecGhostRestoreLocalForm(b.vec(), &b_ghosted);
  // Finalize assembly of global tensor
  b.apply_ghosts();
}
//-----------------------------------------------------------------------------
std::shared_ptr<PETScMatrix> multiphenics::fem::block_assemble(const BlockForm2& a)
{
  std::shared_ptr<PETScMatrix> A = init_matrix(a);
  block_assemble(*A, a);
  return A;
}
//-----------------------------------------------------------------------------
void multiphenics::fem::block_assemble(PETScMatrix& A, const BlockForm2& a)
{
  // This method is adapted from
  //    dolfin::fem::assemble (variant with PETScMatrix as first input) in dolfin/fem/assembler.cpp
  
  // Assemble using standard assembler
  for (unsigned int i(0); i < a.block_size(0); ++i)
  {
    for (unsigned int j(0); j < a.block_size(1); ++j)
    {
      std::shared_ptr<PETScMatrix> A_ij = std::make_shared<BlockPETScSubMatrix>(A, i, j, a.block_function_spaces()[0]->block_dofmap(), a.block_function_spaces()[1]->block_dofmap(), BlockInsertMode::ADD_VALUES);
      assemble_matrix(*A_ij, a(i, j), {}, {});
    }
  }

  // Finalize assembly of global tensor
  A.apply(PETScMatrix::AssemblyType::FINAL);
}
//-----------------------------------------------------------------------------
std::shared_ptr<PETScVector> multiphenics::fem::init_vector(const BlockForm1& L)
{
  // This method is adapted from
  //    dolfin::fem::init_global_tensor (PETScVector case) in dolfin/fem/AssemblerBase.cpp
  assert(L.block_function_spaces()[0]->block_dofmap()->index_map());
  return std::make_shared<PETScVector>(*L.block_function_spaces()[0]->block_dofmap()->index_map());
}
//-----------------------------------------------------------------------------
std::shared_ptr<PETScMatrix> multiphenics::fem::init_matrix(const BlockForm2& a)
{
  // This method is adapted from
  //    dolfin::fem::init_matrix in dolfin/fem/utils.cpp
  bool keep_diagonal = true; // TODO why is this not an input argument in dolfinx?
  
  // Get dof maps
  std::array<const GenericDofMap*, 2> dofmaps
      = {{a.block_function_spaces()[0]->block_dofmap().get(),
          a.block_function_spaces()[1]->block_dofmap().get()}};

  // Get mesh
  assert(a.mesh());
  const Mesh& mesh = *(a.mesh());

  Timer t0("Build sparsity");

  // Get IndexMaps for each dimension
  std::array<std::shared_ptr<const IndexMap>, 2> index_maps
      = {{dofmaps[0]->index_map(), dofmaps[1]->index_map()}};
      
  // Check integral types in block form. Note that this might create
  // an overly conservative sparsity pattern.
  bool has_cell_integrals(false);
  bool has_interior_facet_integrals(false);
  bool has_exterior_facet_integrals(false);
  bool has_vertex_integrals(false);
  for (unsigned int i(0); i < a.block_size(0); ++i)
  {
    for (unsigned int j(0); j < a.block_size(1); ++j)
    {
      const Form & a_ij(a(i, j));
      if (a_ij.integrals().num_cell_integrals() > 0)
        has_cell_integrals = true;
      if (a_ij.integrals().num_interior_facet_integrals() > 0)
        has_interior_facet_integrals = true;
      if (a_ij.integrals().num_exterior_facet_integrals() > 0)
        has_exterior_facet_integrals = true;
      if (a_ij.integrals().num_vertex_integrals() > 0)
        has_vertex_integrals = true;
    }
  }
  
  // Create and build sparsity pattern
  SparsityPattern pattern = SparsityPatternBuilder::build(
      mesh.mpi_comm(), mesh, dofmaps,
      has_cell_integrals,
      has_interior_facet_integrals,
      has_exterior_facet_integrals,
      has_vertex_integrals,
      keep_diagonal);
  t0.stop();

  // Initialize matrix
  Timer t1("Init tensor");
  std::shared_ptr<PETScMatrix> A = std::make_shared<PETScMatrix>(a.mesh()->mpi_comm(), pattern);
  t1.stop();

  // Insert zeros to dense rows in increasing order of column index
  // to avoid CSR data reallocation when assembling in random order
  // resulting in quadratic complexity; this has to be done before
  // inserting to diagonal below

  // Tabulate indices of dense rows
  Eigen::Array<std::size_t, Eigen::Dynamic, 1> global_dofs
      = dofmaps[0]->tabulate_global_dofs();
  if (global_dofs.size() > 0)
  {
    // Get local row range
    const IndexMap& index_map_0 = *dofmaps[0]->index_map();
    const auto row_range = A->local_range(0);

    assert(index_map_0.block_size() == 1);

    // Set zeros in dense rows in order of increasing column index
    const PetscScalar block = 0.0;
    PetscInt IJ[2];
    for (Eigen::Index i = 0; i < global_dofs.size(); ++i)
    {
      const std::int64_t I = index_map_0.local_to_global(global_dofs[i]);
      if (I >= row_range[0] && I < row_range[1])
      {
        IJ[0] = I;
        for (std::int64_t J = 0; J < A->size()[1]; J++)
        {
          IJ[1] = J;
          A->set(&block, 1, &IJ[0], 1, &IJ[1]);
        }
      }
    }

    // Eventually wait with assembly flush for keep_diagonal
    if (!keep_diagonal)
      A->apply(PETScMatrix::AssemblyType::FLUSH);
  }

  // Insert zeros on the diagonal as diagonal entries may be
  // optimised away, e.g. when calling PETScMatrix::apply.
  if (keep_diagonal)
  {
    // Loop over rows and insert 0.0 on the diagonal
    const PetscScalar block = 0.0;
    const auto row_range = A->local_range(0);
    const std::int64_t range = std::min(row_range[1], A->size()[1]);

    for (std::int64_t i = row_range[0]; i < range; i++)
    {
      const PetscInt _i = i;
      A->set(&block, 1, &_i, 1, &_i);
    }

    A->apply(PETScMatrix::AssemblyType::FLUSH);
  }
  
  // Return
  return A;
}

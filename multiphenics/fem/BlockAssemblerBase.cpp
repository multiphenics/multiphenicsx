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

#include <dolfin/common/Timer.h>
#include <dolfin/fem/SparsityPatternBuilder.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/TensorLayout.h>
#include <dolfin/la/SparsityPattern.h>
#include <multiphenics/fem/BlockDofMap.h>
#include <multiphenics/fem/BlockAssemblerBase.h>
#include <multiphenics/log/log.h>

using namespace dolfin;
using namespace dolfin::fem;
using namespace multiphenics;
using namespace multiphenics::fem;

//-----------------------------------------------------------------------------
BlockAssemblerBase::BlockAssemblerBase() : 
  add_values(false), finalize_tensor(true),
  keep_diagonal(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BlockAssemblerBase::init_global_tensor(GenericTensor& A, const BlockFormBase& a)
{
  // This method is adapted from
  //    AssemblerBase::init_global_tensor
  
  // Get mesh
  dolfin_assert(a.mesh());
  const Mesh& mesh = *(a.mesh());
  
  if (A.empty())
  {
    Timer t0("Build sparsity");

    // Create layout for initialising tensor
    std::shared_ptr<TensorLayout> tensor_layout;
    tensor_layout = create_layout(mesh.mpi_comm(), a.rank()); // TODO complete rewrite
    dolfin_assert(tensor_layout);
    
    // Get dimensions and mapping across processes for each dimension
    std::vector<std::shared_ptr<const IndexMap>> index_maps;
    for (std::size_t i = 0; i < a.rank(); i++)
    {
      index_maps.push_back(a.block_function_spaces()[i]->block_dofmap()->index_map());
    }

    // Initialise tensor layout
    tensor_layout->init(index_maps, TensorLayout::Ghosts::UNGHOSTED);

    // Build sparsity pattern if required
    if (tensor_layout->sparsity_pattern())
    {
      SparsityPattern& pattern = *tensor_layout->sparsity_pattern();
      
      // Initialize sparsity pattern
      pattern.init(index_maps);
      
      // Build sparsity pattern for each block
      const BlockForm2& a_form2 = dynamic_cast<const BlockForm2&>(a);
      for (std::size_t i = 0; i < a_form2.block_size(0); i++)
      {
        for (std::size_t j = 0; j < a_form2.block_size(1); j++)
        {
          const Form& a_ij = a_form2(i, j);
          if (a_ij.ufc_form())
          {
            std::vector<const GenericDofMap*> dofmaps{
              &a.block_function_spaces()[0]->block_dofmap()->view(i),
              &a.block_function_spaces()[1]->block_dofmap()->view(j)
            };
            SparsityPatternBuilder::build(pattern,
                                          mesh, dofmaps,
                                          a_ij.ufc_form()->has_cell_integrals(),
                                          a_ij.ufc_form()->has_interior_facet_integrals(),
                                          a_ij.ufc_form()->has_exterior_facet_integrals(),
                                          a_ij.ufc_form()->has_vertex_integrals(),
                                          keep_diagonal,
                                          /* initialize = */ false,
                                          /* finalize = */ false);
          }
        }
      }
      
      // Finalize sparsity pattern
      pattern.apply();
    }
    t0.stop();

    // Initialize tensor
    Timer t1("Init tensor");
    A.init(*tensor_layout);
    t1.stop();

    // Insert zeros on the diagonal as diagonal entries may be
    // prematurely optimised away by the linear algebra backend when
    // calling GenericMatrix::apply, e.g. PETSc does this then errors
    // when matrices have no diagonal entry inserted.
    if (A.rank() == 2 && keep_diagonal)
    {
      // Down cast to GenericMatrix
      GenericMatrix& _matA = as_type<GenericMatrix>(A);

      // Loop over rows and insert 0.0 on the diagonal
      const double block = 0.0;
      const std::pair<std::size_t, std::size_t> row_range = A.local_range(0);
      const std::size_t range = std::min(row_range.second, A.size(1));
      for (std::size_t i = row_range.first; i < range; i++)
      {
        dolfin::la_index _i = i;
        _matA.set(&block, 1, &_i, 1, &_i);
      }
      A.apply("flush");
    }

    // Delete sparsity pattern
    Timer t2("Delete sparsity");
    t2.stop();
  }
  else
  {
    // If tensor is not reset, check that dimensions are correct
    for (std::size_t i = 0; i < a.rank(); ++i)
    {
      if (A.size(i) != a.block_function_spaces()[i]->block_dofmap()->global_dimension())
      {
        multiphenics_error("BlockAssemblerBase.cpp",
                           "assemble form",
                           "Dim " + std::to_string(i) + " of tensor does not match form");
      }
    }
  }

  if (!add_values)
    A.zero();
}

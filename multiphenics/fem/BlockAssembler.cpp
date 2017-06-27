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

#include <memory>
#include <dolfin/fem/Assembler.h>
#include <dolfin/fem/UFC.h>
#include <dolfin/la/GenericLinearAlgebraFactory.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/GenericVector.h>
#include <multiphenics/fem/BlockForm1.h>
#include <multiphenics/fem/BlockForm2.h>
#include <multiphenics/fem/BlockAssembler.h>
#include <multiphenics/la/GenericBlockLinearAlgebraFactory.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BlockAssembler::BlockAssembler()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BlockAssembler::assemble(GenericTensor& A, const BlockFormBase& a)
{
  // This method is adapted from
  //    Assembler::assemble

  // Initialize global tensor
  init_global_tensor(A, a);
  
  // Get the block linear algebra factory to extract sub-blocks
  GenericBlockLinearAlgebraFactory& block_linear_algebra_factory = dynamic_cast<GenericBlockLinearAlgebraFactory&>(A.factory());
  
  // Assemble using standard assembler
  Assembler assembler;
  assembler.add_values = add_values;
  assembler.finalize_tensor = false;
  
  dolfin_assert(a.rank() == 2 || a.rank() == 1)
  if (a.rank() == 2)
  {
    GenericMatrix& A_mat = dynamic_cast<GenericMatrix&>(A);
    const BlockForm2& a_form2 = dynamic_cast<const BlockForm2&>(a);
    for (unsigned int i(0); i < a_form2.block_size(0); ++i)
      for (unsigned int j(0); j < a_form2.block_size(1); ++j)
      {
        if (i == j)
          assembler.keep_diagonal = keep_diagonal;
        else
          assembler.keep_diagonal = false;
        std::shared_ptr<GenericMatrix> A_ij = block_linear_algebra_factory.create_sub_matrix(A_mat, i, j, BlockInsertMode::ADD_VALUES);
        const Form& a_ij = a_form2(i, j);
        this->sub_assemble(*A_ij, a_ij, assembler);
      }
  }
  else if (a.rank() == 1)
  {
    GenericVector& A_vec = dynamic_cast<GenericVector&>(A);
    const BlockForm1& a_form1 = dynamic_cast<const BlockForm1&>(a);
    for (unsigned int i(0); i < a_form1.block_size(0); ++i)
    {
      std::shared_ptr<GenericVector> A_i = block_linear_algebra_factory.create_sub_vector(A_vec, i, BlockInsertMode::ADD_VALUES);
      const Form& a_i = a_form1(i);
      this->sub_assemble(*A_i, a_i, assembler);
    }
  }

  // Finalize assembly of global tensor
  if (finalize_tensor)
    A.apply("add");
}

void BlockAssembler::sub_assemble(GenericTensor& A, const Form& a, Assembler& assembler)
{
  // This method is adapted from
  //    Assembler::assemble
  // removing calls to initialization and finalization of the global tensor
  
  // Get cell domains
  std::shared_ptr<const MeshFunction<std::size_t>>
    cell_domains = a.cell_domains();

  // Get exterior facet domains
  std::shared_ptr<const MeshFunction<std::size_t>> exterior_facet_domains
      = a.exterior_facet_domains();

  // Get interior facet domains
  std::shared_ptr<const MeshFunction<std::size_t>> interior_facet_domains
      = a.interior_facet_domains();

  // Get vertex domains
  std::shared_ptr<const MeshFunction<std::size_t>> vertex_domains
    = a.vertex_domains();

  // Check form
  //AssemblerBase::check(a); // commented out because "check" is protected

  // Create data structure for local assembly data
  UFC ufc(a);

  // Update off-process coefficients
  const std::vector<std::shared_ptr<const GenericFunction>>
    coefficients = a.coefficients();

  // Initialize global tensor
  // ~> already done!

  // Assemble over cells
  assembler.assemble_cells(A, a, ufc, cell_domains, NULL);

  // Assemble over exterior facets
  assembler.assemble_exterior_facets(A, a, ufc, exterior_facet_domains, NULL);

  // Assemble over interior facets
  assembler.assemble_interior_facets(A, a, ufc, interior_facet_domains,
                                     cell_domains, NULL);

  // Assemble over vertices
  assembler.assemble_vertices(A, a, ufc, vertex_domains);

  // Finalize assembly of global tensor
  // ~> done only once, at the end!
}

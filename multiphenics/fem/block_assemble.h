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

#ifndef __BLOCK_ASSEMBLER_H
#define __BLOCK_ASSEMBLER_H

#include <multiphenics/fem/BlockForm1.h>
#include <multiphenics/fem/BlockForm2.h>

namespace multiphenics
{
  namespace fem
  {
    /// Assemble block vector from given block form of rank 1
    std::shared_ptr<dolfin::la::PETScVector> block_assemble(const BlockForm1& L);
    
    /// Assemble block vector from given block form of rank 1, re-using existing vector
    void block_assemble(dolfin::la::PETScVector& b, const BlockForm1& L);
    
    /// Assemble block vector from given block form of rank 2
    std::shared_ptr<dolfin::la::PETScMatrix> block_assemble(const BlockForm2& a);
    
    /// Assemble block vector from given block form of rank 2, re-using existing matrix
    void block_assemble(dolfin::la::PETScMatrix& A, const BlockForm2& a);
    
    // Initialize block vector
    std::shared_ptr<dolfin::la::PETScVector> init_vector(const BlockForm1& L);
    
    // Initialize block matrix
    std::shared_ptr<dolfin::la::PETScMatrix> init_matrix(const BlockForm2& a);
  }
}

#endif

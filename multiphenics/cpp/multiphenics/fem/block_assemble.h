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

#ifndef __BLOCK_ASSEMBLE_H
#define __BLOCK_ASSEMBLE_H

#include <petscmat.h>
#include <petscvec.h>
#include <multiphenics/fem/BlockForm1.h>
#include <multiphenics/fem/BlockForm2.h>

namespace multiphenics
{
  namespace fem
  {
    /// Assemble block vector from given block form of rank 1
    Vec block_assemble(const BlockForm1& L);

    /// Assemble block vector from given block form of rank 1, re-using existing vector
    void block_assemble(Vec b, const BlockForm1& L);

    /// Assemble block vector from given block form of rank 2
    Mat block_assemble(const BlockForm2& a);

    /// Assemble block vector from given block form of rank 2, re-using existing matrix
    void block_assemble(Mat A, const BlockForm2& a);

    // Initialize block vector
    Vec init_vector(const BlockForm1& L);

    // Initialize block matrix
    Mat init_matrix(const BlockForm2& a);
  }
}

#endif

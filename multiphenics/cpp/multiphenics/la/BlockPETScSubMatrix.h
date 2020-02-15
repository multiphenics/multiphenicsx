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

#ifndef __BLOCK_PETSC_SUB_MATRIX_H
#define __BLOCK_PETSC_SUB_MATRIX_H

#include <petscmat.h>
#include <multiphenics/fem/BlockDofMap.h>

namespace multiphenics
{
  namespace la
  {
    /// This class initializes a PETSc sub matrix associated to a specific block pair.
    class BlockPETScSubMatrix
    {
    public:
      /// Constructor
      BlockPETScSubMatrix(Mat A,
                          std::array<std::size_t, 2> block_indices,
                          std::array<std::shared_ptr<const multiphenics::fem::BlockDofMap>, 2> block_dofmaps);

      /// Destructor
      ~BlockPETScSubMatrix();

      /// Copy constructor (deleted)
      BlockPETScSubMatrix(const BlockPETScSubMatrix& A) = delete;

      /// Move constructor (deleted)
      BlockPETScSubMatrix(BlockPETScSubMatrix&& A) = delete;

      /// Assignment operator (deleted)
      BlockPETScSubMatrix& operator=(const BlockPETScSubMatrix& A) = delete;

      /// Move assignment operator (deleted)
      BlockPETScSubMatrix& operator=(BlockPETScSubMatrix&& A) = delete;

      /// Pointer to submatrix
      Mat mat() const;

    private:
      Mat _global_matrix;
      std::array<IS, 2> _is;
      Mat _A;
    };
  }
}

#endif

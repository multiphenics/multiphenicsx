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

#ifndef __BLOCK_PETSC_SUB_VECTOR_READ_WRAPPER_H
#define __BLOCK_PETSC_SUB_VECTOR_READ_WRAPPER_H

#include <petscvec.h>
#include <multiphenics/fem/BlockDofMap.h>

namespace multiphenics
{
  namespace la
  {

    /// This class initializes an eigen wrapper to a PETSc sub vector associated to a specific block.
    class BlockPETScSubVectorReadWrapper
    {
    public:
      /// Constructor
      BlockPETScSubVectorReadWrapper(Vec x,
                                     std::size_t block_index,
                                     std::shared_ptr<const multiphenics::fem::BlockDofMap> block_dofmap);

      /// Destructor
      ~BlockPETScSubVectorReadWrapper();

      /// Copy constructor (deleted)
      BlockPETScSubVectorReadWrapper(const BlockPETScSubVectorReadWrapper&) = delete;

      /// Move constructor (deleted)
      BlockPETScSubVectorReadWrapper(BlockPETScSubVectorReadWrapper&&) = delete;

      // Assignment operator (deleted)
      BlockPETScSubVectorReadWrapper& operator=(const BlockPETScSubVectorReadWrapper&) = delete;

      /// Move assignment operator (deleted)
      BlockPETScSubVectorReadWrapper& operator=(BlockPETScSubVectorReadWrapper&&) = delete;

      /// Public attribute to store content to be added/inserted in an eigen matrix
      Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> content;

    private:
      std::vector<PetscScalar> _content;
    };
  }
}

#endif

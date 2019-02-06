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

#ifndef __BLOCK_PETSC_SUB_VECTOR_WRAPPER_H
#define __BLOCK_PETSC_SUB_VECTOR_WRAPPER_H

#include <multiphenics/la/BlockPETScSubVectorReadWrapper.h>

namespace multiphenics
{
  namespace la
  {
    
    /// This class initializes an eigen wrapper to a PETSc sub vector associated to a specific block.
    class BlockPETScSubVectorWrapper: public BlockPETScSubVectorReadWrapper
    {
    public:
      /// Constructor
      BlockPETScSubVectorWrapper(Vec x,
                                 std::size_t block_index,
                                 std::shared_ptr<const multiphenics::fem::BlockDofMap> block_dofmap,
                                 InsertMode insert_mode);
      
      /// Destructor
      ~BlockPETScSubVectorWrapper();
      
      /// Copy constructor (deleted)
      BlockPETScSubVectorWrapper(const BlockPETScSubVectorWrapper&) = delete;

      /// Move constructor (deleted)
      BlockPETScSubVectorWrapper(BlockPETScSubVectorWrapper&&) = delete;
      
      // Assignment operator (deleted)
      BlockPETScSubVectorWrapper& operator=(const BlockPETScSubVectorWrapper&) = delete;

      /// Move assignment operator (deleted)
      BlockPETScSubVectorWrapper& operator=(BlockPETScSubVectorWrapper&&) = delete;
      
    private:
      Vec _global_vector;
      const std::map<PetscInt, PetscInt> & _original_to_block;
      const std::map<PetscInt, PetscInt> & _sub_block_to_original;
      InsertMode _insert_mode;
    };
  }
}

#endif

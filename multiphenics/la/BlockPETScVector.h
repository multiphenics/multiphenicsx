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

#ifndef __BLOCK_PETSC_VECTOR_H
#define __BLOCK_PETSC_VECTOR_H

#include <dolfin/la/PETScVector.h>
#include <multiphenics/fem/BlockDofMap.h>

namespace multiphenics
{
  namespace la
  {
    class BlockPETScSubVector;
    
    class BlockPETScVector : public dolfin::PETScVector
    {
    public:
      /// Create empty vector (on MPI_COMM_WORLD)
      BlockPETScVector();

      /// Create vector
      BlockPETScVector(const dolfin::common::IndexMap& map);

      /// Create vector
      BlockPETScVector(MPI_Comm comm, std::array<std::int64_t, 2> range,
                       const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& ghost_indices,
                       int block_size);

      /// Copy constructor
      BlockPETScVector(const BlockPETScVector& x);
      
      /// Move constructor
      BlockPETScVector(BlockPETScVector&& x);

      /// Create vector wrapper of PETSc Vec pointer. The reference
      /// counter of the Vec will be increased, and decreased upon
      /// destruction of this object.
      explicit BlockPETScVector(Vec x);

      /// Destructor
      virtual ~BlockPETScVector();
      
      // Assignment operator (disabled)
      BlockPETScVector& operator=(const BlockPETScVector& x) = delete;

      /// Move Assignment operator
      BlockPETScVector& operator=(BlockPETScVector&& x);
      
      //--- Special block functions ---
      
      /// Attach BlockDofMap for subvector creation
      void attach_block_dof_map(std::shared_ptr<const BlockDofMap> block_dof_map);
      
      /// Get BlockDofMap for subvector creation
      std::shared_ptr<const BlockDofMap> get_block_dof_map() const;
      
      /// Check if BlockDofMap for subvector creation has been attached
      bool has_block_dof_map() const;
      
      /// Block access
      std::shared_ptr<dolfin::GenericVector> operator()(std::size_t block_i, BlockInsertMode insert_mode) const;
      
    private:
      std::shared_ptr<const BlockDofMap> _block_dof_map;
    };
  }
}

#endif

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

      /// Create empty vector on an MPI communicator
      explicit BlockPETScVector(MPI_Comm comm);

      /// Create vector of size N
      BlockPETScVector(MPI_Comm comm, std::size_t N);

      /// Create vector
      explicit BlockPETScVector(const dolfin::SparsityPattern& sparsity_pattern);

      /// Copy constructor
      BlockPETScVector(const BlockPETScVector& x);

      /// Create vector wrapper of PETSc Vec pointer. The reference
      /// counter of the Vec will be increased, and decreased upon
      /// destruction of this object.
      explicit BlockPETScVector(Vec x);

      /// Destructor
      virtual ~BlockPETScVector();
      
      //--- Implementation of the GenericVector interface ---

      /// Return copy of vector
      virtual std::shared_ptr<dolfin::GenericVector> copy() const;
      
      /// Multiply vector by given number
      virtual const BlockPETScVector& operator*= (double a);

      /// Multiply vector by another vector pointwise
      virtual const BlockPETScVector& operator*= (const dolfin::GenericVector& x);

      /// Divide vector by given number
      virtual const BlockPETScVector& operator/= (double a);

      /// Add given vector
      virtual const BlockPETScVector& operator+= (const dolfin::GenericVector& x);

      /// Add number to all components of a vector
      virtual const BlockPETScVector& operator+= (double a);

      /// Subtract given vector
      virtual const BlockPETScVector& operator-= (const dolfin::GenericVector& x);

      /// Subtract number from all components of a vector
      virtual const BlockPETScVector& operator-= (double a);

      /// Assignment operator
      virtual const BlockPETScVector& operator= (const dolfin::GenericVector& x);

      /// Assignment operator
      virtual const BlockPETScVector& operator= (double a);
      
      //--- Special PETSc functions ---
      
      /// Assignment operator
      const BlockPETScVector& operator= (const BlockPETScVector& x);
      
      //--- Special block functions ---
      
      /// Attach BlockDofMap for subvector creation
      virtual void attach_block_dof_map(std::shared_ptr<const BlockDofMap> block_dof_map);
      
      /// Get BlockDofMap for subvector creation
      virtual std::shared_ptr<const BlockDofMap> get_block_dof_map() const;
      
      /// Check if BlockDofMap for subvector creation has been attached
      virtual bool has_block_dof_map() const;
      
      /// Block access
      virtual std::shared_ptr<dolfin::GenericVector> operator()(std::size_t block_i, BlockInsertMode insert_mode) const;
      
    private:
      std::shared_ptr<const BlockDofMap> _block_dof_map;
    };
  }
}

#endif

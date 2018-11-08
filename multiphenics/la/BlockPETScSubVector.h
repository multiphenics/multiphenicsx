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

#ifndef __BLOCK_PETSC_SUB_VECTOR_H
#define __BLOCK_PETSC_SUB_VECTOR_H

#include <dolfin/la/PETScVector.h>
#include <multiphenics/fem/BlockDofMap.h>
#include <multiphenics/la/BlockInsertMode.h>

namespace multiphenics
{
  namespace fem
  {
    // Forward declaration for friend function
    class BlockForm1;
    void block_assemble(dolfin::la::PETScVector&, const multiphenics::fem::BlockForm1&);
  }
  
  namespace la
  {
    
    /// This is an extension of PETScVector to be used while assemblying block forms, that
    /// a) carries out the extraction of a sub vector
    /// b) in case of restrictions, overrides get/set/add methods to convert original index without restriction to index with restriction
    class BlockPETScSubVector : public dolfin::la::PETScVector
    {
    public:
      /// Constructor
      BlockPETScSubVector(const dolfin::la::PETScVector & x,
                          std::size_t block_i,
                          std::shared_ptr<const multiphenics::fem::BlockDofMap> block_dof_map,
                          BlockInsertMode insert_mode);

      /// Destructor
      virtual ~BlockPETScSubVector();
      
      /// Copy constructor (deleted)
      BlockPETScSubVector(const BlockPETScSubVector& A) = delete;

      /// Move constructor (deleted)
      BlockPETScSubVector(BlockPETScSubVector&& A) = delete;
      
      // Assignment operator (deleted)
      BlockPETScSubVector& operator=(const BlockPETScSubVector& x) = delete;

      /// Move assignment operator (deleted)
      BlockPETScSubVector& operator=(BlockPETScSubVector&& x) = delete;
      
      /// Return size of vector.
      std::int64_t size() const;
      
      /// Set all entries to 'a' using VecSet. This is local and does not
      /// update ghost entries.
      void set(PetscScalar a);
      
      /// A scalar 'a' using VecSet. This is local and does not update ghost
      /// entries.
      void shift(PetscScalar a);

      /// Multiply by scala a
      void scale(PetscScalar a);

      /// Multiply vector by vector x pointwise
      void mult(const BlockPETScSubVector& x);

      /// Finalize assembly of vector. Communicates off-process entries
      /// added or set on this process to the owner, and receives from other
      /// processes changes to owned entries.
      void apply();

      /// Update owned entries owned by this process and which are ghosts on
      /// other processes, i.e., have been added to by a remote process.
      /// This is more efficient that apply() when processes only add/set
      /// their owned entries and the pre-defined ghosts.
      void apply_ghosts();

      /// Update ghost values (gathers ghost values from the owning
      /// processes)
      void update_ghosts();
      
      /// Get block of values using local indices
      void get_local(PetscScalar* block, std::size_t m,
                     const PetscInt* rows) const;

      /// Set block of values using global indices
      void set(const PetscScalar* block, std::size_t m,
               const PetscInt* rows);

      /// Set block of values using local indices
      void set_local(const PetscScalar* block, std::size_t m,
                     const PetscInt* rows);

      /// Add block of values using global indices
      void add(const PetscScalar* block, std::size_t m,
               const PetscInt* rows);

      /// Add block of values using local indices
      void add_local(const PetscScalar* block, std::size_t m,
                     const PetscInt* rows);

      /// Get all values on local process
      void get_local(std::vector<PetscScalar>& values) const;

      /// Set all values on local process
      void set_local(const std::vector<PetscScalar>& values);

      /// Add values to each entry on local process
      void add_local(const std::vector<PetscScalar>& values);
      
      /// Gather entries (given by global indices) into local
      /// (MPI_COMM_SELF) vector x. Provided x must be empty or of correct
      /// dimension (same as provided indices). This operation is
      /// collective.
      void gather(BlockPETScSubVector& y,
                  const std::vector<PetscInt>& indices) const;
      
      /// Add multiple of given vector (AXPY operation)
      void axpy(PetscScalar a, const BlockPETScSubVector& x);

      /// Replace all entries in the vector by their absolute values
      void abs();

      /// Return dot product with given vector. For complex vectors, the
      /// argument v gets complex conjugate.
      PetscScalar dot(const BlockPETScSubVector& v) const;

      /// Return norm of vector
      PetscReal norm(dolfin::la::Norm norm_type) const;

      /// Normalize vector with respect to the l2 norm. Returns the norm
      /// before normalization.
      PetscReal normalize();

      /// Return minimum value of vector, and location of entry. For complex
      /// vectors returns the minimum real part.
      std::pair<double, PetscInt> min() const;

      /// Return maximum value of vector, and location of entry. For complex
      /// vectors returns the maximum real part.
      std::pair<double, PetscInt> max() const;

      /// Return sum of entries
      PetscScalar sum() const;

      //--- Special PETSc functions ---

      /// Sets the prefix used by PETSc when searching the options
      /// database
      void set_options_prefix(std::string options_prefix);

      /// Returns the prefix used by PETSc when searching the options
      /// database
      std::string get_options_prefix() const;

      /// Call PETSc function VecSetFromOptions on the underlying Vec
      /// object
      void set_from_options();
      
    private:
      // Hide operations with PETScVector arguments
      using PETScVector::mult;
      using PETScVector::gather;
      using PETScVector::axpy;
      using PETScVector::dot;
      
      // Allow block_assemble to access to_restricted_vector_indices
      friend void multiphenics::fem::block_assemble(dolfin::la::PETScVector&, const multiphenics::fem::BlockForm1&);
      
      void to_restricted_subvector_indices(
        const std::vector<PetscInt> & block_unrestricted_subvector_indices, std::vector<PetscInt> & block_restricted_subvector_indices,
        std::vector<bool> * is_in_restriction = NULL
      ) const;
      void to_restricted_vector_indices(
        const std::vector<PetscInt> & block_unrestricted_subvector_indices, std::vector<PetscInt> & block_restricted_vector_indices,
        std::vector<bool> * is_in_restriction = NULL
      ) const;
      void to_restricted_subvector_indices_and_values(
        const std::vector<PetscInt> & block_unrestricted_subvector_indices, std::vector<PetscInt> & block_restricted_subvector_indices,
        const std::vector<PetscScalar> & block_unrestricted_subvector_values, std::vector<PetscScalar> & block_restricted_subvector_values
      ) const;
      void from_restricted_subvector_values(
        std::vector<PetscScalar> & block_unrestricted_subvector_values, const std::vector<PetscScalar> & block_restricted_subvector_values,
        const std::vector<bool> & is_in_restriction
      ) const;
      
      const dolfin::la::PETScVector & _global_vector;
      const std::map<PetscInt, PetscInt> & _original_to_sub_block;
      const std::map<PetscInt, PetscInt> & _original_to_block;
      /*PETSc*/ InsertMode _insert_mode;
      IS _is;
      Vec _x;
      VecScatter _scatter;
    };
  }
}

#endif

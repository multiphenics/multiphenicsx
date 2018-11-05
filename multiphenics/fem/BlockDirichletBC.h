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

#ifndef __BLOCK_DIRICHLET_BC_H
#define __BLOCK_DIRICHLET_BC_H

#include <dolfin/common/Variable.h>
#include <dolfin/fem/DirichletBC.h>
#include <multiphenics/function/BlockFunctionSpace.h>

namespace multiphenics
{

  class BlockDirichletBC: public dolfin::Variable
  {
  public:
    typedef dolfin::DirichletBC::Map Map;
    
    /// Create boundary condition for subdomain
    ///
    /// @param    bcs (list of list _DirichletBC_)
    ///         List (over blocks) of list (due to possible multiple BCs for each block) of DirichletBC objects
    BlockDirichletBC(std::vector<std::vector<std::shared_ptr<const dolfin::DirichletBC>>> bcs,
                     std::shared_ptr<const BlockFunctionSpace> block_function_space);

    /// Destructor
    virtual ~BlockDirichletBC();

    /// Apply boundary condition to a matrix
    ///
    /// @param     A (_GenericMatrix_)
    ///         The matrix to apply boundary condition to.
    void apply(dolfin::GenericMatrix& A,
               std::vector<std::vector<bool>> zero_off_block_diagonal) const;

    /// Apply boundary condition to a vector
    ///
    /// @param     b (_GenericVector_)
    ///         The vector to apply boundary condition to.
    void apply(dolfin::GenericVector& b) const;

    /// Apply boundary condition to a linear system
    ///
    /// @param     A (_GenericMatrix_)
    ///         The matrix to apply boundary condition to.
    /// @param     b (_GenericVector_)
    ///         The vector to apply boundary condition to.
    void apply(dolfin::GenericMatrix& A,
               dolfin::GenericVector& b,
               std::vector<std::vector<bool>> zero_off_block_diagonal) const;

    /// Apply boundary condition to vectors for a nonlinear problem
    ///
    /// @param    b (_GenericVector_)
    ///         The vector to apply boundary conditions to.
    /// @param     x (_GenericVector_)
    ///         Another vector (nonlinear problem).
    void apply(dolfin::GenericVector& b,
               const dolfin::GenericVector& x) const;

    /// Apply boundary condition to a linear system for a nonlinear problem
    ///
    /// @param     A (_GenericMatrix_)
    ///         The matrix to apply boundary conditions to.
    /// @param     b (_GenericVector_)
    ///         The vector to apply boundary conditions to.
    /// @param     x (_GenericVector_)
    ///         Another vector (nonlinear problem).
    void apply(dolfin::GenericMatrix& A,
               dolfin::GenericVector& b,
               const dolfin::GenericVector& x,
               std::vector<std::vector<bool>> zero_off_block_diagonal) const;
               
    /// Get Dirichlet dofs and values. If a method other than 'pointwise' is
    /// used in parallel, the map may not be complete for local vertices since
    /// a vertex can have a bc applied, but the partition might not have a
    /// facet on the boundary. To ensure all local boundary dofs are marked,
    /// it is necessary to call gather() on the returned boundary values.
    ///
    /// @param[in,out] boundary_values (Map&)
    ///         Map from dof to boundary value.
    void get_boundary_values(Map& boundary_values) const;

    /// Get boundary values from neighbour processes. If a method other than
    /// "pointwise" is used, this is necessary to ensure all boundary dofs are
    /// marked on all processes.
    ///
    /// @param[in,out] boundary_values (Map&)
    ///         Map from dof to boundary value.
    void gather(Map& boundary_values) const;
               
    /// Make rows of matrix associated with boundary condition zero,
    /// useful for non-diagonal matrices in a block matrix.
    ///
    /// @param[in,out] A (GenericMatrix&)
    ///         The matrix
    void zero(dolfin::GenericMatrix& A,
              std::vector<std::vector<bool>> zero_off_block_diagonal) const;
    
    /// Return the block function space
    ///
    /// @return BlockFunctionSpace
    ///         The block function space to which boundary conditions are applied.
    std::shared_ptr<const BlockFunctionSpace> block_function_space() const;

  private:
    void _original_to_block_boundary_values(Map& boundary_values, const Map& boundary_values_I, std::size_t I) const;

    std::vector<std::vector<std::shared_ptr<const dolfin::DirichletBC>>> _bcs;
    std::shared_ptr<const BlockFunctionSpace> _block_function_space;

  };

}

#endif

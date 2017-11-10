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

#ifndef __CONDENSED_BLOCK_SLEPC_EIGEN_SOLVER_H
#define __CONDENSED_BLOCK_SLEPC_EIGEN_SOLVER_H

#ifdef HAS_SLEPC

#include <multiphenics/fem/BlockDirichletBC.h>
#include <multiphenics/la/BlockPETScMatrix.h>
#include <multiphenics/la/CondensedSLEPcEigenSolver.h>

namespace multiphenics
{

  /// This class provides an eigenvalue solver for PETSc block matrices. It
  /// is a wrapper for the SLEPc eigenvalue solver. It also allows to
  /// constrain degrees of freedom associate to Dirichlet BCs.

  class CondensedBlockSLEPcEigenSolver : public dolfin::CondensedSLEPcEigenSolver
  {
  public:

    /// Create eigenvalue solver
    explicit CondensedBlockSLEPcEigenSolver(MPI_Comm comm);

    /// Create eigenvalue solver from EPS object
    explicit CondensedBlockSLEPcEigenSolver(EPS eps);

    /// Create eigenvalue solver for Ax = \lambda
    CondensedBlockSLEPcEigenSolver(std::shared_ptr<const BlockPETScMatrix> A,
                                   std::shared_ptr<const BlockDirichletBC> block_bcs);

    /// Create eigenvalue solver for Ax = \lambda x
    CondensedBlockSLEPcEigenSolver(MPI_Comm comm, std::shared_ptr<const BlockPETScMatrix> A,
                                   std::shared_ptr<const BlockDirichletBC> block_bcs);

    /// Create eigenvalue solver for Ax = \lambda x on MPI_COMM_WORLD
    CondensedBlockSLEPcEigenSolver(std::shared_ptr<const BlockPETScMatrix> A,
                                   std::shared_ptr<const BlockPETScMatrix> B,
                                   std::shared_ptr<const BlockDirichletBC> block_bcs);

    /// Create eigenvalue solver for Ax = \lambda x
    CondensedBlockSLEPcEigenSolver(MPI_Comm comm, std::shared_ptr<const BlockPETScMatrix> A,
                                   std::shared_ptr<const BlockPETScMatrix> B,
                                   std::shared_ptr<const BlockDirichletBC> block_bcs);

    /// Destructor
    ~CondensedBlockSLEPcEigenSolver();

    /// Set boundary conditions
    void set_boundary_conditions(std::shared_ptr<const BlockDirichletBC> block_bcs);
    
    /// Get ith eigenpair
    void get_eigenpair(double& lr, double& lc,
                       dolfin::GenericVector& r, dolfin::GenericVector& c, std::size_t i) const;

    /// Get ith eigenpair
    void get_eigenpair(double& lr, double& lc,
                       dolfin::PETScVector& r, dolfin::PETScVector& c, std::size_t i) const;
  
  private:
    /// Hide Parent's version of boundary conditions setter
    using CondensedSLEPcEigenSolver::set_boundary_conditions;
  };

}

#endif

#endif

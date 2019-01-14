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

#ifndef __CONDENSED_SLEPC_EIGEN_SOLVER_H
#define __CONDENSED_SLEPC_EIGEN_SOLVER_H

#ifdef HAS_SLEPC

#include <dolfin/fem/DirichletBC.h>
#include <dolfin/la/SLEPcEigenSolver.h>

namespace dolfin
{

  /// This class provides an eigenvalue solver for PETSc matrices. It
  /// is a wrapper for the SLEPc eigenvalue solver. It also allows to
  /// constrain degrees of freedom associate to Dirichlet BCs.

  class CondensedSLEPcEigenSolver : public SLEPcEigenSolver
  {
  public:

    /// Create eigenvalue solver
    explicit CondensedSLEPcEigenSolver(MPI_Comm comm);

    /// Create eigenvalue solver from EPS object
    explicit CondensedSLEPcEigenSolver(EPS eps);

    /// Create eigenvalue solver for Ax = \lambda
    CondensedSLEPcEigenSolver(std::shared_ptr<const PETScMatrix> A,
                              std::vector<std::shared_ptr<const DirichletBC>> bcs);

    /// Create eigenvalue solver for Ax = \lambda x
    CondensedSLEPcEigenSolver(MPI_Comm comm, std::shared_ptr<const PETScMatrix> A,
                              std::vector<std::shared_ptr<const DirichletBC>> bcs);

    /// Create eigenvalue solver for Ax = \lambda x on MPI_COMM_WORLD
    CondensedSLEPcEigenSolver(std::shared_ptr<const PETScMatrix> A,
                              std::shared_ptr<const PETScMatrix> B,
                              std::vector<std::shared_ptr<const DirichletBC>> bcs);

    /// Create eigenvalue solver for Ax = \lambda x
    CondensedSLEPcEigenSolver(MPI_Comm comm, std::shared_ptr<const PETScMatrix> A,
                              std::shared_ptr<const PETScMatrix> B,
                              std::vector<std::shared_ptr<const DirichletBC>> bcs);

    /// Destructor
    ~CondensedSLEPcEigenSolver();
    
    /// Set opeartors (B may be nullptr for regular eigenvalues
    /// problems)
    void set_operators(std::shared_ptr<const PETScMatrix> A,
                       std::shared_ptr<const PETScMatrix> B);
    
    /// Set boundary conditions
    void set_boundary_conditions(std::vector<std::shared_ptr<const DirichletBC>> bcs);

    /// Get ith eigenpair
    void get_eigenpair(double& lr, double& lc,
                       GenericVector& r, GenericVector& c, std::size_t i) const;

    /// Get ith eigenpair
    void get_eigenpair(double& lr, double& lc,
                       PETScVector& r, PETScVector& c, std::size_t i) const;
    
  protected:
    IS _is;
    
  private:
    Mat _condense_matrix(std::shared_ptr<const PETScMatrix> mat);
    
    std::shared_ptr<const PETScMatrix> _A;
    std::shared_ptr<const PETScMatrix> _B;
    Mat _condensed_A;
    Mat _condensed_B;

  };

}

#endif

#endif

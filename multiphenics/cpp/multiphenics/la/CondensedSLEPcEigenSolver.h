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

#ifndef __CONDENSED_SLEPC_EIGEN_SOLVER_H
#define __CONDENSED_SLEPC_EIGEN_SOLVER_H

#ifdef HAS_SLEPC

#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/la/SLEPcEigenSolver.h>

namespace dolfinx
{
  namespace la
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

      /// Destructor
      ~CondensedSLEPcEigenSolver();

      /// Set operators (B may be nullptr for regular eigenvalues
      /// problems)
      void set_operators(const Mat A, const Mat B);

      /// Set boundary conditions. This method must be called *before* setting operators.
      void set_boundary_conditions(std::vector<std::shared_ptr<const fem::DirichletBC>> bcs);

      /// Get ith eigenpair
      void get_eigenpair(PetscScalar& lr, PetscScalar& lc,
                         Vec r, Vec c, std::size_t i) const;

    protected:
      IS _is;

    private:
      Mat _condense_matrix(const Mat mat) const;

      Mat _A;
      Mat _B;
      Mat _condensed_A;
      Mat _condensed_B;

    };
  }
}

#endif

#endif

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

#ifndef __CONDENSED_BLOCK_SLEPC_EIGEN_SOLVER_H
#define __CONDENSED_BLOCK_SLEPC_EIGEN_SOLVER_H

#ifdef HAS_SLEPC

#include <multiphenics/fem/BlockDirichletBC.h>
#include <multiphenics/la/CondensedSLEPcEigenSolver.h>

namespace multiphenics
{
  namespace la
  {
    /// This class provides an eigenvalue solver for PETSc block matrices. It
    /// is a wrapper for the SLEPc eigenvalue solver. It also allows to
    /// constrain degrees of freedom associate to Dirichlet BCs.

    class CondensedBlockSLEPcEigenSolver : public dolfinx::la::CondensedSLEPcEigenSolver
    {
    public:

      /// Create eigenvalue solver
      explicit CondensedBlockSLEPcEigenSolver(MPI_Comm comm);

      /// Create eigenvalue solver from EPS object
      explicit CondensedBlockSLEPcEigenSolver(EPS eps);

      /// Destructor
      ~CondensedBlockSLEPcEigenSolver();

      /// Set boundary conditions. This method must be called *before* setting operators.
      void set_boundary_conditions(std::shared_ptr<const fem::BlockDirichletBC> block_bcs);
      
    private:
      /// Hide Parent's version of boundary conditions setter
      using CondensedSLEPcEigenSolver::set_boundary_conditions;
    };
  }
}

#endif

#endif

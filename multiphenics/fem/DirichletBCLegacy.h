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

#ifndef __DIRICHLET_BC_LEGACY_H
#define __DIRICHLET_BC_LEGACY_H

#include <petscmat.h>
#include <petscvec.h>
#include <dolfin/fem/DirichletBC.h>

namespace dolfin
{
  namespace fem
  {
    class DirichletBCLegacy
    {
    public:
      /// Apply list of boundary conditions to a matrix
      static void apply(std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>> bcs,
                        Mat A,
                        PetscScalar diag);

      /// Apply list of boundary conditions to a vector
      static void apply(std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>> bcs,
                        Vec b);

      /// Apply list of boundary conditions to vectors for a nonlinear problem
      static void apply(std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>> bcs,
                        Vec b,
                        const Vec x);
    };
  }
}

#endif

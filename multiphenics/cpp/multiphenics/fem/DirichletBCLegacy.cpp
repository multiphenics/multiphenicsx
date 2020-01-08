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

#include <dolfin/fem/assembler.h>
#include <multiphenics/fem/DirichletBCLegacy.h>

using dolfin::fem::DirichletBCLegacy;
using dolfin::fem::set_bc;

//-----------------------------------------------------------------------------
void DirichletBCLegacy::apply(std::vector<std::shared_ptr<const DirichletBC>> bcs, Mat A, PetscScalar diag)
{
  for (auto bc: bcs)
  {
    const auto rows = bc->dof_indices();
    MatZeroRowsLocal(A, rows.size(), rows.data(), diag, NULL, NULL);
  }
}
//-----------------------------------------------------------------------------
void DirichletBCLegacy::apply(std::vector<std::shared_ptr<const DirichletBC>> bcs, Vec b)
{
  set_bc(b, bcs, nullptr);
}
//-----------------------------------------------------------------------------
void DirichletBCLegacy::apply(std::vector<std::shared_ptr<const DirichletBC>> bcs, Vec b, const Vec x)
{
  set_bc(b, bcs, x, -1.);
}
//-----------------------------------------------------------------------------

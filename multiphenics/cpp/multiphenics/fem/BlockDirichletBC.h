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

#include <dolfinx/fem/DirichletBC.h>
#include <multiphenics/function/BlockFunctionSpace.h>

namespace multiphenics
{
  namespace fem
  {
    class BlockDirichletBC
    {
    public:
      /// Create boundary condition for subdomain
      ///
      /// @param    bcs (list of list _DirichletBC_)
      ///         List (over blocks) of list (due to possible multiple BCs for each block) of DirichletBC objects
      BlockDirichletBC(std::vector<std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC>>> bcs,
                       std::shared_ptr<const multiphenics::function::BlockFunctionSpace> block_function_space);

      /// Destructor
      ~BlockDirichletBC();
      
      /// Return the block function space
      ///
      /// @return multiphenics::function::BlockFunctionSpace
      ///         The block function space to which boundary conditions are applied.
      std::shared_ptr<const multiphenics::function::BlockFunctionSpace> block_function_space() const;
      
      std::size_t size() const;
      std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC>> operator[](std::size_t I) const;

    private:
      std::vector<std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC>>> _bcs;
      std::shared_ptr<const multiphenics::function::BlockFunctionSpace> _block_function_space;

    };
  }
}

#endif

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
  namespace fem
  {
    class BlockDirichletBC: public dolfin::common::Variable
    {
    public:
      typedef dolfin::fem::DirichletBC::Map Map;
      
      /// Create boundary condition for subdomain
      ///
      /// @param    bcs (list of list _DirichletBC_)
      ///         List (over blocks) of list (due to possible multiple BCs for each block) of DirichletBC objects
      BlockDirichletBC(std::vector<std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>>> bcs,
                       std::shared_ptr<const multiphenics::function::BlockFunctionSpace> block_function_space);

      /// Destructor
      ~BlockDirichletBC();
      
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
            
      /// Return the block function space
      ///
      /// @return multiphenics::function::BlockFunctionSpace
      ///         The block function space to which boundary conditions are applied.
      std::shared_ptr<const multiphenics::function::BlockFunctionSpace> block_function_space() const;
      
      std::size_t size() const;
      std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>> operator[](std::size_t I) const;

    private:
      void _original_to_block_boundary_values(Map& boundary_values, const Map& boundary_values_I, std::size_t I) const;

      std::vector<std::vector<std::shared_ptr<const dolfin::fem::DirichletBC>>> _bcs;
      std::shared_ptr<const multiphenics::function::BlockFunctionSpace> _block_function_space;

    };
  }
}

#endif

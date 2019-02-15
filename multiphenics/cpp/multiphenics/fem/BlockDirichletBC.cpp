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

#include <multiphenics/fem/BlockDirichletBC.h>

using namespace multiphenics;
using namespace multiphenics::fem;

using dolfin::fem::DirichletBC;
using multiphenics::function::BlockFunctionSpace;

//-----------------------------------------------------------------------------
BlockDirichletBC::BlockDirichletBC(std::vector<std::vector<std::shared_ptr<const DirichletBC>>> bcs,
                                   std::shared_ptr<const BlockFunctionSpace> block_function_space)
  : _bcs(bcs), _block_function_space(block_function_space)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockDirichletBC::~BlockDirichletBC()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::shared_ptr<const BlockFunctionSpace> BlockDirichletBC::block_function_space() const
{
  return _block_function_space;
}
//-----------------------------------------------------------------------------
std::size_t BlockDirichletBC::size() const
{
  return _bcs.size();
}
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<const DirichletBC>> BlockDirichletBC::operator[](std::size_t I) const
{
  return _bcs[I];
}
//-----------------------------------------------------------------------------

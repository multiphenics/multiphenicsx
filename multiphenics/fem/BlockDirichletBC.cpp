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

using namespace dolfin;
using namespace dolfin::fem;
using namespace multiphenics;
using namespace multiphenics::fem;

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
void BlockDirichletBC::get_boundary_values(Map& boundary_values) const
{
  for (std::size_t I(0); I < _bcs.size(); ++I)
  {
    Map boundary_values_I;
    for (auto & bc_I: _bcs[I])
    {
      bc_I->get_boundary_values(boundary_values_I);
    }
    _original_to_block_boundary_values(boundary_values, boundary_values_I, I);
  }
}
//-----------------------------------------------------------------------------
void BlockDirichletBC::gather(Map& boundary_values) const
{
  for (std::size_t I(0); I < _bcs.size(); ++I)
  {
    Map boundary_values_I;
    for (auto & bc_I: _bcs[I])
    {
      bc_I->gather(boundary_values_I);
    }
    _original_to_block_boundary_values(boundary_values, boundary_values_I, I);
  }
}
//-----------------------------------------------------------------------------
void BlockDirichletBC::_original_to_block_boundary_values(Map& boundary_values, const Map& boundary_values_I, std::size_t I) const
{
  const auto & original_to_block = _block_function_space->block_dofmap()->original_to_block(I);
  for (auto bc_original_local_index_to_value : boundary_values_I)
  {
    auto original_local_index = bc_original_local_index_to_value.first;
    auto value = bc_original_local_index_to_value.second;
    if (original_to_block.count(original_local_index) > 0)
    {
      auto block_local_index = original_to_block.at(original_local_index);
      boundary_values[block_local_index] = value;
    }
  }
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

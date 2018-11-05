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

#include <multiphenics/fem/BlockFormBase.h>

using namespace dolfin;
using namespace multiphenics;

//-----------------------------------------------------------------------------
BlockFormBase::BlockFormBase(std::vector<std::shared_ptr<const BlockFunctionSpace>> block_function_spaces):
  _block_function_spaces(block_function_spaces)
{
}
//-----------------------------------------------------------------------------
BlockFormBase::~BlockFormBase()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<const BlockFunctionSpace>> BlockFormBase::block_function_spaces() const
{
  return _block_function_spaces;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Mesh> BlockFormBase::mesh() const
{
  std::shared_ptr<const Mesh> mesh = _block_function_spaces[0]->mesh();
  #ifdef DEBUG
  for (auto & block_function_space : _block_function_spaces)
    dolfin_assert(block_function_space->mesh() == mesh);
  #endif
  return mesh;
}

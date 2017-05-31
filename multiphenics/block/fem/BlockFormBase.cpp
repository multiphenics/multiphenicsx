// Copyright (C) 2016-2017 by the block_ext authors
//
// This file is part of block_ext.
//
// block_ext is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// block_ext is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with block_ext. If not, see <http://www.gnu.org/licenses/>.
//

#include <block/fem/BlockFormBase.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BlockFormBase::BlockFormBase(std::vector<std::shared_ptr<const BlockFunctionSpace>> block_function_spaces):
  Hierarchical<BlockFormBase>(*this), _block_function_spaces(block_function_spaces)
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
  for (auto & block_function_space : _block_function_spaces)
    dolfin_assert(block_function_space->mesh() == mesh);
  return mesh;
}

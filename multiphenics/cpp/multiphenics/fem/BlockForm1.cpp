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

#include <multiphenics/fem/BlockForm1.h>

using namespace multiphenics;
using namespace multiphenics::fem;

using dolfinx::fem::Form;
using dolfinx::mesh::Mesh;
using multiphenics::function::BlockFunctionSpace;

//-----------------------------------------------------------------------------
BlockForm1::BlockForm1(std::vector<std::shared_ptr<const Form>> forms,
                       std::vector<std::shared_ptr<const BlockFunctionSpace>> block_function_spaces):
  _forms(forms), _block_function_spaces(block_function_spaces), _block_size(forms.size())
{
}
//-----------------------------------------------------------------------------
BlockForm1::~BlockForm1()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<const BlockFunctionSpace>> BlockForm1::block_function_spaces() const
{
  return _block_function_spaces;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Mesh> BlockForm1::mesh() const
{
  return _block_function_spaces[0]->mesh();
}
//-----------------------------------------------------------------------------
unsigned int BlockForm1::block_size(unsigned int d) const
{
  return _block_size;
}
//-----------------------------------------------------------------------------
const Form & BlockForm1::operator()(std::size_t i) const
{
  return *_forms[i];
}
//-----------------------------------------------------------------------------

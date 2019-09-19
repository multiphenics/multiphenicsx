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

#include <multiphenics/fem/BlockForm2.h>

using namespace multiphenics;
using namespace multiphenics::fem;

using dolfin::fem::Form;
using dolfin::mesh::Mesh;
using multiphenics::function::BlockFunctionSpace;

//-----------------------------------------------------------------------------
BlockForm2::BlockForm2(std::vector<std::vector<std::shared_ptr<const Form>>> forms,
                       std::vector<std::shared_ptr<const BlockFunctionSpace>> block_function_spaces):
  _forms(forms), _block_function_spaces(block_function_spaces), _block_size(2)
{
  _block_size[0] = forms.size();
  _block_size[1] = forms[0].size();
}
//-----------------------------------------------------------------------------
BlockForm2::~BlockForm2()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<const BlockFunctionSpace>> BlockForm2::block_function_spaces() const
{
  return _block_function_spaces;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Mesh> BlockForm2::mesh() const
{
  std::shared_ptr<const Mesh> mesh = _block_function_spaces[0]->mesh;
  assert(_block_function_spaces[1]->mesh == mesh);
  return mesh;
}
//-----------------------------------------------------------------------------
unsigned int BlockForm2::block_size(unsigned int d) const
{
  return _block_size[d];
}
//-----------------------------------------------------------------------------
const Form & BlockForm2::operator()(std::size_t i, std::size_t j) const
{
  return *_forms[i][j];
}
//-----------------------------------------------------------------------------

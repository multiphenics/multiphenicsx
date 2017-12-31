// Copyright (C) 2016-2018 by the multiphenics authors
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

using namespace dolfin;
using namespace multiphenics;

//-----------------------------------------------------------------------------
BlockForm1::BlockForm1(std::vector<std::shared_ptr<const Form>> forms,
                       std::vector<std::shared_ptr<const BlockFunctionSpace>> block_function_spaces):
  BlockFormBase(block_function_spaces), _forms(forms), _block_size(forms.size())
{
}
//-----------------------------------------------------------------------------
BlockForm1::~BlockForm1()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t BlockForm1::rank() const
{
  return 1;
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
bool BlockForm1::has_cell_integrals() const
{
  for (auto & form : _forms)
    if (form->ufc_form()->has_cell_integrals())
      return true;
  return false;
}
//-----------------------------------------------------------------------------
bool BlockForm1::has_interior_facet_integrals() const
{
  for (auto & form : _forms)
    if (form->ufc_form()->has_interior_facet_integrals())
      return true;
  return false;
}
//-----------------------------------------------------------------------------
bool BlockForm1::has_exterior_facet_integrals() const
{
  for (auto & form : _forms)
    if (form->ufc_form()->has_exterior_facet_integrals())
      return true;
  return false;
}
//-----------------------------------------------------------------------------
bool BlockForm1::has_vertex_integrals() const
{
  for (auto & form : _forms)
    if (form->ufc_form()->has_vertex_integrals())
      return true;
  return false;
}
//-----------------------------------------------------------------------------

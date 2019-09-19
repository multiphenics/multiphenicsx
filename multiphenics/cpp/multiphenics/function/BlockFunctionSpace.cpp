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

#include <dolfin/common/IndexMap.h>
#include <dolfin/common/UniqueIdGenerator.h>
#include <multiphenics/function/BlockFunctionSpace.h>

using namespace multiphenics;
using namespace multiphenics::function;

using dolfin::common::UniqueIdGenerator;
using dolfin::EigenRowArrayXXd;
using dolfin::fem::DofMap;
using dolfin::fem::FiniteElement;
using dolfin::function::FunctionSpace;
using dolfin::mesh::Mesh;
using dolfin::mesh::MeshFunction;
using multiphenics::fem::BlockDofMap;

//-----------------------------------------------------------------------------
BlockFunctionSpace::BlockFunctionSpace(std::vector<std::shared_ptr<const FunctionSpace>> function_spaces)
  : function_spaces(function_spaces), restrictions(function_spaces.size()), _root_space_id(UniqueIdGenerator::id())
{
  _init_mesh_and_elements_and_dofmaps_from_function_spaces();
  _init_block_dofmap_from_dofmaps_and_restrictions();
}
//-----------------------------------------------------------------------------
BlockFunctionSpace::BlockFunctionSpace(std::vector<std::shared_ptr<const FunctionSpace>> function_spaces,
                                       std::vector<std::vector<std::shared_ptr<const MeshFunction<std::size_t>>>> restrictions)
  : function_spaces(function_spaces), restrictions(restrictions), _root_space_id(UniqueIdGenerator::id())
{
  _init_mesh_and_elements_and_dofmaps_from_function_spaces();
  _init_block_dofmap_from_dofmaps_and_restrictions();
}
//-----------------------------------------------------------------------------
BlockFunctionSpace::BlockFunctionSpace(std::shared_ptr<const Mesh> mesh,
                                       std::vector<std::shared_ptr<const FiniteElement>> elements,
                                       std::vector<std::shared_ptr<const DofMap>> dofmaps)
  : mesh(mesh), elements(elements), dofmaps(dofmaps), restrictions(dofmaps.size()), _root_space_id(UniqueIdGenerator::id())
{
  _init_function_spaces_from_elements_and_dofmaps();
  _init_block_dofmap_from_dofmaps_and_restrictions();
}
//-----------------------------------------------------------------------------
BlockFunctionSpace::BlockFunctionSpace(std::shared_ptr<const Mesh> mesh,
                                       std::vector<std::shared_ptr<const FiniteElement>> elements,
                                       std::vector<std::shared_ptr<const DofMap>> dofmaps,
                                       std::vector<std::vector<std::shared_ptr<const MeshFunction<std::size_t>>>> restrictions)
  : mesh(mesh), elements(elements), dofmaps(dofmaps), restrictions(restrictions), _root_space_id(UniqueIdGenerator::id())
{
  _init_function_spaces_from_elements_and_dofmaps();
  _init_block_dofmap_from_dofmaps_and_restrictions();
}
//-----------------------------------------------------------------------------
void BlockFunctionSpace::_init_mesh_and_elements_and_dofmaps_from_function_spaces() {
  mesh = function_spaces[0]->mesh;
  for (auto& function_space : function_spaces) 
  {
    assert(mesh == function_space->mesh);
    elements.push_back(function_space->element);
    dofmaps.push_back(function_space->dofmap);
  }
}
//-----------------------------------------------------------------------------
void BlockFunctionSpace::_init_function_spaces_from_elements_and_dofmaps() {
  assert(elements.size() == dofmaps.size());
  for (unsigned int i(0); i < elements.size(); ++i) 
  {
    std::shared_ptr<const FunctionSpace> function_space_i(new FunctionSpace(mesh, elements[i], dofmaps[i]));
    function_spaces.push_back(function_space_i);
  }
}
//-----------------------------------------------------------------------------
void BlockFunctionSpace::_init_block_dofmap_from_dofmaps_and_restrictions() {
  block_dofmap = std::make_shared<BlockDofMap>(dofmaps, restrictions, *mesh);
}
//-----------------------------------------------------------------------------
bool BlockFunctionSpace::operator==(const BlockFunctionSpace& V) const
{
  // Compare pointers to shared objects
  
  // -> elements
  if (elements.size() != V.elements.size())
    return false;
  for (unsigned int i(0); i < elements.size(); ++i)
    if (elements[i].get() != V.elements[i].get())
      return false;
      
  // -> dofmaps
  if (dofmaps.size() != V.dofmaps.size())
    return false;
  for (unsigned int i(0); i < dofmaps.size(); ++i)
    if (dofmaps[i].get() != V.dofmaps[i].get())
      return false;
      
  // -> restrictions
  if (restrictions.size() != V.restrictions.size())
    return false;
  for (unsigned int i(0); i < restrictions.size(); ++i)
    for (unsigned int d(0); d < restrictions[i].size(); ++d)
      if (restrictions[i][d].get() != V.restrictions[i][d].get())
        return false;
      
  // -> function_spaces
  if (function_spaces.size() != V.function_spaces.size())
    return false;
  for (unsigned int i(0); i < function_spaces.size(); ++i)
    if (function_spaces[i].get() != V.function_spaces[i].get())
      return false;
      
  // -> mesh and block_dofmap
  return 
    mesh.get() == V.mesh.get() &&
    block_dofmap.get() == V.block_dofmap.get();
}
//-----------------------------------------------------------------------------
bool BlockFunctionSpace::operator!=(const BlockFunctionSpace& V) const
{
  // Compare pointers to shared objects
  return !(*this == V);
}
//-----------------------------------------------------------------------------
std::int64_t BlockFunctionSpace::dim() const
{
  assert(block_dofmap);
  return block_dofmap->index_map->size_global();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FunctionSpace> BlockFunctionSpace::operator[] (std::size_t i) const
{
  return function_spaces[i];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FunctionSpace> BlockFunctionSpace::sub(std::size_t i) const
{
  return function_spaces[i];
}
//-----------------------------------------------------------------------------
std::shared_ptr<BlockFunctionSpace>
BlockFunctionSpace::extract_block_sub_space(const std::vector<std::size_t>& component, bool with_restrictions) const
{
  assert(mesh);

  // Check if sub space is already in the cache
  BlockSubpsacesType* block_subspaces;
  if (with_restrictions)
    block_subspaces = &_block_subspaces__with_restrictions;
  else
    block_subspaces = &_block_subspaces__without_restrictions;
  BlockSubpsacesType::const_iterator subspace = block_subspaces->find(component);
  if (subspace != block_subspaces->end())
    return subspace->second;
  else
  {
    // Extract sub elements
    std::vector<std::shared_ptr<const FiniteElement>> sub_elements;
    for (auto c: component)
      sub_elements.push_back(elements[c]);

    // Extract sub dofmaps
    std::vector<std::shared_ptr<const DofMap>> sub_dofmaps;
    for (auto c: component)
      sub_dofmaps.push_back(dofmaps[c]);

    // Extract restrictions, if required
    std::vector<std::vector<std::shared_ptr<const MeshFunction<std::size_t>>>> sub_restrictions;
    if (with_restrictions)
      for (auto c: component)
        sub_restrictions.push_back(restrictions[c]);
    
    // Create new block sub space
    std::shared_ptr<BlockFunctionSpace>
      new_block_sub_space(new BlockFunctionSpace(mesh, sub_elements, sub_dofmaps, sub_restrictions));

    // Set root space id and component w.r.t. root
    new_block_sub_space->_root_space_id = _root_space_id;
    auto& new_component = new_block_sub_space->_component;
    new_component.clear();
    new_component.insert(new_component.end(), _component.begin(), _component.end());
    new_component.insert(new_component.end(), component.begin(), component.end());

    // Insert new sub space into cache
    block_subspaces->insert(std::pair<std::vector<std::size_t>, std::shared_ptr<BlockFunctionSpace>>(
                              component, new_block_sub_space
                            ));

    return new_block_sub_space;
  }
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> BlockFunctionSpace::component() const
{
  return _component;
}
//-----------------------------------------------------------------------------
std::string BlockFunctionSpace::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    // No verbose output implemented
  }
  else
    s << "<BlockFunctionSpace of dimension " << dim() << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
bool BlockFunctionSpace::contains(const BlockFunctionSpace& V) const
{
  // Is the root space same?
  if (_root_space_id != V._root_space_id)
    return false;

  // Is V possibly our superspace?
  if (_component.size() > V._component.size())
    return false;

  // Are our components same as leading components of V?
  for (std::size_t i = 0; i < _component.size(); ++i)
  {
    if (_component[i] != V._component[i])
      return false;
  }

  // Ok, V is really our subspace
  return true;
}
//-----------------------------------------------------------------------------
EigenRowArrayXXd BlockFunctionSpace::tabulate_dof_coordinates() const
{
  // Geometric dimension
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim();
  
  // Get local size
  assert(block_dofmap);
  std::size_t local_size = block_dofmap->index_map->size_local();
  
  // Vector to hold coordinates and return
  EigenRowArrayXXd dof_coordinates(local_size, gdim);
  
  // Loop over subspaces
  for (unsigned int i(0); i < function_spaces.size(); ++i)
  {
    auto function_space = function_spaces[i];
    
    // Get dof coordinates of function space
    auto sub_dof_coordinates = function_space->tabulate_dof_coordinates();
    
    // Get original to block numbering
    auto original_to_block = block_dofmap->original_to_block(i);
    
    // Loop over all original dofs
    for (unsigned int d(0); d < sub_dof_coordinates.rows(); ++d)
    {
      if (original_to_block.count(d) > 0) // skip all dofs which have been removed by restriction
      {
        dof_coordinates.row(original_to_block.at(d)) = sub_dof_coordinates.row(d);
      }
    }
  }
  
  return dof_coordinates;
}
//-----------------------------------------------------------------------------

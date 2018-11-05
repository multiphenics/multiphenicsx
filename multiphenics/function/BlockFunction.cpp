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

#include <dolfin/la/GenericVector.h>
#include <multiphenics/function/BlockFunction.h>
#include <multiphenics/log/log.h>

using namespace dolfin;
using namespace dolfin::function;
using namespace multiphenics;
using namespace multiphenics::function;

//-----------------------------------------------------------------------------
BlockFunction::BlockFunction(std::shared_ptr<const BlockFunctionSpace> V)
  : _block_function_space(V), _sub_function_spaces(V->function_spaces())
{
  // Initialize block vector
  init_block_vector();
  
  // Initialize sub functions
  init_sub_functions();
}
//-----------------------------------------------------------------------------
BlockFunction::BlockFunction(std::shared_ptr<const BlockFunctionSpace> V,
                             std::vector<std::shared_ptr<Function>> sub_functions)
  : _block_function_space(V), _sub_function_spaces(V->function_spaces()), _sub_functions(sub_functions)
{
  // Initialize block vector
  init_block_vector();
  
  // Apply from subfunctions
  apply("from subfunctions");
}
//-----------------------------------------------------------------------------
BlockFunction::BlockFunction(std::shared_ptr<const BlockFunctionSpace> V,
                             std::shared_ptr<GenericVector> x)
  : _block_function_space(V), _block_vector(x), _sub_function_spaces(V->function_spaces())
{
  // Initialize sub functions
  init_sub_functions();
  
  // Apply to subfunctions
  apply("to subfunctions");
}
//-----------------------------------------------------------------------------
BlockFunction::BlockFunction(std::shared_ptr<const BlockFunctionSpace> V,
                             std::shared_ptr<GenericVector> x,
                             std::vector<std::shared_ptr<Function>> sub_functions)
  : _block_function_space(V), _block_vector(x), _sub_function_spaces(V->function_spaces()), _sub_functions(sub_functions)
{
  // Apply to subfunctions
  apply("to subfunctions");
}
//-----------------------------------------------------------------------------
BlockFunction::BlockFunction(const BlockFunction& v)
{
  // Assign data
  *this = v;
}
//-----------------------------------------------------------------------------
BlockFunction::~BlockFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const BlockFunction& BlockFunction::operator= (const BlockFunction& v)
{
  // Copy function space
  _block_function_space = v._block_function_space;

  // Copy vector
  _block_vector = v._block_vector->copy();
  
  // Copy sub function spaces
  _sub_function_spaces = v._sub_function_spaces;
  
  // Copy sub functions
  _sub_functions.clear();
  for (auto v_sub_function: v._sub_functions)
  {
    _sub_functions.push_back(std::make_shared<Function>(*v_sub_function));
  }

  return *this;
}
//-----------------------------------------------------------------------------
std::shared_ptr<Function> BlockFunction::operator[] (std::size_t i) const
{
  return _sub_functions[i];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const BlockFunctionSpace> BlockFunction::block_function_space() const
{
  dolfin_assert(_block_function_space);
  return _block_function_space;
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> BlockFunction::block_vector()
{
  dolfin_assert(_block_vector);
  return _block_vector;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const GenericVector> BlockFunction::block_vector() const
{
  dolfin_assert(_block_vector);
  return _block_vector;
}
//-----------------------------------------------------------------------------
void BlockFunction::init_block_vector()
{
  // This method has been adapted from
  //    Function::init_vector
  
  // Get dof map
  dolfin_assert(_block_function_space);
  dolfin_assert(_block_function_space->block_dofmap());
  const GenericDofMap& dofmap = *(_block_function_space->block_dofmap());

  // Get index map
  std::shared_ptr<const IndexMap> index_map = dofmap.index_map();
  dolfin_assert(index_map);

  // Create layout for initialising tensor
  std::shared_ptr<TensorLayout> tensor_layout;
  tensor_layout = create_layout(_block_function_space->mesh()->mpi_comm(), 1);
  dolfin_assert(tensor_layout);
  dolfin_assert(!tensor_layout->sparsity_pattern());
  dolfin_assert(_block_function_space->mesh());
  tensor_layout->init({index_map}, TensorLayout::Ghosts::GHOSTED);

  // Create vector of dofs
  if (!_block_vector)
  {
    _block_vector = std::make_shared<BlockPETScVector>(_block_function_space->mesh()->mpi_comm());
    block_vector->attach_block_dof_map(_block_function_space->block_dofmap());
  }
  dolfin_assert(_block_vector);
  if (!_block_vector->empty())
  {
    multiphenics_error("BlockFunction.cpp",
                       "initialize vector of degrees of freedom for function",
                       "Cannot re-initialize a non-empty vector. Consider creating a new function");

  }
  _block_vector->init(*tensor_layout);
  _block_vector->zero();
}
//-----------------------------------------------------------------------------
void BlockFunction::init_sub_functions()
{
  for (auto sub_function_space : _sub_function_spaces)
    _sub_functions.push_back(std::make_shared<Function>(sub_function_space));
}
//-----------------------------------------------------------------------------
void BlockFunction::apply(std::string mode, int only)
{
  auto block_dof_map(_block_function_space->block_dofmap());
  unsigned int i(0);
  unsigned int i_max(_sub_functions.size());
  if (only >= 0) {
    i = static_cast<unsigned int>(only);
    i_max = i + 1;
  }
  for (; i < i_max; ++i)
  {
    if (mode == "to subfunctions")
    {
      std::vector<dolfin::la_index> indices;
      std::vector<dolfin::la_index> sub_indices;
      for (auto & block_to_original: block_dof_map->block_to_original(i))
      {
        indices.push_back(block_to_original.first);
        sub_indices.push_back(block_to_original.second);
      }
      std::vector<double> values(indices.size());
      _block_vector->get_local(values.data(), indices.size(), indices.data());
      _sub_functions[i]->vector()->set_local(values.data(), sub_indices.size(), sub_indices.data());
      _sub_functions[i]->vector()->apply("insert");
    }
    else if (mode == "from subfunctions")
    {
      std::shared_ptr<GenericVector> sub_block_vector(
        _block_vector->operator()(i, BlockInsertMode::INSERT_VALUES)
      );
      std::vector<double> local_sub_vector_i;
      _sub_functions[i]->vector()->get_local(local_sub_vector_i);
      sub_block_vector->set_local(local_sub_vector_i);
    }
    else
      multiphenics_error("BlockFunction.cpp",
                         "apply to block function",
                         "Invalid mode");
  }
  if (mode == "from subfunctions")
    _block_vector->apply("insert");
}

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

#include <multiphenics/function/BlockFunction.h>
#include <multiphenics/la/BlockPETScSubVectorReadWrapper.h>
#include <multiphenics/la/BlockPETScSubVectorWrapper.h>

using namespace multiphenics;
using namespace multiphenics::function;

using dolfin::common::IndexMap;
using dolfin::fem::DofMap;
using dolfin::function::Function;
using dolfin::la::create_petsc_vector;
using dolfin::la::petsc_error;
using dolfin::la::VecReadWrapper;
using dolfin::la::VecWrapper;
using multiphenics::la::BlockPETScSubVectorReadWrapper;
using multiphenics::la::BlockPETScSubVectorWrapper;

//-----------------------------------------------------------------------------
BlockFunction::BlockFunction(std::shared_ptr<const BlockFunctionSpace> V)
  : _block_function_space(V), _sub_function_spaces(V->function_spaces)
{
  // Initialize block vector
  init_block_vector();
  
  // Initialize sub functions
  init_sub_functions();
}
//-----------------------------------------------------------------------------
BlockFunction::BlockFunction(std::shared_ptr<const BlockFunctionSpace> V,
                             std::vector<std::shared_ptr<Function>> sub_functions)
  : _block_function_space(V), _sub_function_spaces(V->function_spaces), _sub_functions(sub_functions)
{
  // Initialize block vector
  init_block_vector();
  
  // Apply from subfunctions
  apply("from subfunctions");
}
//-----------------------------------------------------------------------------
BlockFunction::BlockFunction(std::shared_ptr<const BlockFunctionSpace> V,
                             Vec x)
  : _block_function_space(V), _block_vector(x), _sub_function_spaces(V->function_spaces)
{
  // Initialize sub functions
  init_sub_functions();
  
  // Apply to subfunctions
  apply("to subfunctions");
}
//-----------------------------------------------------------------------------
BlockFunction::BlockFunction(std::shared_ptr<const BlockFunctionSpace> V,
                             Vec x,
                             std::vector<std::shared_ptr<Function>> sub_functions)
  : _block_function_space(V), _block_vector(x), _sub_function_spaces(V->function_spaces), _sub_functions(sub_functions)
{
  // Apply to subfunctions
  apply("to subfunctions");
}
//-----------------------------------------------------------------------------
std::shared_ptr<Function> BlockFunction::operator[](std::size_t i) const
{
  return _sub_functions[i];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const BlockFunctionSpace> BlockFunction::block_function_space() const
{
  assert(_block_function_space);
  return _block_function_space;
}
//-----------------------------------------------------------------------------
Vec BlockFunction::block_vector()
{
  assert(_block_vector);
  return _block_vector;
}
//-----------------------------------------------------------------------------
void BlockFunction::init_block_vector()
{
  // This method has been adapted from
  //    Function::init_vector
  
  // Get dof map
  assert(_block_function_space);
  assert(_block_function_space->block_dofmap);
  const auto dofmap = _block_function_space->block_dofmap;
  // Get index map
  std::shared_ptr<const IndexMap> index_map = dofmap->index_map;
  assert(index_map);
  // Initialize vector
  _block_vector = create_petsc_vector(*index_map);
  VecSet(_block_vector, 0.0);
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
  PetscErrorCode ierr;
  auto block_dofmap(_block_function_space->block_dofmap);
  unsigned int i(0);
  unsigned int i_max(_sub_functions.size());
  if (only >= 0) {
    i = static_cast<unsigned int>(only);
    i_max = i + 1;
  }
  for (; i < i_max; ++i)
  {
    Vec sub_vector_i = _sub_functions[i]->vector().vec();
    if (mode == "to subfunctions")
    {
      {
        BlockPETScSubVectorReadWrapper block_vector_i(_block_vector, i, _block_function_space->block_dofmap);
        VecWrapper sub_vector_i_(sub_vector_i);
        sub_vector_i_.x = block_vector_i.content;
      } // assignment of values in Vec occurs when VecWrapper gets out of scope. Afterwards, update ghosts:
      ierr = VecGhostUpdateBegin(sub_vector_i, INSERT_VALUES, SCATTER_FORWARD);
      if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostUpdateBegin");
      ierr = VecGhostUpdateEnd(sub_vector_i, INSERT_VALUES, SCATTER_FORWARD);
      if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostUpdateEnd");
    }
    else if (mode == "from subfunctions")
    {
      BlockPETScSubVectorWrapper block_vector_i(_block_vector, i, _block_function_space->block_dofmap, INSERT_VALUES);
      VecReadWrapper sub_vector_i_(sub_vector_i);
      block_vector_i.content = sub_vector_i_.x;
    }
    else
      throw std::runtime_error("Invalid mode when calling apply in block function");
  }
  if (mode == "from subfunctions")
  {
    ierr = VecGhostUpdateBegin(_block_vector, INSERT_VALUES, SCATTER_FORWARD);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostUpdateBegin");
    ierr = VecGhostUpdateEnd(_block_vector, INSERT_VALUES, SCATTER_FORWARD);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecGhostUpdateEnd");
  }
}

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

#include <dolfin/la/GenericLinearAlgebraFactory.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <multiphenics/la/GenericBlockLinearAlgebraFactory.h>
#include <multiphenics/fem/BlockDirichletBC.h>

using namespace dolfin;
using namespace dolfin::fem;
using namespace multiphenics;
using namespace multiphenics::fem;

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
void BlockDirichletBC::apply(GenericMatrix& A,
                             std::vector<std::vector<bool>> zero_off_block_diagonal) const
{
  GenericBlockLinearAlgebraFactory& block_linear_algebra_factory = dynamic_cast<GenericBlockLinearAlgebraFactory&>(A.factory());
  for (std::size_t I(0); I < _bcs.size(); ++I)
    for (std::size_t J(0); J < _bcs.size(); ++J)
    {
      std::shared_ptr<GenericMatrix> A_IJ = block_linear_algebra_factory.create_sub_matrix(A, I, J, BlockInsertMode::INSERT_VALUES);
      for (auto & bc_I: _bcs[I])
        if (I == J)
          bc_I->apply(*A_IJ);
        else if (zero_off_block_diagonal[I][J])
          bc_I->zero(*A_IJ);
    }
}
//-----------------------------------------------------------------------------
void BlockDirichletBC::apply(GenericVector& b) const
{
  GenericBlockLinearAlgebraFactory& block_linear_algebra_factory = dynamic_cast<GenericBlockLinearAlgebraFactory&>(b.factory());
  for (std::size_t I(0); I < _bcs.size(); ++I)
  {
    std::shared_ptr<GenericVector> b_I = block_linear_algebra_factory.create_sub_vector(b, I, BlockInsertMode::INSERT_VALUES);
    for (auto & bc_I: _bcs[I])
      bc_I->apply(*b_I);
  }
}
//-----------------------------------------------------------------------------
void BlockDirichletBC::apply(GenericMatrix& A,
                             GenericVector& b,
                             std::vector<std::vector<bool>> zero_off_block_diagonal) const
{
  GenericBlockLinearAlgebraFactory& block_linear_algebra_factory_A = dynamic_cast<GenericBlockLinearAlgebraFactory&>(A.factory());
  GenericBlockLinearAlgebraFactory& block_linear_algebra_factory_b = dynamic_cast<GenericBlockLinearAlgebraFactory&>(b.factory());
  for (std::size_t I(0); I < _bcs.size(); ++I)
  {
    std::shared_ptr<GenericVector> b_I = block_linear_algebra_factory_b.create_sub_vector(b, I, BlockInsertMode::INSERT_VALUES);
    for (std::size_t J(0); J < _bcs.size(); ++J)
    {
      std::shared_ptr<GenericMatrix> A_IJ = block_linear_algebra_factory_A.create_sub_matrix(A, I, J, BlockInsertMode::INSERT_VALUES);
      for (auto & bc_I: _bcs[I])
        if (I == J)
          bc_I->apply(*A_IJ, *b_I);
        else if (zero_off_block_diagonal[I][J])
          bc_I->zero(*A_IJ);
    }
  }
}
//-----------------------------------------------------------------------------
void BlockDirichletBC::apply(GenericVector& b,
                             const GenericVector& x) const
{
  GenericBlockLinearAlgebraFactory& block_linear_algebra_factory_b = dynamic_cast<GenericBlockLinearAlgebraFactory&>(b.factory());
  GenericBlockLinearAlgebraFactory& block_linear_algebra_factory_x = dynamic_cast<GenericBlockLinearAlgebraFactory&>(x.factory());
  for (std::size_t I(0); I < _bcs.size(); ++I)
  {
    std::shared_ptr<GenericVector> b_I = block_linear_algebra_factory_b.create_sub_vector(b, I, BlockInsertMode::INSERT_VALUES);
    std::shared_ptr<GenericVector> x_I = block_linear_algebra_factory_x.create_sub_vector(x, I, BlockInsertMode::INSERT_VALUES);
    for (auto & bc_I: _bcs[I])
      bc_I->apply(*b_I, *x_I);
  }
}
//-----------------------------------------------------------------------------
void BlockDirichletBC::apply(GenericMatrix& A,
                             GenericVector& b,
                             const GenericVector& x,
                             std::vector<std::vector<bool>> zero_off_block_diagonal) const
{
  GenericBlockLinearAlgebraFactory& block_linear_algebra_factory_A = dynamic_cast<GenericBlockLinearAlgebraFactory&>(A.factory());
  GenericBlockLinearAlgebraFactory& block_linear_algebra_factory_b = dynamic_cast<GenericBlockLinearAlgebraFactory&>(b.factory());
  GenericBlockLinearAlgebraFactory& block_linear_algebra_factory_x = dynamic_cast<GenericBlockLinearAlgebraFactory&>(x.factory());
  for (std::size_t I(0); I < _bcs.size(); ++I)
  {
    std::shared_ptr<GenericVector> b_I = block_linear_algebra_factory_b.create_sub_vector(b, I, BlockInsertMode::INSERT_VALUES);
    std::shared_ptr<GenericVector> x_I = block_linear_algebra_factory_x.create_sub_vector(x, I, BlockInsertMode::INSERT_VALUES);
    for (std::size_t J(0); J < _bcs.size(); ++J)
    {
      std::shared_ptr<GenericMatrix> A_IJ = block_linear_algebra_factory_A.create_sub_matrix(A, I, J, BlockInsertMode::INSERT_VALUES);
      for (auto & bc_I: _bcs[I])
        if (I == J)
          bc_I->apply(*A_IJ, *b_I, *x_I);
        else if (zero_off_block_diagonal[I][J])
          bc_I->zero(*A_IJ);
    }
  }
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
void BlockDirichletBC::zero(GenericMatrix& A,
                            std::vector<std::vector<bool>> zero_off_block_diagonal) const
{
  GenericBlockLinearAlgebraFactory& block_linear_algebra_factory = dynamic_cast<GenericBlockLinearAlgebraFactory&>(A.factory());
  for (std::size_t I(0); I < _bcs.size(); ++I)
    for (std::size_t J(0); J < _bcs.size(); ++J)
    {
      std::shared_ptr<GenericMatrix> A_IJ = block_linear_algebra_factory.create_sub_matrix(A, I, J, BlockInsertMode::INSERT_VALUES);
      for (auto & bc_I: _bcs[I])
        if (I == J || zero_off_block_diagonal[I][J])
          bc_I->zero(*A_IJ);
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

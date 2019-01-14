// Copyright (C) 2016-2019 by the multiphenics authors
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

#ifndef __BLOCK_FUNCTION_H
#define __BLOCK_FUNCTION_H

#include <dolfin/function/Function.h>
#include <multiphenics/function/BlockFunctionSpace.h>

namespace multiphenics
{

  class BlockFunction : public dolfin::Hierarchical<BlockFunction>
  {
  public:

    /// Create function on given block function space (shared data)
    ///
    /// *Arguments*
    ///     V (_BlockFunctionSpace_)
    ///         The block function space.
    explicit BlockFunction(std::shared_ptr<const BlockFunctionSpace> V);
    
    /// Create function on given block function space and with given subfunctions (shared data)
    ///
    /// *Arguments*
    ///     V (_BlockFunctionSpace_)
    ///         The block function space.
    BlockFunction(std::shared_ptr<const BlockFunctionSpace> V,
                  std::vector<std::shared_ptr<dolfin::Function>> sub_functions);

    /// Create function on given function space with a given vector
    /// (shared data)
    ///
    /// *Warning: This constructor is intended for internal library use only*
    ///
    /// *Arguments*
    ///     V (_BlockFunctionSpace_)
    ///         The block function space.
    ///     x (_GenericVector_)
    ///         The block vector.
    BlockFunction(std::shared_ptr<const BlockFunctionSpace> V,
                  std::shared_ptr<dolfin::GenericVector> x);
                  
    /// Create function on given function space with a given vector
    /// and given subfunctions (shared data)
    ///
    /// *Warning: This constructor is intended for internal library use only*
    ///
    /// *Arguments*
    ///     V (_BlockFunctionSpace_)
    ///         The block function space.
    ///     x (_GenericVector_)
    ///         The block vector.
    BlockFunction(std::shared_ptr<const BlockFunctionSpace> V,
                  std::shared_ptr<dolfin::GenericVector> x,
                  std::vector<std::shared_ptr<dolfin::Function>> sub_functions);

    /// Copy constructor
    ///
    /// *Arguments*
    ///     v (_BlockFunction_)
    ///         The object to be copied.
    BlockFunction(const BlockFunction& v);

    /// Destructor
    virtual ~BlockFunction();

    /// Assignment from function
    ///
    /// *Arguments*
    ///     v (_BlockFunction_)
    ///         Another function.
    const BlockFunction& operator= (const BlockFunction& v);

    /// Extract subfunction
    ///
    /// *Arguments*
    ///     i (std::size_t)
    ///         Index of subfunction.
    /// *Returns*
    ///     _Function_
    ///         The subfunction.
    std::shared_ptr<dolfin::Function> operator[] (std::size_t i) const;

    /// Return shared pointer to function space
    ///
    /// *Returns*
    ///     _FunctionSpace_
    ///         Return the shared pointer.
    virtual std::shared_ptr<const BlockFunctionSpace> block_function_space() const;

    /// Return vector of expansion coefficients (non-const version)
    ///
    /// *Returns*
    ///     _GenericVector_
    ///         The vector of expansion coefficients.
    std::shared_ptr<dolfin::GenericVector> block_vector();

    /// Return vector of expansion coefficients (const version)
    ///
    /// *Returns*
    ///     _GenericVector_
    ///         The vector of expansion coefficients (const).
    std::shared_ptr<const dolfin::GenericVector> block_vector() const;
    
    /// Sync block vector and sub functions
    void apply(std::string mode, int only = -1);
    
  private:

    // Initialize vector
    void init_block_vector();
    
    // Initialize sub functions
    void init_sub_functions();

    // The function space
    std::shared_ptr<const BlockFunctionSpace> _block_function_space;

    // The vector of expansion coefficients (local)
    std::shared_ptr<dolfin::GenericVector> _block_vector;
    
    // Sub functions
    std::vector<std::shared_ptr<const dolfin::FunctionSpace>> _sub_function_spaces;
    std::vector<std::shared_ptr<dolfin::Function>> _sub_functions;

  };

}

#endif

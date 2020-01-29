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

#ifndef __BLOCK_FUNCTION_H
#define __BLOCK_FUNCTION_H

#include <petscvec.h>
#include <dolfinx/function/Function.h>
#include <multiphenics/function/BlockFunctionSpace.h>

namespace multiphenics
{
  namespace function
  {
    class BlockFunction
    {
    public:

      /// Create function on given block function space
      /// @param[in] V The block function space
      explicit BlockFunction(std::shared_ptr<const BlockFunctionSpace> V);
      
      /// Create function on given block function space and with given subfunctions
      /// @param[in] V The block function space
      /// @param[in] sub_functions Existing subfunctions
      BlockFunction(std::shared_ptr<const BlockFunctionSpace> V,
                    std::vector<std::shared_ptr<dolfinx::function::Function>> sub_functions);

      /// Create function on given function space with a given vector
      ///
      /// *Warning: This constructor is intended for internal library use only*
      ///
      /// @param[in] V The block function space
      /// @param[in] x The block vector
      BlockFunction(std::shared_ptr<const BlockFunctionSpace> V,
                    Vec x);
                    
      /// Create function on given function space with a given vector
      /// and given subfunctions
      ///
      /// *Warning: This constructor is intended for internal library use only*
      ///
      /// @param[in] V The block function space
      /// @param[in] x The block vector
      /// @param[in] sub_functions Existing subfunctions
      BlockFunction(std::shared_ptr<const BlockFunctionSpace> V,
                    Vec x,
                    std::vector<std::shared_ptr<dolfinx::function::Function>> sub_functions);

      // Copy constructor
      BlockFunction(const BlockFunction& v) = delete;

      /// Move constructor
      BlockFunction(BlockFunction&& v) = default;

      /// Destructor
      virtual ~BlockFunction() = default;
      
      /// Move assignment
      BlockFunction& operator=(BlockFunction&& v) = default;

      // Assignment
      BlockFunction& operator=(const BlockFunction& v) = delete;

      /// Extract subfunction
      /// @param[in] i Index of subfunction
      /// @return The subfunction
      std::shared_ptr<dolfinx::function::Function> operator[](std::size_t i) const;

      /// Return shared pointer to block function space
      /// @returns The block function space
      std::shared_ptr<const BlockFunctionSpace> block_function_space() const;

      /// Return vector of expansion coefficients (non-const version)
      /// @returns The vector of expansion coefficients.
      Vec block_vector();
            
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
      Vec _block_vector;
      
      // Sub functions
      std::vector<std::shared_ptr<const dolfinx::function::FunctionSpace>> _sub_function_spaces;
      std::vector<std::shared_ptr<dolfinx::function::Function>> _sub_functions;

    };
  }
}

#endif

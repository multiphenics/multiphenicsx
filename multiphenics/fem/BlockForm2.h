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

#ifndef __BLOCK_FORM_2_H
#define __BLOCK_FORM_2_H

#include <vector>
#include <dolfin/fem/Form.h>
#include <multiphenics/function/BlockFunctionSpace.h>

namespace multiphenics
{
  namespace fem
  {
    class BlockForm2
    {
    public:
      /// Create form (shared data)
      ///
      /// @param[in] forms (std::vector<std::vector<_Form_>>)
      ///         Matrix of forms.
      /// @param[in] function_spaces (std::vector<_multiphenics::function::BlockFunctionSpace_>)
      ///         Vector of function spaces, of size 2.
      BlockForm2(std::vector<std::vector<std::shared_ptr<const dolfin::fem::Form>>> forms,
                 std::vector<std::shared_ptr<const multiphenics::function::BlockFunctionSpace>> block_function_spaces);
           
      /// Destructor
      ~BlockForm2();
      
      /// Extract common mesh from form
      ///
      /// @return Mesh
      ///         Shared pointer to the mesh.
      std::shared_ptr<const dolfin::mesh::Mesh> mesh() const;

      /// Return function spaces for arguments
      ///
      /// @return    std::vector<_FunctionSpace_>
      ///         Vector of function space shared pointers.
      std::vector<std::shared_ptr<const multiphenics::function::BlockFunctionSpace>> block_function_spaces() const;
            
      unsigned int block_size(unsigned int d) const;
      
      const dolfin::fem::Form & operator()(std::size_t i, std::size_t j) const;
      
    protected:
      // Block forms
      std::vector<std::vector<std::shared_ptr<const dolfin::fem::Form>>> _forms;
      // Block function spaces (one for each argument)
      std::vector<std::shared_ptr<const multiphenics::function::BlockFunctionSpace>> _block_function_spaces;
      // Number of block forms
      std::vector<unsigned int> _block_size;
    };
  }
}

#endif

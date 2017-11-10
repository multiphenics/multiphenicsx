// Copyright (C) 2016-2017 by the multiphenics authors
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
#include <multiphenics/fem/BlockFormBase.h>

namespace multiphenics
{

  class BlockForm2 : public BlockFormBase
  {
  public:
    /// Create form (shared data)
    ///
    /// @param[in] forms (std::vector<std::vector<_Form_>>)
    ///         Matrix of forms.
    /// @param[in] function_spaces (std::vector<_BlockFunctionSpace_>)
    ///         Vector of function spaces, of size 2.
    BlockForm2(std::vector<std::vector<std::shared_ptr<const dolfin::Form>>> forms,
               std::vector<std::shared_ptr<const BlockFunctionSpace>> block_function_spaces);
         
    /// Destructor
    virtual ~BlockForm2();

    /// Return rank of form (bilinear form = 2)
    ///
    /// @return std::size_t
    ///         The rank of the form.
    virtual std::size_t rank() const;
    
    virtual unsigned int block_size(unsigned int d) const;
    
    const dolfin::Form & operator()(std::size_t i, std::size_t j) const;
    
  protected:
  
    virtual bool has_cell_integrals() const;
    virtual bool has_interior_facet_integrals() const;
    virtual bool has_exterior_facet_integrals() const;
    virtual bool has_vertex_integrals() const;

    // Block forms
    std::vector<std::vector<std::shared_ptr<const dolfin::Form>>> _forms;
    
    std::vector<unsigned int> _block_size;
  };
  
}

#endif

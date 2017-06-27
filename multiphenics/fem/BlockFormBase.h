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

#ifndef __BLOCK_FORM_BASE_H
#define __BLOCK_FORM_BASE_H

#include <dolfin/common/Hierarchical.h>
#include <dolfin/mesh/Mesh.h>
#include <multiphenics/function/BlockFunctionSpace.h>

namespace dolfin
{

  class BlockFormBase : public Hierarchical<BlockFormBase>
  {
  public:
    /// Create form (shared data)
    ///
    /// @param[in] function_spaces (std::vector<_BlockFunctionSpace_>)
    ///         Vector of function spaces.
    BlockFormBase(std::vector<std::shared_ptr<const BlockFunctionSpace>> block_function_spaces);
         
    /// Destructor
    virtual ~BlockFormBase();
    
    /// Return rank of form (bilinear form = 2, linear form = 1)
    ///
    /// @return std::size_t
    ///         The rank of the form.
    virtual std::size_t rank() const = 0;
    
    /// Extract common mesh from form
    ///
    /// @return Mesh
    ///         Shared pointer to the mesh.
    std::shared_ptr<const Mesh> mesh() const;

    /// Return function spaces for arguments
    ///
    /// @return    std::vector<_FunctionSpace_>
    ///         Vector of function space shared pointers.
    std::vector<std::shared_ptr<const BlockFunctionSpace>> block_function_spaces() const;
    
    virtual unsigned int block_size(unsigned int d) const = 0;
    
  protected:
  
    virtual bool has_cell_integrals() const = 0;
    virtual bool has_interior_facet_integrals() const = 0;
    virtual bool has_exterior_facet_integrals() const = 0;
    virtual bool has_vertex_integrals() const = 0;
    friend class BlockAssemblerBase;

    // Block function spaces (one for each argument)
    std::vector<std::shared_ptr<const BlockFunctionSpace>> _block_function_spaces;
  };
  
}

#endif

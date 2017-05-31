// Copyright (C) 2016-2017 by the block_ext authors
//
// This file is part of block_ext.
//
// block_ext is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// block_ext is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with block_ext. If not, see <http://www.gnu.org/licenses/>.
//

#ifndef __GENERIC_BLOCK_MATRIX_H
#define __GENERIC_BLOCK_MATRIX_H

#include <dolfin/la/GenericMatrix.h>
#include <block/fem/BlockDofMap.h>
#include <block/la/BlockInsertMode.h>

namespace dolfin
{
  class GenericBlockMatrix
  {
  public:
    virtual ~GenericBlockMatrix();
    
    //--- Special block functions ---
    
    /// Attach BlockDofMap for submatrix creation
    virtual void attach_block_dof_map(std::shared_ptr<const BlockDofMap> block_dof_map_0, std::shared_ptr<const BlockDofMap> block_dof_map_1) = 0;
    
    /// Get BlockDofMap for submatrix creation
    virtual std::shared_ptr<const BlockDofMap> get_block_dof_map(std::size_t d) const = 0;
    
    /// Check if BlockDofMap for submatrix creation has been attached
    virtual bool has_block_dof_map(std::size_t d) const = 0;
    
    /// Block access
    virtual std::shared_ptr<GenericMatrix> operator()(std::size_t block_i, std::size_t block_j, BlockInsertMode insert_mode) const = 0;
  };
  
}

#endif

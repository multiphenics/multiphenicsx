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

#ifndef __GENERIC_BLOCK_VECTOR_H
#define __GENERIC_BLOCK_VECTOR_H

#include <dolfin/la/GenericVector.h>
#include <multiphenics/fem/BlockDofMap.h>
#include <multiphenics/la/BlockInsertMode.h>

namespace multiphenics
{
  namespace la
  {
    class GenericBlockVector
    {
    public:
      virtual ~GenericBlockVector();
      
      //--- Special block functions ---
      
      /// Attach BlockDofMap for subvector creation
      virtual void attach_block_dof_map(std::shared_ptr<const BlockDofMap> block_dof_map) = 0;
      
      /// Get BlockDofMap for subvector creation
      virtual std::shared_ptr<const BlockDofMap> get_block_dof_map() const = 0;
      
      /// Check if BlockDofMap for subvector creation has been attached
      virtual bool has_block_dof_map() const = 0;
      
      /// Block access
      virtual std::shared_ptr<dolfin::GenericVector> operator()(std::size_t block_i, BlockInsertMode insert_mode) const = 0;
    };
  }
}

#endif

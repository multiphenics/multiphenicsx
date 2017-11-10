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

#ifndef __BLOCK_ASSEMBLER_H
#define __BLOCK_ASSEMBLER_H

#include <multiphenics/fem/BlockAssemblerBase.h>

namespace multiphenics
{
  /// Class for block assembly
  class BlockAssembler : public BlockAssemblerBase
  {
  public:

    /// Constructor
    BlockAssembler();

    /// Assemble block tensor from given block form
    ///
    /// @param[out] A (GenericTensor)
    ///         The block tensor to assemble.
    /// @param[in]  a (BlockFormBase&)
    ///         The block form to assemble the tensor from.
    void assemble(dolfin::GenericTensor& A, const BlockFormBase& a);
    
  private:
    /// Assemble subtensor from given form
    ///
    /// @param[out] A (GenericTensor)
    ///         The tensor to assemble.
    /// @param[in]  a (Form&)
    ///         The form to assemble the tensor from.
    void sub_assemble(dolfin::GenericTensor& A, const dolfin::Form& a, dolfin::Assembler& assembler);

  };

}

#endif

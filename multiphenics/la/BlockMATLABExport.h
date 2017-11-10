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

#ifndef __BLOCK_MATLAB_EXPORT_H
#define __BLOCK_MATLAB_EXPORT_H

#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>

namespace multiphenics
{
  class BlockMATLABExport
  {
  public:
    /// Export matrix
    static void export_(const dolfin::PETScMatrix & A, std::string A_filename);
    
    /// Export vector
    static void export_(const dolfin::PETScVector & b, std::string b_filename);
  };
  
}

#endif

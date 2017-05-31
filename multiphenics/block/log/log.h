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

#ifndef __BLOCK_LOG_H
#define __BLOCK_LOG_H

#include <dolfin/log/log.h>

namespace dolfin
{

  /// Print error message. Prefer this to the above generic error message.
  ///
  /// *Arguments*
  ///     location (std::string)
  ///         Name of the file from which the error message was generated.
  ///     task (std::string)
  ///         Name of the task that failed.
  ///         Note that this string should begin with lowercase.
  ///         Note that this string should not be punctuated.
  ///     reason (std::string)
  ///         A format string explaining the reason for the failure.
  ///         Note that this string should begin with uppercase.
  ///         Note that this string should not be punctuated.
  ///
  void _block_error(std::string location,
                    std::string task,
                    std::string reason);
                       
}

#define block_error(location, task, reason) \
  do { \
    dolfin::_block_error(location, task, reason); \
    /* Avoid warnings related to return types by explicitly throwing an error, */ \
    /* even though this line will never be reached because execution terminates at dolfin_error */ \
    throw std::runtime_error("block_ext encountered an error."); \
  } while (false)

#endif

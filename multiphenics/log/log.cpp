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

#include <unistd.h>  // needed for getpid()
#include <cstdio> // std::remove
#include <cstdlib> // std::system
#include <sys/types.h> // pid_t
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <multiphenics/log/log.h>

#define USE_GDB 1

using namespace dolfin;
using namespace multiphenics;

//-----------------------------------------------------------------------------
void multiphenics::_multiphenics_error(std::string location,
                                 std::string task,
                                 std::string reason)
{
  #ifdef USE_GDB
  // Run gdb and print backtrace. Adapted from libmesh/src/base/print_trace.C
  {
    char temp_file[] = "temp_print_trace.XXXXXX";
    int fd = mkstemp(temp_file);

    if (fd > 0)
    {
      std::cerr << "multiphenics encountered an error. Running gdb to provide a backtrace." << std::endl;
      
      pid_t this_pid = getpid();

      std::ostringstream command;
      command << "gdb -p " << this_pid
        << " -batch -ex bt -ex detach "
        << " 2>/dev/null"
        << " 1>" << temp_file;
        
      int exit_status = std::system(command.str().c_str());
      
      std::ifstream fin(temp_file);
      if (fin && (fin.peek() != std::ifstream::traits_type::eof()) && (exit_status == 0))
        std::cerr << fin.rdbuf();
      else
        std::cerr << "Failed to obtain a gdb backtrace." << std::endl;
    }

    std::remove(temp_file);
  }
  #endif
  
  // Call standard dolfin error
  dolfin_error(location, task, reason);
}
//-----------------------------------------------------------------------------

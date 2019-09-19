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

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace multiphenics_wrappers
{
  void function(py::module& m);
  void fem(py::module& m);
  void la(py::module& m);
}

PYBIND11_MODULE(SIGNATURE, m)
{
  // Create module for C++ wrappers
  m.doc() = "multiphenics Python interface";
  
  // Create function submodule
  py::module function = m.def_submodule("function", "Function module");
  multiphenics_wrappers::function(function);

  // Create fem submodule
  py::module fem = m.def_submodule("fem", "FEM module");
  multiphenics_wrappers::fem(fem);

  // Create la submodule
  py::module la = m.def_submodule("la", "Linear algebra module");
  multiphenics_wrappers::la(la);
}

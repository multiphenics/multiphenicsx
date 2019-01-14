// Copyright (C) 2016-2019 by the multiphenics authors
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
  void io(py::module& m);
  void la(py::module& m);
  void log(py::module& m);
  void mesh(py::module& m);
  void nls(py::module& m);
}

PYBIND11_MODULE(SIGNATURE, m)
{
  // Create module for C++ wrappers
  m.doc() = "multiphenics Python interface";
  
  // Create log submodule [log]
  py::module log = m.def_submodule("log", "Logging module");
  multiphenics_wrappers::log(log);

  // Create function submodule [function]
  py::module function = m.def_submodule("function", "Function module");
  multiphenics_wrappers::function(function);

  // Create mesh submodule [mesh]
  py::module mesh = m.def_submodule("mesh", "Mesh library module");
  multiphenics_wrappers::mesh(mesh);

  // Create fem submodule [fem]
  py::module fem = m.def_submodule("fem", "FEM module");
  multiphenics_wrappers::fem(fem);

  // Create io submodule
  py::module io = m.def_submodule("io", "I/O module");
  multiphenics_wrappers::io(io);

  // Create la submodule
  py::module la = m.def_submodule("la", "Linear algebra module");
  multiphenics_wrappers::la(la);

  // Create nls submodule
  py::module nls = m.def_submodule("nls", "Nonlinear solver module");
  multiphenics_wrappers::nls(nls);
}

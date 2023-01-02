// Copyright (C) 2016-2023 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace multiphenicsx_wrappers
{
  void fem(py::module& m);
  void la(py::module& m);
}

PYBIND11_MODULE(SIGNATURE, m)
{
  // Create module for C++ wrappers
  m.doc() = "multiphenicsx Python interface";

  // Create fem submodule
  py::module fem = m.def_submodule("fem", "FEM module");
  multiphenicsx_wrappers::fem(fem);

  // Create la submodule
  py::module la = m.def_submodule("la", "Linear algebra module");
  multiphenicsx_wrappers::la(la);
}

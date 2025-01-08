// Copyright (C) 2016-2025 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace multiphenicsx_wrappers
{
void fem(nb::module_& m);
void la(nb::module_& m);
} // namespace multiphenicsx_wrappers

NB_MODULE(multiphenicsx_cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "multiphenicsx Python interface";

#ifdef NDEBUG
  nanobind::set_leak_warnings(false);
#endif

  // Create fem submodule
  nb::module_ fem = m.def_submodule("fem", "FEM module");
  multiphenicsx_wrappers::fem(fem);

  // Create la submodule
  nb::module_ la = m.def_submodule("la", "Linear algebra module");
  multiphenicsx_wrappers::la(la);
}

// Copyright (C) 2016-2025 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <array>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx_wrappers/caster_petsc.h>
#include <memory>
#include <multiphenicsx/la/petsc.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <petsc4py/petsc4py.h>
#include <petscis.h>
#include <vector>

namespace nb = nanobind;

namespace nanobind::detail
{
PETSC_CASTER_MACRO(IS, IS, is);
} // namespace nanobind::detail

namespace multiphenicsx_wrappers
{
void la_petsc_module(nb::module_& m)
{
  import_petsc4py();

  nb::class_<multiphenicsx::la::petsc::MatSubMatrixWrapper>(
      m, "MatSubMatrixWrapper")
      .def(nb::init<Mat, std::array<IS, 2>>(), nb::arg("A"),
           nb::arg("index_sets"))
      .def(nb::init<
               Mat, std::array<IS, 2>, std::array<IS, 2>,
               std::array<std::unordered_map<std::int32_t, std::int32_t>, 2>,
               std::array<int, 2>>(),
           nb::arg("A"), nb::arg("unrestricted_index_sets"),
           nb::arg("restricted_index_sets"),
           nb::arg("unrestricted_to_restricted"),
           nb::arg("unrestricted_to_restricted_bs"))
      .def("restore", &multiphenicsx::la::petsc::MatSubMatrixWrapper::restore)
      .def("mat",
           [](const multiphenicsx::la::petsc::MatSubMatrixWrapper& self)
           {
             Mat mat = self.mat();
             PyObject* obj = PyPetscMat_New(mat);
             return nb::borrow(obj);
           });

  nb::class_<multiphenicsx::la::petsc::VecSubVectorReadWrapper>(
      m, "VecSubVectorReadWrapper")
      .def(nb::init<Vec, IS, bool>(), nb::arg("x"), nb::arg("index_set"),
           nb::arg("ghosted") = true)
      .def(nb::init<Vec, IS, IS,
                    const std::unordered_map<std::int32_t, std::int32_t>&, int,
                    bool>(),
           nb::arg("x"), nb::arg("unrestricted_index_set"),
           nb::arg("restricted_index_set"),
           nb::arg("unrestricted_to_restricted"),
           nb::arg("unrestricted_to_restricted_bs"), nb::arg("ghosted") = true)
      .def_prop_ro(
          "content",
          [](multiphenicsx::la::petsc::VecSubVectorReadWrapper& self)
          {
            std::vector<PetscScalar>& array = self.mutable_content();
            return nb::ndarray<PetscScalar, nb::numpy>(
                array.data(), {array.size()}, nb::handle());
          },
          nb::rv_policy::reference_internal);

  nb::class_<multiphenicsx::la::petsc::VecSubVectorWrapper,
             multiphenicsx::la::petsc::VecSubVectorReadWrapper>(
      m, "VecSubVectorWrapper")
      .def(nb::init<Vec, IS, bool>(), nb::arg("x"), nb::arg("index_set"),
           nb::arg("ghosted") = true)
      .def(nb::init<Vec, IS, IS,
                    const std::unordered_map<std::int32_t, std::int32_t>&, int,
                    bool>(),
           nb::arg("x"), nb::arg("unrestricted_index_set"),
           nb::arg("restricted_index_set"),
           nb::arg("unrestricted_to_restricted"),
           nb::arg("unrestricted_to_restricted_bs"), nb::arg("ghosted") = true)
      .def("restore", &multiphenicsx::la::petsc::VecSubVectorWrapper::restore);

  nb::enum_<multiphenicsx::la::petsc::GhostBlockLayout>(m, "GhostBlockLayout")
      .value("intertwined",
             multiphenicsx::la::petsc::GhostBlockLayout::intertwined)
      .value("trailing", multiphenicsx::la::petsc::GhostBlockLayout::trailing);

  m.def(
      "create_index_sets",
      [](const std::vector<std::pair<const dolfinx::common::IndexMap*, int>>&
             maps,
         const std::vector<int> is_bs, bool ghosted,
         multiphenicsx::la::petsc::GhostBlockLayout ghost_block_layout)
      {
        std::vector<std::pair<
            std::reference_wrapper<const dolfinx::common::IndexMap>, int>>
            _maps;
        for (auto m : maps)
          _maps.push_back({*m.first, m.second});
        std::vector<IS> index_sets
            = multiphenicsx::la::petsc::create_index_sets(_maps, is_bs, ghosted,
                                                          ghost_block_layout);

        std::vector<nb::object> py_index_sets;
        for (auto is : index_sets)
        {
          PyObject* obj = PyPetscIS_New(is);
          PetscObjectDereference((PetscObject)is);
          py_index_sets.push_back(nb::steal(obj));
        }
        return py_index_sets;
      },
      nb::arg("maps"), nb::arg("is_bs"), nb::arg("ghosted") = true,
      nb::arg("ghost_block_layout")
      = multiphenicsx::la::petsc::GhostBlockLayout::intertwined);
}

void la(nb::module_& m)
{
  nb::module_ petsc_mod
      = m.def_submodule("petsc", "PETSc-specific linear algebra");
  la_petsc_module(petsc_mod);
}
} // namespace multiphenicsx_wrappers

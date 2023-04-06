// Copyright (C) 2016-2023 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <array>
#include <caster_petsc.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <multiphenicsx/fem/DofMapRestriction.h>
#include <multiphenicsx/fem/petsc.h>
#include <multiphenicsx/fem/utils.h>
#include <petsc4py/petsc4py.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <set>
#include <string>
#include <vector>

namespace py = pybind11;

namespace multiphenicsx_wrappers
{
void fem_petsc_module(py::module& m)
{
  // Create PETSc matrices
  m.def("create_matrix",
        [](const dolfinx::fem::Form<PetscScalar, double>& a,
           std::vector<std::reference_wrapper<const dolfinx::common::IndexMap>> index_maps_,
           const std::array<int, 2> index_maps_bs,
           std::array<const dolfinx::graph::AdjacencyList<std::int32_t>*, 2> dofmaps,
           const std::string& matrix_type) {
          // Due to pybind11#2123 the argument index_maps_ is of type
          //   std::vector<std::reference_wrapper<const dolfinx::common::IndexMap>> index_maps
          // rather than
          //   std::array<std::reference_wrapper<const dolfinx::common::IndexMap>, 2> index_maps
          // as in the C++ backend. Convert here std::vector to a std::array.
          assert(index_maps_.size() == 2);
          std::array<std::reference_wrapper<const dolfinx::common::IndexMap>, 2> index_maps{{
            index_maps_[0], index_maps_[1]
          }};
          return multiphenicsx::fem::petsc::create_matrix(a, index_maps, index_maps_bs, dofmaps, matrix_type);
        },
        py::return_value_policy::take_ownership,
        py::arg("a"), py::arg("index_maps"), py::arg("index_maps_bs"), py::arg("dofmaps"),
        py::arg("matrix_type") = std::string(),
        "Create a PETSc Mat for bilinear form.");
  m.def("create_matrix_block",
        &multiphenicsx::fem::petsc::create_matrix_block<double>,
        py::return_value_policy::take_ownership,
        py::arg("a"), py::arg("index_maps"), py::arg("index_maps_bs"), py::arg("dofmaps"),
        py::arg("matrix_type") = std::string(),
        "Create monolithic sparse matrix for stacked bilinear forms.");
  m.def("create_matrix_nest",
        &multiphenicsx::fem::petsc::create_matrix_nest<double>,
        py::return_value_policy::take_ownership,
        py::arg("mesh"), py::arg("index_maps"), py::arg("index_maps_bs"), py::arg("dofmaps"),
        py::arg("matrix_types") = std::vector<std::vector<std::string>>(),
        "Create nested sparse matrix for bilinear forms.");
}

void fem(py::module& m)
{
  py::module petsc_mod
      = m.def_submodule("petsc", "PETSc-specific finite element module");
  fem_petsc_module(petsc_mod);

  // multiphenicsx::fem::DofMapRestriction
  py::class_<multiphenicsx::fem::DofMapRestriction, std::shared_ptr<multiphenicsx::fem::DofMapRestriction>>(
      m, "DofMapRestriction", "DofMapRestriction object")
      .def(py::init<std::shared_ptr<const dolfinx::fem::DofMap>,
                    const std::vector<std::int32_t>&>(),
           py::arg("dofmap"), py::arg("restriction"))
      .def("cell_dofs",
           [](const multiphenicsx::fem::DofMapRestriction& self, int cell) {
             auto dofs = self.cell_dofs(cell);
             return py::array_t<std::int32_t>(dofs.size(), dofs.data(),
                                              py::cast(self));
           })
      .def_property_readonly("dofmap", &multiphenicsx::fem::DofMapRestriction::dofmap)
      .def_property_readonly("unrestricted_to_restricted",
                             &multiphenicsx::fem::DofMapRestriction::unrestricted_to_restricted)
      .def_property_readonly("restricted_to_unrestricted",
                             &multiphenicsx::fem::DofMapRestriction::restricted_to_unrestricted)
      .def("list", &multiphenicsx::fem::DofMapRestriction::list)
      .def_readonly("index_map", &multiphenicsx::fem::DofMapRestriction::index_map)
      .def_property_readonly("index_map_bs",
                             &multiphenicsx::fem::DofMapRestriction::index_map_bs);

}
} // namespace multiphenics_wrappers

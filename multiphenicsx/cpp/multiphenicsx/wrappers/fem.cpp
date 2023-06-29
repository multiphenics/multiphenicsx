// Copyright (C) 2016-2023 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <array>
#include <caster_petsc.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/Form.h>
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

namespace
{
  template <class T>
  std::array<T, 2> convert_vector_to_array(const std::vector<T>& input)
  {
    // Workaround for pybind11#2123.
    assert(input.size() == 2);
    std::array<T, 2> output {{input[0], input[1]}};
    return output;
  }

  template <class T>
  std::span<const T> convert_pyarray_to_span(const py::array_t<T, py::array::c_style>& input)
  {
    return std::span(input.data(), input.size());
  }

  template <class T>
  std::vector<std::span<const T>> convert_pyarray_to_span(const std::vector<py::array_t<T, py::array::c_style>>& input)
  {
    std::vector<std::span<const T>> output;
    output.reserve(input.size());
    for (auto& input_: input)
      output.push_back(convert_pyarray_to_span(input_));
    return output;
  }

  template <class T>
  std::array<std::span<const T>, 2> convert_pyarray_to_span(
    const std::array<py::array_t<T, py::array::c_style>, 2>& input)
  {
    return {{convert_pyarray_to_span(input[0]), convert_pyarray_to_span(input[1])}};
  }

  template <class T>
  std::array<std::vector<std::span<const T>>, 2> convert_pyarray_to_span(
    const std::array<std::vector<py::array_t<T, py::array::c_style>>, 2>& input)
  {
    return {{convert_pyarray_to_span(input[0]), convert_pyarray_to_span(input[1])}};
  }
}

namespace multiphenicsx_wrappers
{
void fem_petsc_module(py::module& m)
{
  // Create PETSc matrices
  m.def("create_matrix",
        [](const dolfinx::fem::Form<PetscScalar, PetscReal>& a,
           std::vector<std::reference_wrapper<const dolfinx::common::IndexMap>> index_maps_,
           const std::array<int, 2> index_maps_bs,
           std::array<py::array_t<std::int32_t, py::array::c_style>, 2> dofmaps_list_,
           std::array<py::array_t<std::size_t, py::array::c_style>, 2> dofmaps_bounds_,
           const std::string& matrix_type) {
          auto index_maps = convert_vector_to_array(index_maps_);
          auto dofmaps_list = convert_pyarray_to_span(dofmaps_list_);
          auto dofmaps_bounds = convert_pyarray_to_span(dofmaps_bounds_);
          return multiphenicsx::fem::petsc::create_matrix(
            a, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds, matrix_type);
        },
        py::return_value_policy::take_ownership,
        py::arg("a"), py::arg("index_maps"), py::arg("index_maps_bs"), py::arg("dofmaps_list"),
        py::arg("dofmaps_bounds"), py::arg("matrix_type") = std::string(),
        "Create a PETSc Mat for bilinear form.");
  m.def("create_matrix_block",
        [](const std::vector<std::vector<const dolfinx::fem::Form<PetscScalar, PetscReal>*>>& a,
           std::array<std::vector<std::reference_wrapper<const dolfinx::common::IndexMap>>, 2> index_maps,
           const std::array<std::vector<int>, 2> index_maps_bs,
           std::array<std::vector<py::array_t<std::int32_t, py::array::c_style>>, 2> dofmaps_list_,
           std::array<std::vector<py::array_t<std::size_t, py::array::c_style>>, 2> dofmaps_bounds_,
           const std::string& matrix_type) {
          auto dofmaps_list = convert_pyarray_to_span(dofmaps_list_);
          auto dofmaps_bounds = convert_pyarray_to_span(dofmaps_bounds_);
          return multiphenicsx::fem::petsc::create_matrix_block(
            a, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds, matrix_type);
        },
        py::return_value_policy::take_ownership,
        py::arg("a"), py::arg("index_maps"), py::arg("index_maps_bs"), py::arg("dofmaps_list"),
        py::arg("dofmaps_bounds"), py::arg("matrix_type") = std::string(),
        "Create monolithic sparse matrix for stacked bilinear forms.");
  m.def("create_matrix_nest",
        [](const std::vector<std::vector<const dolfinx::fem::Form<PetscScalar, PetscReal>*>>& a,
           std::array<std::vector<std::reference_wrapper<const dolfinx::common::IndexMap>>, 2> index_maps,
           const std::array<std::vector<int>, 2> index_maps_bs,
           std::array<std::vector<py::array_t<std::int32_t, py::array::c_style>>, 2> dofmaps_list_,
           std::array<std::vector<py::array_t<std::size_t, py::array::c_style>>, 2> dofmaps_bounds_,
           const std::vector<std::vector<std::string>>& matrix_types) {
          auto dofmaps_list = convert_pyarray_to_span(dofmaps_list_);
          auto dofmaps_bounds = convert_pyarray_to_span(dofmaps_bounds_);
          return multiphenicsx::fem::petsc::create_matrix_nest(
            a, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds, matrix_types);
        },
        py::return_value_policy::take_ownership,
        py::arg("a"), py::arg("index_maps"), py::arg("index_maps_bs"), py::arg("dofmaps_list"),
        py::arg("dofmaps_bounds"), py::arg("matrix_types") = std::vector<std::vector<std::string>>(),
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
      .def("map",
           [](const multiphenicsx::fem::DofMapRestriction& self) {
             auto map = self.map();
             return std::make_pair(
                py::array_t<std::int32_t>(map.first.size(), map.first.data(), py::cast(self)),
                py::array_t<std::size_t>(map.second.size(), map.second.data(), py::cast(self)));
           },
           py::return_value_policy::reference_internal)
      .def_readonly("index_map", &multiphenicsx::fem::DofMapRestriction::index_map)
      .def_property_readonly("index_map_bs",
                             &multiphenicsx::fem::DofMapRestriction::index_map_bs);

}
} // namespace multiphenics_wrappers

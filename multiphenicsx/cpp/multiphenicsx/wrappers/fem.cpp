// Copyright (C) 2016-2021 by the multiphenicsx authors
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

namespace
{
  template<class T>
  std::set<T> convert_vector_to_set(const std::vector<T>& input)
  {
    // TODO Remove this when pybind11#2122 is fixed.
    std::set<T> output(input.begin(), input.end());
    return output;
  }

  template<class T>
  std::vector<std::vector<std::set<T>>> convert_vector_to_set(
    const std::vector<std::vector<std::vector<T>>>& input)
  {
    // TODO Remove this when pybind11#2122 is fixed.
    std::size_t rows = input.size();
    std::size_t cols = input[0].size();
    assert(std::all_of(input.begin(), input.end(),
      [&cols](const std::vector<std::vector<T>>& input_){
      return cols == input_.size();}));
    std::vector<std::vector<std::set<T>>> output;
    output.reserve(rows);
    for (std::size_t row = 0; row < rows; row++)
    {
      std::vector<std::set<T>> output_row;
      output_row.reserve(cols);
      for (std::size_t col = 0; col < cols; col++)
      {
        output_row.push_back(convert_vector_to_set(input[row][col]));
      }
      output.push_back(output_row);
    }
    return output;
  }
}

namespace multiphenicsx_wrappers
{
void fem_petsc_module(py::module& m)
{
  // Create PETSc matrices
  m.def("create_matrix",
        [](const dolfinx::mesh::Mesh& mesh,
           std::vector<std::reference_wrapper<const dolfinx::common::IndexMap>> index_maps_,
           const std::array<int, 2> index_maps_bs,
           const std::vector<dolfinx::fem::IntegralType>& integral_types_,
           std::array<const dolfinx::graph::AdjacencyList<std::int32_t>*, 2> dofmaps,
           const std::string& matrix_type) {
          // Due to pybind11#2123 the argument index_maps_ is of type
          //   std::vector<std::reference_wrapper<const dolfinx::common::IndexMap>> index_maps
          // rather than
          //   std::array<std::reference_wrapper<const dolfinx::common::IndexMap>, 2> index_maps
          // as in the C++ backend. Convert here std::vector to a std::array.
          // TODO Remove this when pybind11#2123 is fixed.
          assert(index_maps_.size() == 2);
          std::array<std::reference_wrapper<const dolfinx::common::IndexMap>, 2> index_maps{{
            index_maps_[0], index_maps_[1]
          }};
          // Due to pybind11#2122 the argument integral_types_ is of type
          //   const std::vector<dolfinx::fem::IntegralType>&
          // rather than
          //   const std::set<dolfinx::fem::IntegralType>&
          // as in the C++ backend. Convert here std::vector to a std::set.
          // TODO Remove this when pybind11#2122 is fixed.
          auto integral_types = convert_vector_to_set(integral_types_);
          //
          return multiphenicsx::fem::petsc::create_matrix(
                   mesh, index_maps, index_maps_bs, integral_types, dofmaps, matrix_type);
        },
        py::return_value_policy::take_ownership,
        py::arg("mesh"), py::arg("index_maps"), py::arg("index_maps_bs"),
        py::arg("integral_types"), py::arg("dofmaps"), py::arg("matrix_type") = std::string(),
        "Create a PETSc Mat for bilinear form.");
  m.def("create_matrix_block",
        [](const dolfinx::mesh::Mesh& mesh,
           std::array<std::vector<std::reference_wrapper<const dolfinx::common::IndexMap>>, 2> index_maps,
           const std::array<std::vector<int>, 2> index_maps_bs,
           const std::vector<std::vector<std::vector<dolfinx::fem::IntegralType>>>& integral_types_,
           const std::array<std::vector<const dolfinx::graph::AdjacencyList<std::int32_t>*>, 2>& dofmaps,
           const std::string& matrix_type) {
          // Due to pybind11#2122 the argument integral_types_ is of type
          //   const std::vector<std::vector<std::vector<dolfinx::fem::IntegralType>>>&
          // rather than
          //   const std::vector<std::vector<std::set<dolfinx::fem::IntegralType>>>&
          // as in the C++ backend. Convert here each inner std::vector to a std::set.
          // TODO Remove this when pybind11#2122 is fixed.
          auto integral_types = convert_vector_to_set(integral_types_);
          //
          return multiphenicsx::fem::petsc::create_matrix_block(
                   mesh, index_maps, index_maps_bs,integral_types, dofmaps, matrix_type);
        },
        py::return_value_policy::take_ownership,
        py::arg("mesh"), py::arg("index_maps"), py::arg("index_maps_bs"),
        py::arg("integral_types"), py::arg("dofmaps"), py::arg("matrix_type") = std::string(),
        "Create monolithic sparse matrix for stacked bilinear forms.");
  m.def("create_matrix_nest",
        [](const dolfinx::mesh::Mesh& mesh,
           std::array<std::vector<std::reference_wrapper<const dolfinx::common::IndexMap>>, 2> index_maps,
           const std::array<std::vector<int>, 2> index_maps_bs,
           const std::vector<std::vector<std::vector<dolfinx::fem::IntegralType>>>& integral_types_,
           const std::array<std::vector<const dolfinx::graph::AdjacencyList<std::int32_t>*>, 2>& dofmaps,
           const std::vector<std::vector<std::string>>& matrix_types) {
          // Due to pybind11#2122 the argument integral_types_ is of type
          //   const std::vector<std::vector<std::vector<dolfinx::fem::IntegralType>>>&
          // rather than
          //   const std::vector<std::vector<std::set<dolfinx::fem::IntegralType>>>&
          // as in the C++ backend. Convert here each inner std::vector to a std::set.
          // TODO Remove this when pybind11#2122 is fixed.
          auto integral_types = convert_vector_to_set(integral_types_);
          //
          return multiphenicsx::fem::petsc::create_matrix_nest(
                   mesh, index_maps, index_maps_bs, integral_types, dofmaps, matrix_types);
        },
        py::return_value_policy::take_ownership,
        py::arg("mesh"), py::arg("index_maps"), py::arg("index_maps_bs"),
        py::arg("integral_types"), py::arg("dofmaps"),
        py::arg("matrix_types") = std::vector<std::vector<std::string>>(),
        "Create nested sparse matrix for bilinear forms.");
}

void fem(py::module& m)
{
  py::module petsc_mod
      = m.def_submodule("petsc", "PETSc-specific finite element module");
  fem_petsc_module(petsc_mod);

  // utils
  m.def("get_integral_types_from_form", &multiphenicsx::fem::get_integral_types_from_form<PetscScalar>,
        "Extract integral types from a Form.");

  // multiphenicsx::fem::DofMapRestriction
  py::class_<multiphenicsx::fem::DofMapRestriction, std::shared_ptr<multiphenicsx::fem::DofMapRestriction>>(
      m, "DofMapRestriction", "DofMapRestriction object")
      .def(py::init<std::shared_ptr<const dolfinx::fem::DofMap>,
                    const std::vector<std::int32_t>&>(),
           py::arg("dofmap"), py::arg("restriction"))
      .def("cell_dofs",
           [](const multiphenicsx::fem::DofMapRestriction& self, int cell) {
             tcb::span<const std::int32_t> dofs = self.cell_dofs(cell);
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

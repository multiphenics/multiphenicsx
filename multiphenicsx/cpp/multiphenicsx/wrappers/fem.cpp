// Copyright (C) 2016-2024 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <array>
#include <caster_petsc.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <memory>
#include <multiphenicsx/fem/DofMapRestriction.h>
#include <multiphenicsx/fem/petsc.h>
#include <multiphenicsx/fem/utils.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <petsc4py/petsc4py.h>
#include <span>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace
{
template <class T>
std::array<std::reference_wrapper<const T>, 2>
convert_shared_ptr_to_reference_wrapper(
    const std::array<std::shared_ptr<const T>, 2>& input)
{
  return {{*input[0], *input[1]}};
}

template <class T>
std::array<std::vector<std::reference_wrapper<const T>>, 2>
convert_shared_ptr_to_reference_wrapper(
    const std::array<std::vector<std::shared_ptr<const T>>, 2>& input)
{
  std::array<std::vector<std::reference_wrapper<const T>>, 2> output;
  for (int i(0); i < 2; ++i)
  {
    output[i].reserve(input[i].size());
    for (auto& input_i : input[i])
    {
      output[i].push_back(*input_i);
    }
  }
  return output;
}

template <class T, class... Args>
std::span<const T>
convert_ndarray_to_span(const nb::ndarray<const T, Args...>& input)
{
  return std::span(input.data(), input.size());
}

template <class T, class... Args>
std::vector<std::span<const T>>
convert_ndarray_to_span(const std::vector<nb::ndarray<const T, Args...>>& input)
{
  std::vector<std::span<const T>> output;
  output.reserve(input.size());
  for (auto& input_ : input)
    output.push_back(convert_ndarray_to_span(input_));
  return output;
}

template <class T, class... Args>
std::array<std::span<const T>, 2> convert_ndarray_to_span(
    const std::array<nb::ndarray<const T, Args...>, 2>& input)
{
  return {
      {convert_ndarray_to_span(input[0]), convert_ndarray_to_span(input[1])}};
}

template <class T, class... Args>
std::array<std::vector<std::span<const T>>, 2> convert_ndarray_to_span(
    const std::array<std::vector<nb::ndarray<const T, Args...>>, 2>& input)
{
  return {
      {convert_ndarray_to_span(input[0]), convert_ndarray_to_span(input[1])}};
}
} // namespace

namespace multiphenicsx_wrappers
{
void fem_petsc_module(nb::module_& m)
{
  import_petsc4py();

  // Create PETSc matrices
  m.def(
      "create_matrix",
      [](const dolfinx::fem::Form<PetscScalar, PetscReal>& a,
         std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2>
             index_maps_,
         const std::array<int, 2> index_maps_bs,
         std::array<nb::ndarray<const std::int32_t, nb::c_contig>, 2>
             dofmaps_list_,
         std::array<nb::ndarray<const std::size_t, nb::ndim<1>, nb::c_contig>,
                    2>
             dofmaps_bounds_,
         const std::string& matrix_type)
      {
        auto index_maps = convert_shared_ptr_to_reference_wrapper(index_maps_);
        auto dofmaps_list = convert_ndarray_to_span(dofmaps_list_);
        auto dofmaps_bounds = convert_ndarray_to_span(dofmaps_bounds_);
        return multiphenicsx::fem::petsc::create_matrix(
            a, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds,
            matrix_type);
      },
      nb::rv_policy::take_ownership, nb::arg("a"), nb::arg("index_maps"),
      nb::arg("index_maps_bs"), nb::arg("dofmaps_list"),
      nb::arg("dofmaps_bounds"), nb::arg("matrix_type") = std::string(),
      "Create a PETSc Mat for bilinear form.");
  m.def(
      "create_matrix_block",
      [](const std::vector<
             std::vector<const dolfinx::fem::Form<PetscScalar, PetscReal>*>>& a,
         std::array<
             std::vector<std::shared_ptr<const dolfinx::common::IndexMap>>, 2>
             index_maps_,
         const std::array<std::vector<int>, 2> index_maps_bs,
         std::array<std::vector<nb::ndarray<const std::int32_t, nb::c_contig>>,
                    2>
             dofmaps_list_,
         std::array<std::vector<nb::ndarray<const std::size_t, nb::ndim<1>,
                                            nb::c_contig>>,
                    2>
             dofmaps_bounds_,
         const std::string& matrix_type)
      {
        auto index_maps = convert_shared_ptr_to_reference_wrapper(index_maps_);
        auto dofmaps_list = convert_ndarray_to_span(dofmaps_list_);
        auto dofmaps_bounds = convert_ndarray_to_span(dofmaps_bounds_);
        return multiphenicsx::fem::petsc::create_matrix_block(
            a, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds,
            matrix_type);
      },
      nb::rv_policy::take_ownership, nb::arg("a"), nb::arg("index_maps"),
      nb::arg("index_maps_bs"), nb::arg("dofmaps_list"),
      nb::arg("dofmaps_bounds"), nb::arg("matrix_type") = std::string(),
      "Create monolithic sparse matrix for stacked bilinear forms.");
  m.def(
      "create_matrix_nest",
      [](const std::vector<
             std::vector<const dolfinx::fem::Form<PetscScalar, PetscReal>*>>& a,
         std::array<
             std::vector<std::shared_ptr<const dolfinx::common::IndexMap>>, 2>
             index_maps_,
         const std::array<std::vector<int>, 2> index_maps_bs,
         std::array<std::vector<nb::ndarray<const std::int32_t, nb::c_contig>>,
                    2>
             dofmaps_list_,
         std::array<std::vector<nb::ndarray<const std::size_t, nb::ndim<1>,
                                            nb::c_contig>>,
                    2>
             dofmaps_bounds_,
         const std::vector<std::vector<std::string>>& matrix_types)
      {
        auto index_maps = convert_shared_ptr_to_reference_wrapper(index_maps_);
        auto dofmaps_list = convert_ndarray_to_span(dofmaps_list_);
        auto dofmaps_bounds = convert_ndarray_to_span(dofmaps_bounds_);
        return multiphenicsx::fem::petsc::create_matrix_nest(
            a, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds,
            matrix_types);
      },
      nb::rv_policy::take_ownership, nb::arg("a"), nb::arg("index_maps"),
      nb::arg("index_maps_bs"), nb::arg("dofmaps_list"),
      nb::arg("dofmaps_bounds"),
      nb::arg("matrix_types") = std::vector<std::vector<std::string>>(),
      "Create nested sparse matrix for bilinear forms.");
}

void fem(nb::module_& m)
{
  nb::module_ petsc_mod
      = m.def_submodule("petsc", "PETSc-specific finite element module");
  fem_petsc_module(petsc_mod);

  // multiphenicsx::fem::DofMapRestriction
  nb::class_<multiphenicsx::fem::DofMapRestriction>(m, "DofMapRestriction",
                                                    "DofMapRestriction object")
      .def(nb::init<std::shared_ptr<const dolfinx::fem::DofMap>,
                    const std::vector<std::int32_t>&>(),
           nb::arg("dofmap"), nb::arg("restriction"))
      .def(
          "cell_dofs",
          [](const multiphenicsx::fem::DofMapRestriction& self, int cell)
          {
            auto dofs = self.cell_dofs(cell);
            return nb::ndarray<const std::int32_t, nb::numpy>(dofs.data(),
                                                              {dofs.size()});
          },
          nb::rv_policy::reference_internal, nb::arg("cell"))
      .def_prop_ro("dofmap", &multiphenicsx::fem::DofMapRestriction::dofmap)
      .def_prop_ro(
          "unrestricted_to_restricted",
          &multiphenicsx::fem::DofMapRestriction::unrestricted_to_restricted)
      .def_prop_ro(
          "restricted_to_unrestricted",
          &multiphenicsx::fem::DofMapRestriction::restricted_to_unrestricted)
      .def(
          "map",
          [](const multiphenicsx::fem::DofMapRestriction& self)
          {
            auto map = self.map();
            return std::make_pair(nb::ndarray<const std::int32_t, nb::numpy>(
                                      map.first.data(), {map.first.size()}),
                                  nb::ndarray<const std::size_t, nb::numpy>(
                                      map.second.data(), {map.second.size()}));
          },
          nb::rv_policy::reference_internal)
      .def_ro("index_map", &multiphenicsx::fem::DofMapRestriction::index_map)
      .def_prop_ro("index_map_bs",
                   &multiphenicsx::fem::DofMapRestriction::index_map_bs);
}
} // namespace multiphenicsx_wrappers

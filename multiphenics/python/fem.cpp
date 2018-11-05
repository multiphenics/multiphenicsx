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
#include <pybind11/stl.h>

namespace py = pybind11;

namespace multiphenics_wrappers
{
  void fem(py::module& m)
  {
    // multiphenics::fem::BlockDofMap
    py::class_<multiphenics::fem::BlockDofMap, std::shared_ptr<multiphenics::fem::BlockDofMap>, dolfin::GenericDofMap>
      (m, "BlockDofMap", "multiphenics BlockDofMap object")
      .def(py::init<std::vector<std::shared_ptr<const dolfin::GenericDofMap>>,
                    std::vector<std::vector<std::shared_ptr<const dolfin::MeshFunction<bool>>>>,
                    const dolfin::Mesh&>())
      .def("dofmaps", &multiphenics::fem::BlockDofMap::dofmaps)
      .def("ownership_range", &multiphenics::fem::BlockDofMap::ownership_range)
      .def("global_dimension", &multiphenics::fem::BlockDofMap::global_dimension)
      .def("original_to_block", &multiphenics::fem::BlockDofMap::original_to_block)
      .def("block_to_original", &multiphenics::fem::BlockDofMap::block_to_original)
      .def("sub_index_map", &multiphenics::fem::BlockDofMap::sub_index_map);
      
    // multiphenics::fem::BlockFormBase
    py::class_<multiphenics::fem::BlockFormBase, std::shared_ptr<multiphenics::fem::BlockFormBase>>
      (m, "BlockFormBase", "multiphenics BlockFormBase object")
      .def("rank", &multiphenics::fem::BlockFormBase::rank)
      .def("mesh", &multiphenics::fem::BlockFormBase::mesh)
      .def("block_size", &multiphenics::fem::BlockFormBase::block_size);
      
    // multiphenics::fem::BlockForm1
    py::class_<multiphenics::fem::BlockForm1, std::shared_ptr<multiphenics::fem::BlockForm1>, multiphenics::fem::BlockFormBase>
      (m, "BlockForm1", "multiphenics BlockForm1 object")
      .def(py::init<std::vector<std::shared_ptr<const dolfin::Form>>,
                    std::vector<std::shared_ptr<const multiphenics::function::BlockFunctionSpace>>>());
                    
    // multiphenics::fem::BlockForm2
    py::class_<multiphenics::fem::BlockForm2, std::shared_ptr<multiphenics::fem::BlockForm2>, multiphenics::fem::BlockFormBase>
      (m, "BlockForm2", "multiphenics BlockForm2 object")
      .def(py::init<std::vector<std::vector<std::shared_ptr<const dolfin::Form>>>,
                    std::vector<std::shared_ptr<const multiphenics::function::BlockFunctionSpace>>>());
    
    // multiphenics::fem::BlockAssemblerBase
    py::class_<multiphenics::fem::BlockAssemblerBase, std::shared_ptr<multiphenics::fem::BlockAssemblerBase>>
      (m, "BlockAssemblerBase", "multiphenics BlockAssemblerBase object")
      .def_readwrite("add_values", &multiphenics::fem::BlockAssemblerBase::add_values)
      .def_readwrite("keep_diagonal", &multiphenics::fem::BlockAssemblerBase::keep_diagonal)
      .def_readwrite("finalize_tensor", &multiphenics::fem::BlockAssemblerBase::finalize_tensor);

    // multiphenics::fem::BlockAssembler
    py::class_<multiphenics::fem::BlockAssembler, std::shared_ptr<multiphenics::fem::BlockAssembler>, multiphenics::fem::BlockAssemblerBase>
      (m, "BlockAssembler", "multiphenics BlockAssembler object")
      .def(py::init<>())
      .def("assemble", &multiphenics::fem::BlockAssembler::assemble);
      
    // multiphenics::fem::BlockDirichletBC
    py::class_<multiphenics::fem::BlockDirichletBC, std::shared_ptr<multiphenics::fem::BlockDirichletBC>, dolfin::common::Variable>
      (m, "BlockDirichletBC", "multiphenics BlockDirichletBC object")
      .def(py::init<std::vector<std::vector<std::shared_ptr<const DirichletBC>>>,
                    std::shared_ptr<const BlockFunctionSpace>>())
      .def("block_function_space", &multiphenics::fem::BlockDirichletBC::block_function_space)
      .def("zero", &multiphenics::fem::BlockDirichletBC::zero)
      .def("get_boundary_values", [](const multiphenics::fem::BlockDirichletBC& instance)
           {
             multiphenics::fem::BlockDirichletBC::Map map;
             instance.get_boundary_values(map);
             return map;
           })
      .def("apply", (void (multiphenics::fem::BlockDirichletBC::*)(dolfin::GenericVector&) const)
           &multiphenics::fem::BlockDirichletBC::apply)
      .def("apply", (void (multiphenics::fem::BlockDirichletBC::*)(dolfin::GenericMatrix&) const)
           &multiphenics::fem::BlockDirichletBC::apply)
      .def("apply", (void (multiphenics::fem::BlockDirichletBC::*)(dolfin::GenericMatrix&, dolfin::GenericVector&) const)
           &multiphenics::fem::BlockDirichletBC::apply)
      .def("apply", (void (multiphenics::fem::BlockDirichletBC::*)(dolfin::GenericVector&, const dolfin::GenericVector&) const)
           &multiphenics::fem::BlockDirichletBC::apply)
      .def("apply", (void (multiphenics::fem::BlockDirichletBC::*)(dolfin::GenericMatrix&, dolfin::GenericVector&, const dolfin::GenericVector&) const)
           &multiphenics::fem::BlockDirichletBC::apply);
  }
}

// Copyright (C) 2016-2017 by the multiphenics authors
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
    // multiphenics::BlockDofMap
    py::class_<multiphenics::BlockDofMap, std::shared_ptr<multiphenics::BlockDofMap>, dolfin::GenericDofMap>
      (m, "BlockDofMap", "multiphenics BlockDofMap object")
      .def(py::init<std::vector<std::shared_ptr<const dolfin::GenericDofMap>>,
                    std::vector<std::vector<std::shared_ptr<const dolfin::MeshFunction<bool>>>>,
                    const dolfin::Mesh&>())
      .def("ownership_range", &multiphenics::BlockDofMap::ownership_range)
      .def("global_dimension", &multiphenics::BlockDofMap::global_dimension)
      .def("block_to_original", &multiphenics::BlockDofMap::block_to_original)
      .def("sub_index_map", &multiphenics::BlockDofMap::sub_index_map)
      ;
      
    // multiphenics::BlockFormBase
    py::class_<multiphenics::BlockFormBase, std::shared_ptr<multiphenics::BlockFormBase>>
      (m, "BlockFormBase", "multiphenics BlockFormBase object")
      .def("rank", &multiphenics::BlockFormBase::rank)
      .def("mesh", &multiphenics::BlockFormBase::mesh)
      .def("block_size", &multiphenics::BlockFormBase::block_size);
      
    // multiphenics::BlockForm1
    py::class_<multiphenics::BlockForm1, std::shared_ptr<multiphenics::BlockForm1>, multiphenics::BlockFormBase>
      (m, "BlockForm1", "multiphenics BlockForm1 object")
      .def(py::init<std::vector<std::shared_ptr<const dolfin::Form>>,
                    std::vector<std::shared_ptr<const multiphenics::BlockFunctionSpace>>>());
                    
    // multiphenics::BlockForm2
    py::class_<multiphenics::BlockForm2, std::shared_ptr<multiphenics::BlockForm2>, multiphenics::BlockFormBase>
      (m, "BlockForm2", "multiphenics BlockForm2 object")
      .def(py::init<std::vector<std::vector<std::shared_ptr<const dolfin::Form>>>,
                    std::vector<std::shared_ptr<const multiphenics::BlockFunctionSpace>>>());
    
    // multiphenics::BlockAssemblerBase
    py::class_<multiphenics::BlockAssemblerBase, std::shared_ptr<multiphenics::BlockAssemblerBase>>
      (m, "BlockAssemblerBase", "multiphenics BlockAssemblerBase object")
      .def_readwrite("add_values", &multiphenics::BlockAssemblerBase::add_values)
      .def_readwrite("keep_diagonal", &multiphenics::BlockAssemblerBase::keep_diagonal)
      .def_readwrite("finalize_tensor", &multiphenics::BlockAssemblerBase::finalize_tensor);

    // multiphenics::BlockAssembler
    py::class_<multiphenics::BlockAssembler, std::shared_ptr<multiphenics::BlockAssembler>, multiphenics::BlockAssemblerBase>
      (m, "BlockAssembler", "multiphenics BlockAssembler object")
      .def(py::init<>())
      .def("assemble", &multiphenics::BlockAssembler::assemble);
      
    // multiphenics::BlockDirichletBC
    py::class_<multiphenics::BlockDirichletBC, std::shared_ptr<multiphenics::BlockDirichletBC>, dolfin::Variable>
      (m, "BlockDirichletBC", "multiphenics BlockDirichletBC object")
      .def(py::init<std::vector<std::vector<std::shared_ptr<const DirichletBC>>>,
                    std::shared_ptr<const BlockFunctionSpace>>())
      .def("block_function_space", &multiphenics::BlockDirichletBC::block_function_space)
      .def("zero", &multiphenics::BlockDirichletBC::zero)
      .def("get_boundary_values", [](const multiphenics::BlockDirichletBC& instance)
           {
             multiphenics::BlockDirichletBC::Map map;
             instance.get_boundary_values(map);
             return map;
           })
      .def("apply", (void (multiphenics::BlockDirichletBC::*)(dolfin::GenericVector&) const)
           &multiphenics::BlockDirichletBC::apply)
      .def("apply", (void (multiphenics::BlockDirichletBC::*)(dolfin::GenericMatrix&) const)
           &multiphenics::BlockDirichletBC::apply)
      .def("apply", (void (multiphenics::BlockDirichletBC::*)(dolfin::GenericMatrix&, dolfin::GenericVector&) const)
           &multiphenics::BlockDirichletBC::apply)
      .def("apply", (void (multiphenics::BlockDirichletBC::*)(dolfin::GenericVector&, const dolfin::GenericVector&) const)
           &multiphenics::BlockDirichletBC::apply)
      .def("apply", (void (multiphenics::BlockDirichletBC::*)(dolfin::GenericMatrix&, dolfin::GenericVector&, const dolfin::GenericVector&) const)
           &multiphenics::BlockDirichletBC::apply);
  }
}

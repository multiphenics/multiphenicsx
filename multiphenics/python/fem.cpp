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
    // dolfin::BlockDofMap
    py::class_<dolfin::BlockDofMap, std::shared_ptr<dolfin::BlockDofMap>>
      (m, "BlockDofMap", "multiphenics BlockDofMap object")
      .def(py::init<std::vector<std::shared_ptr<const dolfin::GenericDofMap>>,
                    std::vector<std::vector<std::shared_ptr<const dolfin::MeshFunction<bool>>>>,
                    const dolfin::Mesh&>())
      .def("ownership_range", &dolfin::BlockDofMap::ownership_range);
      
    // dolfin::BlockFormBase
    py::class_<dolfin::BlockFormBase, std::shared_ptr<dolfin::BlockFormBase>>
      (m, "BlockFormBase", "multiphenics BlockFormBase object")
      .def("rank", &dolfin::BlockFormBase::rank)
      .def("mesh", &dolfin::BlockFormBase::mesh);
      
    // dolfin::BlockForm1
    py::class_<dolfin::BlockForm1, std::shared_ptr<dolfin::BlockForm1>>
      (m, "BlockForm1", "multiphenics BlockForm1 object")
      .def(py::init<std::vector<std::shared_ptr<const dolfin::Form>>,
                    std::vector<std::shared_ptr<const dolfin::BlockFunctionSpace>>>());
                    
    // dolfin::BlockForm2
    py::class_<dolfin::BlockForm2, std::shared_ptr<dolfin::BlockForm2>>
      (m, "BlockForm2", "multiphenics BlockForm2 object")
      .def(py::init<std::vector<std::vector<std::shared_ptr<const dolfin::Form>>>,
                    std::vector<std::shared_ptr<const dolfin::BlockFunctionSpace>>>());
    
    // dolfin::BlockAssemblerBase
    py::class_<dolfin::BlockAssemblerBase, std::shared_ptr<dolfin::BlockAssemblerBase>>
      (m, "BlockAssemblerBase", "multiphenics BlockAssemblerBase object")
      .def_readwrite("add_values", &dolfin::BlockAssemblerBase::add_values)
      .def_readwrite("keep_diagonal", &dolfin::BlockAssemblerBase::keep_diagonal)
      .def_readwrite("finalize_tensor", &dolfin::BlockAssemblerBase::finalize_tensor);

    // dolfin::BlockAssembler
    py::class_<dolfin::BlockAssembler, std::shared_ptr<dolfin::BlockAssembler>, dolfin::BlockAssemblerBase>
      (m, "BlockAssembler", "multiphenics BlockAssembler object")
      .def(py::init<>())
      .def("assemble", &dolfin::BlockAssembler::assemble);
      
    // dolfin::BlockDirichletBC
    py::class_<dolfin::BlockDirichletBC, std::shared_ptr<dolfin::BlockDirichletBC>>
      (m, "BlockDirichletBC", "multiphenics BlockDirichletBC object")
      .def(py::init<std::vector<std::vector<std::shared_ptr<const DirichletBC>>>,
                    std::shared_ptr<const BlockFunctionSpace>>())
      .def("block_function_space", &dolfin::BlockDirichletBC::block_function_space)
      .def("zero", &dolfin::BlockDirichletBC::zero)
      .def("get_boundary_values", [](const dolfin::BlockDirichletBC& instance)
           {
             dolfin::BlockDirichletBC::Map map;
             instance.get_boundary_values(map);
             return map;
           })
      .def("apply", (void (dolfin::BlockDirichletBC::*)(dolfin::GenericVector&) const)
           &dolfin::BlockDirichletBC::apply)
      .def("apply", (void (dolfin::BlockDirichletBC::*)(dolfin::GenericMatrix&) const)
           &dolfin::BlockDirichletBC::apply)
      .def("apply", (void (dolfin::BlockDirichletBC::*)(dolfin::GenericMatrix&, dolfin::GenericVector&) const)
           &dolfin::BlockDirichletBC::apply);
  }
}

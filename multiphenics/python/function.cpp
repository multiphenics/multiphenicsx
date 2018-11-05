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
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace multiphenics_wrappers
{
  void function(py::module& m)
  {
    // multiphenics::function::BlockFunctionSpace
    py::class_<multiphenics::function::BlockFunctionSpace, std::shared_ptr<multiphenics::function::BlockFunctionSpace>, dolfin::common::Variable>
      (m, "BlockFunctionSpace", "A finite element block function space", py::dynamic_attr())
      .def(py::init<std::vector<std::shared_ptr<const dolfin::FunctionSpace>>>())
      .def(py::init<std::vector<std::shared_ptr<const dolfin::FunctionSpace>>,
                    std::vector<std::vector<std::shared_ptr<const dolfin::MeshFunction<bool>>>>>())
      .def(py::init<std::shared_ptr<const dolfin::Mesh>,
                    std::vector<std::shared_ptr<const dolfin::FiniteElement>>,
                    std::vector<std::shared_ptr<const dolfin::GenericDofMap>>>())
      .def(py::init<std::shared_ptr<const dolfin::Mesh>,
                    std::vector<std::shared_ptr<const dolfin::FiniteElement>>,
                    std::vector<std::shared_ptr<const dolfin::GenericDofMap>>,
                    std::vector<std::vector<std::shared_ptr<const dolfin::MeshFunction<bool>>>>>())
      .def(py::init<const multiphenics::function::BlockFunctionSpace&>())
      .def("__eq__", &multiphenics::function::BlockFunctionSpace::operator==)
      .def("dim", &multiphenics::function::BlockFunctionSpace::dim)
      .def("contains", &multiphenics::function::BlockFunctionSpace::contains)
      .def("elements", &multiphenics::function::BlockFunctionSpace::elements)
      .def("mesh", &multiphenics::function::BlockFunctionSpace::mesh)
      .def("dofmaps", &multiphenics::function::BlockFunctionSpace::dofmaps)
      .def("block_dofmap", &multiphenics::function::BlockFunctionSpace::block_dofmap)
      .def("sub", (std::shared_ptr<dolfin::FunctionSpace> (multiphenics::function::BlockFunctionSpace::*)(std::size_t) const)
           &multiphenics::function::BlockFunctionSpace::sub)
      .def("extract_block_sub_space", &multiphenics::function::BlockFunctionSpace::extract_block_sub_space)
      .def("tabulate_dof_coordinates", [](const multiphenics::function::BlockFunctionSpace& self)
           {
             const std::size_t gdim = self.mesh()->geometry().dim();
             std::vector<double> coords = self.tabulate_dof_coordinates();
             assert(coords.size() % gdim  == 0);

             py::array_t<double> c({coords.size()/gdim, gdim}, coords.data() );
             return c;
           });
           
    // multiphenics::function::BlockFunction
    py::class_<multiphenics::function::BlockFunction, std::shared_ptr<multiphenics::function::BlockFunction>>
      (m, "BlockFunction", "A finite element block function")
      .def(py::init<std::shared_ptr<const multiphenics::function::BlockFunctionSpace>>(), "Create a function on the given block function space")
      .def(py::init<std::shared_ptr<const multiphenics::function::BlockFunctionSpace>, std::vector<std::shared_ptr<Function>>>())
      .def(py::init<std::shared_ptr<const multiphenics::function::BlockFunctionSpace>, std::shared_ptr<dolfin::GenericVector>>())
      .def(py::init<std::shared_ptr<const multiphenics::function::BlockFunctionSpace>, std::shared_ptr<dolfin::GenericVector>,
                    std::vector<std::shared_ptr<Function>>>())
      .def(py::init<multiphenics::function::BlockFunction&>())
      .def("_assign", (const multiphenics::function::BlockFunction& (multiphenics::function::BlockFunction::*)(const multiphenics::function::BlockFunction&))
           &multiphenics::function::BlockFunction::operator=)
      .def("sub", &multiphenics::function::BlockFunction::operator[])
      .def("block_vector", (std::shared_ptr<const dolfin::GenericVector> (multiphenics::function::BlockFunction::*)() const)
           &multiphenics::function::BlockFunction::block_vector, "Return the block vector associated with the finite element BlockFunction")
      .def("apply", &multiphenics::function::BlockFunction::apply);
  }
}

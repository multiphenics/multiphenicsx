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
    // multiphenics::BlockFunctionSpace
    py::class_<multiphenics::BlockFunctionSpace, std::shared_ptr<multiphenics::BlockFunctionSpace>, dolfin::Variable>
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
      .def(py::init<const multiphenics::BlockFunctionSpace&>())
      .def("__eq__", &multiphenics::BlockFunctionSpace::operator==)
      .def("dim", &multiphenics::BlockFunctionSpace::dim)
      .def("contains", &multiphenics::BlockFunctionSpace::contains)
      .def("elements", &multiphenics::BlockFunctionSpace::elements)
      .def("mesh", &multiphenics::BlockFunctionSpace::mesh)
      .def("dofmaps", &multiphenics::BlockFunctionSpace::dofmaps)
      .def("block_dofmap", &multiphenics::BlockFunctionSpace::block_dofmap)
      .def("sub", (std::shared_ptr<dolfin::FunctionSpace> (multiphenics::BlockFunctionSpace::*)(std::size_t) const)
           &multiphenics::BlockFunctionSpace::sub)
      .def("extract_block_sub_space", &multiphenics::BlockFunctionSpace::extract_block_sub_space)
      .def("tabulate_dof_coordinates", [](const multiphenics::BlockFunctionSpace& self)
           {
             const std::size_t gdim = self.mesh()->geometry().dim();
             std::vector<double> coords = self.tabulate_dof_coordinates();
             assert(coords.size() % gdim  == 0);

             py::array_t<double> c({coords.size()/gdim, gdim}, coords.data() );
             return c;
           });
           
    // multiphenics::BlockFunction
    py::class_<multiphenics::BlockFunction, std::shared_ptr<multiphenics::BlockFunction>>
      (m, "BlockFunction", "A finite element block function")
      .def(py::init<std::shared_ptr<const multiphenics::BlockFunctionSpace>>(), "Create a function on the given block function space")
      .def(py::init<std::shared_ptr<const multiphenics::BlockFunctionSpace>, std::vector<std::shared_ptr<Function>>>())
      .def(py::init<std::shared_ptr<const multiphenics::BlockFunctionSpace>, std::shared_ptr<dolfin::GenericVector>>())
      .def(py::init<std::shared_ptr<const multiphenics::BlockFunctionSpace>, std::shared_ptr<dolfin::GenericVector>,
                    std::vector<std::shared_ptr<Function>>>())
      .def(py::init<multiphenics::BlockFunction&>())
      .def("_assign", (const multiphenics::BlockFunction& (multiphenics::BlockFunction::*)(const multiphenics::BlockFunction&))
           &multiphenics::BlockFunction::operator=)
      .def("sub", &multiphenics::BlockFunction::operator[])
      .def("block_vector", (std::shared_ptr<const dolfin::GenericVector> (multiphenics::BlockFunction::*)() const)
           &multiphenics::BlockFunction::block_vector, "Return the block vector associated with the finite element BlockFunction")
      .def("apply", &multiphenics::BlockFunction::apply);
  }
}

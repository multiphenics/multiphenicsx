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
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace multiphenics_wrappers
{
  void function(py::module& m)
  {
    // dolfin::BlockFunctionSpace
    py::class_<dolfin::BlockFunctionSpace, std::shared_ptr<dolfin::BlockFunctionSpace>, dolfin::Variable>
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
      .def(py::init<const dolfin::BlockFunctionSpace&>())
      .def("__eq__", &dolfin::BlockFunctionSpace::operator==)
      .def("dim", &dolfin::BlockFunctionSpace::dim)
      .def("contains", &dolfin::BlockFunctionSpace::contains)
      .def("elements", &dolfin::BlockFunctionSpace::elements)
      .def("mesh", &dolfin::BlockFunctionSpace::mesh)
      .def("dofmaps", &dolfin::BlockFunctionSpace::dofmaps)
      .def("block_dofmap", &dolfin::BlockFunctionSpace::block_dofmap)
      .def("sub", (std::shared_ptr<dolfin::FunctionSpace> (dolfin::BlockFunctionSpace::*)(std::size_t) const)
           &dolfin::BlockFunctionSpace::sub)
      .def("extract_block_sub_space", &dolfin::BlockFunctionSpace::extract_block_sub_space)
      .def("tabulate_dof_coordinates", [](const dolfin::BlockFunctionSpace& self)
           {
             const std::size_t gdim = self.mesh()->geometry().dim();
             std::vector<double> coords = self.tabulate_dof_coordinates();
             assert(coords.size() % gdim  == 0);

             py::array_t<double> c({coords.size()/gdim, gdim}, coords.data() );
             return c;
           });
           
    // dolfin::BlockFunction
    py::class_<dolfin::BlockFunction, std::shared_ptr<dolfin::BlockFunction>>
      (m, "BlockFunction", "A finite element block function")
      .def(py::init<std::shared_ptr<const dolfin::BlockFunctionSpace>>(), "Create a function on the given block function space")
      .def(py::init<std::shared_ptr<const dolfin::BlockFunctionSpace>, std::vector<std::shared_ptr<Function>>>())
      .def(py::init<dolfin::BlockFunction&>())
      .def("_assign", (const dolfin::BlockFunction& (dolfin::BlockFunction::*)(const dolfin::BlockFunction&))
           &dolfin::BlockFunction::operator=)
      .def("sub", &dolfin::BlockFunction::operator[])
      .def("block_vector", (std::shared_ptr<const dolfin::GenericVector> (dolfin::BlockFunction::*)() const)
           &dolfin::BlockFunction::block_vector, "Return the block vector associated with the finite element BlockFunction");
  }
}

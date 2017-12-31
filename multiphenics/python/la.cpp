// Copyright (C) 2016-2018 by the multiphenics authors
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

#include <multiphenics/python/mpi_casters.h> // TODO remove local copy of DOLFIN's pybind11 files
#include <multiphenics/python/petsc_casters.h> // TODO remove local copy of DOLFIN's pybind11 files

namespace py = pybind11;

namespace multiphenics_wrappers
{
  void la(py::module& m)
  {
    // multiphenics::BlockMATLABExport
    py::class_<multiphenics::BlockMATLABExport>
      (m, "BlockMATLABExport", "multiphenics BlockMATLABExport object")
      .def_static("export_", (void (*)(const dolfin::PETScMatrix&, std::string))
                  &multiphenics::BlockMATLABExport::export_)
      .def_static("export_", (void (*)(const dolfin::PETScVector&, std::string))
                  &multiphenics::BlockMATLABExport::export_);
      
    // multiphenics::BlockInsertMode
    py::enum_<multiphenics::BlockInsertMode>
      (m, "BlockInsertMode", "multiphenics BlockInsertMode enum")
      .value("INSERT_VALUES", multiphenics::BlockInsertMode::INSERT_VALUES)
      .value("ADD_VALUES", multiphenics::BlockInsertMode::ADD_VALUES);
      
    // multiphenics::GenericBlockVector
    py::class_<multiphenics::GenericBlockVector, std::shared_ptr<multiphenics::GenericBlockVector>>
      (m, "GenericBlockVector", "multiphenics GenericBlockVector object", py::dynamic_attr())
      .def("attach_block_dof_map", &multiphenics::GenericBlockVector::attach_block_dof_map)
      .def("get_block_dof_map", &multiphenics::GenericBlockVector::get_block_dof_map)
      .def("has_block_dof_map", &multiphenics::GenericBlockVector::has_block_dof_map);
      
    // multiphenics::GenericBlockMatrix
    py::class_<multiphenics::GenericBlockMatrix, std::shared_ptr<multiphenics::GenericBlockMatrix>>
      (m, "GenericBlockMatrix", "multiphenics GenericBlockMatrix object")
      .def("attach_block_dof_map", &multiphenics::GenericBlockMatrix::attach_block_dof_map)
      .def("get_block_dof_map", &multiphenics::GenericBlockMatrix::get_block_dof_map)
      .def("has_block_dof_map", &multiphenics::GenericBlockMatrix::has_block_dof_map);
      
    #ifdef HAS_PETSC
    // multiphenics::BlockPETScVector
    py::class_<multiphenics::BlockPETScVector, std::shared_ptr<multiphenics::BlockPETScVector>, multiphenics::GenericBlockVector, dolfin::PETScVector>
      (m, "BlockPETScVector", "multiphenics BlockPETScVector object")
      .def(py::init<>())
      .def(py::init<Vec>());
      
    // multiphenics::BlockPETScMatrix
    py::class_<multiphenics::BlockPETScMatrix, std::shared_ptr<multiphenics::BlockPETScMatrix>, multiphenics::GenericBlockMatrix, dolfin::PETScMatrix>
      (m, "BlockPETScMatrix", "multiphenics BlockPETScMatrix object")
      .def(py::init<>())
      .def(py::init<Mat>());
      
    // multiphenics::BlockPETScSubVector
    py::class_<multiphenics::BlockPETScSubVector, std::shared_ptr<multiphenics::BlockPETScSubVector>, dolfin::PETScVector>
      (m, "BlockPETScSubVector", "multiphenics BlockPETScSubVector object");
      
    // multiphenics::BlockPETScSubMatrix
    py::class_<multiphenics::BlockPETScSubMatrix, std::shared_ptr<multiphenics::BlockPETScSubMatrix>, dolfin::PETScMatrix>
      (m, "BlockPETScSubMatrix", "multiphenics BlockPETScSubMatrix object");
    #endif
      
    // multiphenics::GenericBlockLinearAlgebraFactory
    py::class_<multiphenics::GenericBlockLinearAlgebraFactory, std::shared_ptr<multiphenics::GenericBlockLinearAlgebraFactory>, dolfin::GenericLinearAlgebraFactory>
      (m, "GenericBlockLinearAlgebraFactory", "multiphenics GenericBlockLinearAlgebraFactory object");
      
    // multiphenics::BlockDefaultFactory
    py::class_<multiphenics::BlockDefaultFactory, std::shared_ptr<multiphenics::BlockDefaultFactory>, multiphenics::GenericBlockLinearAlgebraFactory>
      (m, "BlockDefaultFactory", "multiphenics BlockDefaultFactory object")
      .def(py::init<>())
      .def_static("factory", &multiphenics::BlockDefaultFactory::factory)
      .def("create_matrix", [](const multiphenics::BlockDefaultFactory &self, const dolfin_wrappers::MPICommWrapper comm)
        { return self.create_matrix(comm.get()); })
      .def("create_vector", [](const multiphenics::BlockDefaultFactory &self, const dolfin_wrappers::MPICommWrapper comm)
        { return self.create_vector(comm.get()); })
      .def("wrap_matrix", &multiphenics::BlockDefaultFactory::wrap_matrix)
      .def("wrap_vector", &multiphenics::BlockDefaultFactory::wrap_vector);
    
    #ifdef HAS_PETSC
    // multiphenics::BlockPETScFactory
    py::class_<multiphenics::BlockPETScFactory, std::shared_ptr<multiphenics::BlockPETScFactory>, multiphenics::GenericBlockLinearAlgebraFactory>
      (m, "BlockPETScFactory", "multiphenics BlockPETScFactory object")
      .def("instance", &multiphenics::BlockPETScFactory::instance)
      .def("create_matrix", [](const multiphenics::BlockPETScFactory &self, const dolfin_wrappers::MPICommWrapper comm)
        { return self.create_matrix(comm.get()); })
      .def("create_vector", [](const multiphenics::BlockPETScFactory &self, const dolfin_wrappers::MPICommWrapper comm)
        { return self.create_vector(comm.get()); })
      .def("wrap_matrix", &multiphenics::BlockPETScFactory::wrap_matrix)
      .def("wrap_vector", &multiphenics::BlockPETScFactory::wrap_vector);
    #endif
    
    #ifdef HAS_SLEPC
    // dolfin::CondensedSLEPcEigenSolver
    py::class_<dolfin::CondensedSLEPcEigenSolver, std::shared_ptr<dolfin::CondensedSLEPcEigenSolver>, dolfin::SLEPcEigenSolver>
      (m, "CondensedSLEPcEigenSolver", "multiphenics CondensedSLEPcEigenSolver object")
      .def(py::init<std::shared_ptr<const dolfin::PETScMatrix>,
                    std::vector<std::shared_ptr<const dolfin::DirichletBC>>>())
      .def(py::init<std::shared_ptr<const dolfin::PETScMatrix>, std::shared_ptr<const dolfin::PETScMatrix>,
                    std::vector<std::shared_ptr<const dolfin::DirichletBC>>>())
      .def("get_eigenpair", [](dolfin::CondensedSLEPcEigenSolver& self, std::size_t i, dolfin::Function& r_fun, dolfin::Function& c_fun)
           {
             double lr, lc;
             dolfin::PETScVector r, c; // cannot use r_fun and c_fun vectors due to different ghosting
             self.get_eigenpair(lr, lc, r, c, i);
             std::vector<double> r_local;
             r.get_local(r_local);
             r_fun.vector()->set_local(r_local);
             r_fun.vector()->apply("insert");
             std::vector<double> c_local;
             c.get_local(c_local);
             c_fun.vector()->set_local(c_local);
             c_fun.vector()->apply("insert");
             return py::make_tuple(lr, lc, r_fun, c_fun);
           });
    
    // multiphenics::CondensedBlockSLEPcEigenSolver
    py::class_<multiphenics::CondensedBlockSLEPcEigenSolver, std::shared_ptr<multiphenics::CondensedBlockSLEPcEigenSolver>, dolfin::CondensedSLEPcEigenSolver>
      (m, "CondensedBlockSLEPcEigenSolver", "multiphenics CondensedBlockSLEPcEigenSolver object")
      .def(py::init<std::shared_ptr<const multiphenics::BlockPETScMatrix>,
                    std::shared_ptr<const multiphenics::BlockDirichletBC>>())
      .def(py::init<std::shared_ptr<const multiphenics::BlockPETScMatrix>, std::shared_ptr<const multiphenics::BlockPETScMatrix>,
                    std::shared_ptr<const multiphenics::BlockDirichletBC>>())
      .def("get_eigenpair", [](multiphenics::CondensedBlockSLEPcEigenSolver& self, std::size_t i, multiphenics::BlockFunction& r_fun, multiphenics::BlockFunction& c_fun)
           {
             double lr, lc;
             multiphenics::BlockPETScVector r, c; // cannot use r_fun and c_fun block vectors due to different ghosting
             self.get_eigenpair(lr, lc, r, c, i);
             std::vector<double> r_local;
             r.get_local(r_local);
             r_fun.block_vector()->set_local(r_local);
             r_fun.block_vector()->apply("insert");
             r_fun.apply("to subfunctions");
             std::vector<double> c_local;
             c.get_local(c_local);
             c_fun.block_vector()->set_local(c_local);
             c_fun.block_vector()->apply("insert");
             c_fun.apply("to subfunctions");
             return py::make_tuple(lr, lc, r_fun, c_fun);
           });
    #endif
  }
}

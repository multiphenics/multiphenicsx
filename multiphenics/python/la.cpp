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

#include <multiphenics/python/mpi_casters.h> // TODO remove local copy of DOLFIN's pybind11 files
#include <multiphenics/python/petsc_casters.h> // TODO remove local copy of DOLFIN's pybind11 files

namespace py = pybind11;

namespace multiphenics_wrappers
{
  void la(py::module& m)
  {
    // multiphenics::la::BlockMATLABExport
    py::class_<multiphenics::la::BlockMATLABExport>
      (m, "BlockMATLABExport", "multiphenics BlockMATLABExport object")
      .def_static("export_", (void (*)(const dolfin::la::PETScMatrix&, std::string))
                  &multiphenics::la::BlockMATLABExport::export_)
      .def_static("export_", (void (*)(const dolfin::la::PETScVector&, std::string))
                  &multiphenics::la::BlockMATLABExport::export_);
      
    // multiphenics::la::BlockInsertMode
    py::enum_<multiphenics::la::BlockInsertMode>
      (m, "BlockInsertMode", "multiphenics BlockInsertMode enum")
      .value("INSERT_VALUES", multiphenics::la::BlockInsertMode::INSERT_VALUES)
      .value("ADD_VALUES", multiphenics::la::BlockInsertMode::ADD_VALUES);
      
    // multiphenics::la::BlockPETScSubVector
    py::class_<multiphenics::la::BlockPETScSubVector, std::shared_ptr<multiphenics::la::BlockPETScSubVector>, dolfin::la::PETScVector>
      (m, "BlockPETScSubVector", "multiphenics BlockPETScSubVector object");
      
    // multiphenics::la::BlockPETScSubMatrix
    py::class_<multiphenics::la::BlockPETScSubMatrix, std::shared_ptr<multiphenics::la::BlockPETScSubMatrix>, dolfin::la::PETScMatrix>
      (m, "BlockPETScSubMatrix", "multiphenics BlockPETScSubMatrix object");
      
    #ifdef HAS_SLEPC
    // dolfin::la::CondensedSLEPcEigenSolver
    py::class_<dolfin::la::CondensedSLEPcEigenSolver, std::shared_ptr<dolfin::la::CondensedSLEPcEigenSolver>, dolfin::la::SLEPcEigenSolver>
      (m, "CondensedSLEPcEigenSolver", "multiphenics CondensedSLEPcEigenSolver object")
      .def(py::init([](const dolfin_wrappers::MPICommWrapper comm) {
        return std::make_unique<dolfin::la::CondensedSLEPcEigenSolver>(comm.get());
      }))
      .def("set_operators",
           &dolfin::la::CondensedSLEPcEigenSolver::set_operators)
      .def("set_boundary_conditions",
           &dolfin::la::CondensedSLEPcEigenSolver::set_boundary_conditions)
      .def("get_eigenpair", [](dolfin::la::CondensedSLEPcEigenSolver& self, dolfin::function::Function& r_fun, dolfin::function::Function& c_fun, std::size_t i)
           {
             double lr, lc;
             dolfin::la::PETScVector r, c; // cannot use r_fun and c_fun vectors due to different ghosting
             self.get_eigenpair(lr, lc, r, c, i);
             std::vector<double> r_local;
             r.get_local(r_local);
             r_fun.vector()->set_local(r_local);
             r_fun.vector()->apply();
             std::vector<double> c_local;
             c.get_local(c_local);
             c_fun.vector()->set_local(c_local);
             c_fun.vector()->apply();
             return py::make_tuple(lr, lc, r_fun, c_fun);
           });
    
    // multiphenics::la::CondensedBlockSLEPcEigenSolver
    py::class_<multiphenics::la::CondensedBlockSLEPcEigenSolver, std::shared_ptr<multiphenics::la::CondensedBlockSLEPcEigenSolver>, dolfin::la::CondensedSLEPcEigenSolver>
      (m, "CondensedBlockSLEPcEigenSolver", "multiphenics CondensedBlockSLEPcEigenSolver object")
      .def(py::init([](const dolfin_wrappers::MPICommWrapper comm) {
        return std::make_unique<multiphenics::la::CondensedBlockSLEPcEigenSolver>(comm.get());
      }))
      .def("set_operators",
           &multiphenics::la::CondensedBlockSLEPcEigenSolver::set_operators)
      .def("set_boundary_conditions", (void (multiphenics::la::CondensedBlockSLEPcEigenSolver::*)(std::shared_ptr<const multiphenics::fem::BlockDirichletBC>))
           &multiphenics::la::CondensedBlockSLEPcEigenSolver::set_boundary_conditions)
      .def("get_eigenpair", [](multiphenics::la::CondensedBlockSLEPcEigenSolver& self, multiphenics::function::BlockFunction& r_fun, multiphenics::function::BlockFunction& c_fun, std::size_t i)
           {
             double lr, lc;
             dolfin::la::PETScVector r, c; // cannot use r_fun and c_fun block vectors due to different ghosting
             self.get_eigenpair(lr, lc, r, c, i);
             std::vector<double> r_local;
             r.get_local(r_local);
             r_fun.block_vector()->set_local(r_local);
             r_fun.block_vector()->apply();
             r_fun.apply("to subfunctions");
             std::vector<double> c_local;
             c.get_local(c_local);
             c_fun.block_vector()->set_local(c_local);
             c_fun.block_vector()->apply();
             c_fun.apply("to subfunctions");
             return py::make_tuple(lr, lc, r_fun, c_fun);
           });
    #endif
  }
}

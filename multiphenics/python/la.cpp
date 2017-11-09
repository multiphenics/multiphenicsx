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
  void la(py::module& m)
  {
    // dolfin::BlockMATLABExport
    py::class_<dolfin::BlockMATLABExport>
      (m, "BlockMATLABExport", "multiphenics BlockMATLABExport object")
      .def_static("export_", (void (*)(const dolfin::PETScMatrix&, std::string))
                  &dolfin::BlockMATLABExport::export_)
      .def_static("export_", (void (*)(const dolfin::PETScVector&, std::string))
                  &dolfin::BlockMATLABExport::export_);
      
    // dolfin::BlockInsertMode
    py::enum_<dolfin::BlockInsertMode>
      (m, "BlockInsertMode", "multiphenics BlockInsertMode enum")
      .value("INSERT_VALUES", dolfin::BlockInsertMode::INSERT_VALUES)
      .value("ADD_VALUES", dolfin::BlockInsertMode::ADD_VALUES);
      
    // dolfin::GenericBlockVector
    py::class_<dolfin::GenericBlockVector, std::shared_ptr<dolfin::GenericBlockVector>>
      (m, "GenericBlockVector", "multiphenics GenericBlockVector object");
      
    // dolfin::GenericBlockMatrix
    py::class_<dolfin::GenericBlockMatrix, std::shared_ptr<dolfin::GenericBlockMatrix>>
      (m, "GenericBlockMatrix", "multiphenics GenericBlockMatrix object");
      
    #ifdef HAS_PETSC
    // dolfin::BlockPETScVector
    py::class_<dolfin::BlockPETScVector, std::shared_ptr<dolfin::BlockPETScVector>, dolfin::GenericBlockVector, dolfin::PETScVector>
      (m, "BlockPETScVector", "multiphenics BlockPETScVector object");
      
    // dolfin::BlockPETScMatrix
    py::class_<dolfin::BlockPETScMatrix, std::shared_ptr<dolfin::BlockPETScMatrix>, dolfin::GenericBlockMatrix, dolfin::PETScMatrix>
      (m, "BlockPETScMatrix", "multiphenics BlockPETScMatrix object");
      
    // dolfin::BlockPETScSubVector
    py::class_<dolfin::BlockPETScSubVector, std::shared_ptr<dolfin::BlockPETScSubVector>, dolfin::PETScVector>
      (m, "BlockPETScSubVector", "multiphenics BlockPETScSubVector object");
      
    // dolfin::BlockPETScSubMatrix
    py::class_<dolfin::BlockPETScSubMatrix, std::shared_ptr<dolfin::BlockPETScSubMatrix>, dolfin::PETScMatrix>
      (m, "BlockPETScSubMatrix", "multiphenics BlockPETScSubMatrix object");
    #endif
      
    // dolfin::GenericBlockLinearAlgebraFactory
    py::class_<dolfin::GenericBlockLinearAlgebraFactory, std::shared_ptr<dolfin::GenericBlockLinearAlgebraFactory>, dolfin::GenericLinearAlgebraFactory>
      (m, "GenericBlockLinearAlgebraFactory", "multiphenics GenericBlockLinearAlgebraFactory object");
      
    // dolfin::BlockDefaultFactory
    py::class_<dolfin::BlockDefaultFactory, std::shared_ptr<dolfin::BlockDefaultFactory>, dolfin::GenericBlockLinearAlgebraFactory>
      (m, "BlockDefaultFactory", "multiphenics BlockDefaultFactory object");
    
    #ifdef HAS_PETSC
    // dolfin::BlockPETScFactory
    py::class_<dolfin::BlockPETScFactory, std::shared_ptr<dolfin::BlockPETScFactory>, dolfin::GenericBlockLinearAlgebraFactory>
      (m, "BlockPETScFactory", "multiphenics BlockPETScFactory object");
    #endif
    
    #ifdef HAS_SLEPC
    // dolfin::CondensedSLEPcEigenSolver
    py::class_<dolfin::CondensedSLEPcEigenSolver, std::shared_ptr<dolfin::CondensedSLEPcEigenSolver>, dolfin::SLEPcEigenSolver>
      (m, "CondensedSLEPcEigenSolver", "multiphenics CondensedSLEPcEigenSolver object")
      .def(py::init<std::shared_ptr<const dolfin::PETScMatrix>,
                    std::vector<std::shared_ptr<const dolfin::DirichletBC>>>())
      .def(py::init<std::shared_ptr<const dolfin::PETScMatrix>, std::shared_ptr<const dolfin::PETScMatrix>,
                    std::vector<std::shared_ptr<const dolfin::DirichletBC>>>());
    
    // dolfin::CondensedBlockSLEPcEigenSolver
    py::class_<dolfin::CondensedBlockSLEPcEigenSolver, std::shared_ptr<dolfin::CondensedBlockSLEPcEigenSolver>, dolfin::CondensedSLEPcEigenSolver>
      (m, "CondensedBlockSLEPcEigenSolver", "multiphenics CondensedBlockSLEPcEigenSolver object")
      .def(py::init<std::shared_ptr<const dolfin::BlockPETScMatrix>,
                    std::shared_ptr<const dolfin::BlockDirichletBC>>())
      .def(py::init<std::shared_ptr<const dolfin::BlockPETScMatrix>, std::shared_ptr<const dolfin::BlockPETScMatrix>,
                    std::shared_ptr<const dolfin::BlockDirichletBC>>());
    #endif
  }
}

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
      (m, "GenericBlockVector", "multiphenics GenericBlockVector object");
      
    // multiphenics::GenericBlockMatrix
    py::class_<multiphenics::GenericBlockMatrix, std::shared_ptr<multiphenics::GenericBlockMatrix>>
      (m, "GenericBlockMatrix", "multiphenics GenericBlockMatrix object");
      
    #ifdef HAS_PETSC
    // multiphenics::BlockPETScVector
    py::class_<multiphenics::BlockPETScVector, std::shared_ptr<multiphenics::BlockPETScVector>, multiphenics::GenericBlockVector, dolfin::PETScVector>
      (m, "BlockPETScVector", "multiphenics BlockPETScVector object");
      
    // multiphenics::BlockPETScMatrix
    py::class_<multiphenics::BlockPETScMatrix, std::shared_ptr<multiphenics::BlockPETScMatrix>, multiphenics::GenericBlockMatrix, dolfin::PETScMatrix>
      (m, "BlockPETScMatrix", "multiphenics BlockPETScMatrix object");
      
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
      (m, "BlockDefaultFactory", "multiphenics BlockDefaultFactory object");
    
    #ifdef HAS_PETSC
    // multiphenics::BlockPETScFactory
    py::class_<multiphenics::BlockPETScFactory, std::shared_ptr<multiphenics::BlockPETScFactory>, multiphenics::GenericBlockLinearAlgebraFactory>
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
    
    // multiphenics::CondensedBlockSLEPcEigenSolver
    py::class_<multiphenics::CondensedBlockSLEPcEigenSolver, std::shared_ptr<multiphenics::CondensedBlockSLEPcEigenSolver>, dolfin::CondensedSLEPcEigenSolver>
      (m, "CondensedBlockSLEPcEigenSolver", "multiphenics CondensedBlockSLEPcEigenSolver object")
      .def(py::init<std::shared_ptr<const multiphenics::BlockPETScMatrix>,
                    std::shared_ptr<const multiphenics::BlockDirichletBC>>())
      .def(py::init<std::shared_ptr<const multiphenics::BlockPETScMatrix>, std::shared_ptr<const multiphenics::BlockPETScMatrix>,
                    std::shared_ptr<const multiphenics::BlockDirichletBC>>());
    #endif
  }
}

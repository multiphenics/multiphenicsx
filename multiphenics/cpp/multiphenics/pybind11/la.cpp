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
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include <dolfinx/function/Function.h>
#include <dolfinx/la/SLEPcEigenSolver.h>
#include <dolfinx/la/utils.h>
#include <caster_mpi.h>
#include <caster_petsc.h>
#include <multiphenics/function/BlockFunction.h>
#include <multiphenics/la/CondensedBlockSLEPcEigenSolver.h>
#include <multiphenics/la/CondensedSLEPcEigenSolver.h>

namespace py = pybind11;

namespace multiphenics_wrappers
{
  void la(py::module& m)
  {
    #ifdef HAS_SLEPC
    // dolfinx::la::SLEPcEigenSolver // TODO remove when it gets wrapped upstream, but be careful with the different signature of get_eigenpair
    py::class_<dolfinx::la::SLEPcEigenSolver, std::shared_ptr<dolfinx::la::SLEPcEigenSolver>>
      (m, "SLEPcEigenSolver", "DOLFIN-X SLEPcEigenSolver object")
      .def(py::init([](const dolfinx_wrappers::MPICommWrapper comm) {
        return std::make_unique<dolfinx::la::SLEPcEigenSolver>(comm.get());
      }))
      .def("set_options_prefix", &dolfinx::la::SLEPcEigenSolver::set_options_prefix)
      .def("set_from_options", &dolfinx::la::SLEPcEigenSolver::set_from_options)
      .def("set_operators", &dolfinx::la::SLEPcEigenSolver::set_operators)
      .def("get_options_prefix", &dolfinx::la::SLEPcEigenSolver::get_options_prefix)
      .def("get_number_converged", &dolfinx::la::SLEPcEigenSolver::get_number_converged)
      .def("solve", (void (dolfinx::la::SLEPcEigenSolver::*)())
           &dolfinx::la::SLEPcEigenSolver::solve)
      .def("solve", (void (dolfinx::la::SLEPcEigenSolver::*)(std::int64_t))
           &dolfinx::la::SLEPcEigenSolver::solve)
      .def("get_eigenvalue", &dolfinx::la::SLEPcEigenSolver::get_eigenvalue)
      .def("get_eigenpair", [](dolfinx::la::SLEPcEigenSolver& self, dolfinx::function::Function& r, dolfinx::function::Function& c, std::size_t i)
           {
             PetscScalar lr, lc;
             self.get_eigenpair(lr, lc, r.vector().vec(), c.vector().vec(), i);
             PetscErrorCode ierr;
             ierr = VecGhostUpdateBegin(r.vector().vec(), INSERT_VALUES, SCATTER_FORWARD);
             if (ierr != 0) dolfinx::la::petsc_error(ierr, __FILE__, "VecGhostUpdateBegin");
             ierr = VecGhostUpdateBegin(c.vector().vec(), INSERT_VALUES, SCATTER_FORWARD);
             if (ierr != 0) dolfinx::la::petsc_error(ierr, __FILE__, "VecGhostUpdateBegin");
             ierr = VecGhostUpdateEnd(r.vector().vec(), INSERT_VALUES, SCATTER_FORWARD);
             if (ierr != 0) dolfinx::la::petsc_error(ierr, __FILE__, "VecGhostUpdateEnd");
             ierr = VecGhostUpdateEnd(c.vector().vec(), INSERT_VALUES, SCATTER_FORWARD);
             if (ierr != 0) dolfinx::la::petsc_error(ierr, __FILE__, "VecGhostUpdateEnd");
             return py::make_tuple(lr, lc);
           });

    // dolfinx::la::CondensedSLEPcEigenSolver
    py::class_<dolfinx::la::CondensedSLEPcEigenSolver, std::shared_ptr<dolfinx::la::CondensedSLEPcEigenSolver>, dolfinx::la::SLEPcEigenSolver>
      (m, "CondensedSLEPcEigenSolver", "multiphenics CondensedSLEPcEigenSolver object")
      .def(py::init([](const dolfinx_wrappers::MPICommWrapper comm) {
        return std::make_unique<dolfinx::la::CondensedSLEPcEigenSolver>(comm.get());
      }))
      .def("set_operators", &dolfinx::la::CondensedSLEPcEigenSolver::set_operators)
      .def("set_boundary_conditions", &dolfinx::la::CondensedSLEPcEigenSolver::set_boundary_conditions)
      .def("get_eigenpair", [](dolfinx::la::CondensedSLEPcEigenSolver& self, dolfinx::function::Function& r, dolfinx::function::Function& c, std::size_t i)
           {
             PetscScalar lr, lc;
             self.get_eigenpair(lr, lc, r.vector().vec(), c.vector().vec(), i);
             PetscErrorCode ierr;
             ierr = VecGhostUpdateBegin(r.vector().vec(), INSERT_VALUES, SCATTER_FORWARD);
             if (ierr != 0) dolfinx::la::petsc_error(ierr, __FILE__, "VecGhostUpdateBegin");
             ierr = VecGhostUpdateBegin(c.vector().vec(), INSERT_VALUES, SCATTER_FORWARD);
             if (ierr != 0) dolfinx::la::petsc_error(ierr, __FILE__, "VecGhostUpdateBegin");
             ierr = VecGhostUpdateEnd(r.vector().vec(), INSERT_VALUES, SCATTER_FORWARD);
             if (ierr != 0) dolfinx::la::petsc_error(ierr, __FILE__, "VecGhostUpdateEnd");
             ierr = VecGhostUpdateEnd(c.vector().vec(), INSERT_VALUES, SCATTER_FORWARD);
             if (ierr != 0) dolfinx::la::petsc_error(ierr, __FILE__, "VecGhostUpdateEnd");
             return py::make_tuple(lr, lc);
           });

    // multiphenics::la::CondensedBlockSLEPcEigenSolver
    py::class_<multiphenics::la::CondensedBlockSLEPcEigenSolver, std::shared_ptr<multiphenics::la::CondensedBlockSLEPcEigenSolver>, dolfinx::la::CondensedSLEPcEigenSolver>
      (m, "CondensedBlockSLEPcEigenSolver", "multiphenics CondensedBlockSLEPcEigenSolver object")
      .def(py::init([](const dolfinx_wrappers::MPICommWrapper comm) {
        return std::make_unique<multiphenics::la::CondensedBlockSLEPcEigenSolver>(comm.get());
      }))
      .def("set_operators", &multiphenics::la::CondensedBlockSLEPcEigenSolver::set_operators)
      .def("set_boundary_conditions", (void (multiphenics::la::CondensedBlockSLEPcEigenSolver::*)(std::shared_ptr<const multiphenics::fem::BlockDirichletBC>))
           &multiphenics::la::CondensedBlockSLEPcEigenSolver::set_boundary_conditions)
      .def("get_eigenpair", [](multiphenics::la::CondensedBlockSLEPcEigenSolver& self, multiphenics::function::BlockFunction& r, multiphenics::function::BlockFunction& c, std::size_t i)
           {
             PetscScalar lr, lc;
             self.get_eigenpair(lr, lc, r.block_vector(), c.block_vector(), i);
             PetscErrorCode ierr;
             ierr = VecGhostUpdateBegin(r.block_vector(), INSERT_VALUES, SCATTER_FORWARD);
             if (ierr != 0) dolfinx::la::petsc_error(ierr, __FILE__, "VecGhostUpdateBegin");
             ierr = VecGhostUpdateBegin(c.block_vector(), INSERT_VALUES, SCATTER_FORWARD);
             if (ierr != 0) dolfinx::la::petsc_error(ierr, __FILE__, "VecGhostUpdateBegin");
             ierr = VecGhostUpdateEnd(r.block_vector(), INSERT_VALUES, SCATTER_FORWARD);
             if (ierr != 0) dolfinx::la::petsc_error(ierr, __FILE__, "VecGhostUpdateEnd");
             ierr = VecGhostUpdateEnd(c.block_vector(), INSERT_VALUES, SCATTER_FORWARD);
             if (ierr != 0) dolfinx::la::petsc_error(ierr, __FILE__, "VecGhostUpdateEnd");
             r.apply("to subfunctions");
             c.apply("to subfunctions");
             return py::make_tuple(lr, lc);
           });
    #endif
  }
}

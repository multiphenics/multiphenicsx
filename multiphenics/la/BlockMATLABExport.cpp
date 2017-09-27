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

#include <dolfin/la/PETScObject.h>
#include <multiphenics/la/BlockMATLABExport.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void BlockMATLABExport::export_(const PETScMatrix & A, std::string A_filename)
{
  PetscErrorCode ierr;
  PetscViewer view_out;
  ierr = PetscViewerASCIIOpen(A.mpi_comm(), (A_filename + ".m").c_str(), &view_out);
  if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "PetscViewerASCIIOpen");
  
  ierr = PetscViewerPushFormat(view_out, PETSC_VIEWER_ASCII_MATLAB);
  if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "PetscViewerPushFormat");

  ierr = MatView(A.mat(), view_out);
  if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "MatView");
  
  ierr = PetscViewerPopFormat(view_out);
  if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "PetscViewerPopFormat");

  ierr = PetscViewerDestroy(&view_out);
  if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "PetscViewerDestroy");
}
//-----------------------------------------------------------------------------
void BlockMATLABExport::export_(const PETScVector & b, std::string b_filename)
{
  PetscErrorCode ierr;
  PetscViewer view_out;
  ierr = PetscViewerASCIIOpen(b.mpi_comm(), (b_filename + ".m").c_str(), &view_out);
  if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "PetscViewerASCIIOpen");
  
  ierr = PetscViewerPushFormat(view_out, PETSC_VIEWER_ASCII_MATLAB);
  if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "PetscViewerPushFormat");

  ierr = VecView(b.vec(), view_out);
  if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "VecView");
  
  ierr = PetscViewerPopFormat(view_out);
  if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "PetscViewerPopFormat");

  ierr = PetscViewerDestroy(&view_out);
  if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "PetscViewerDestroy");
}
//-----------------------------------------------------------------------------

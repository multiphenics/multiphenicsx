// Copyright (C) 2016-2021 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <map>
#include <petscmat.h>

namespace multiphenicsx::la
{

/// Wrapper around a local submatrix of a Mat object, used in combination with DofMapRestriction
class MatSubMatrixWrapper
{
public:
  /// Constructor (for cases without restriction)
  MatSubMatrixWrapper(Mat A,
                      std::array<IS, 2> index_sets),

  /// Constructor (for cases with restriction)
  MatSubMatrixWrapper(Mat A,
                      std::array<IS, 2> unrestricted_index_sets,
                      std::array<IS, 2> restricted_index_sets,
                      std::array<std::map<std::int32_t, std::int32_t>, 2> unrestricted_to_restricted,
                      std::array<int, 2> unrestricted_to_restricted_bs);

  /// Destructor
  ~MatSubMatrixWrapper();

  /// Copy constructor (deleted)
  MatSubMatrixWrapper(const MatSubMatrixWrapper& A) = delete;

  /// Move constructor (deleted)
  MatSubMatrixWrapper(MatSubMatrixWrapper&& A) = delete;

  /// Assignment operator (deleted)
  MatSubMatrixWrapper& operator=(const MatSubMatrixWrapper& A) = delete;

  /// Move assignment operator (deleted)
  MatSubMatrixWrapper& operator=(MatSubMatrixWrapper&& A) = delete;

  /// Restore PETSc Mat object
  void restore();

  /// Pointer to submatrix
  Mat mat() const;
private:
  Mat _global_matrix;
  Mat _sub_matrix;
  std::array<IS, 2> _is;
};
} // namespace multiphenicsx::la

// Copyright (C) 2016-2023 by the multiphenicsx authors
//
// This file is part of multiphenicsx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <set>
#include <dolfinx/fem/Form.h>

namespace multiphenicsx
{

namespace fem
{

using dolfinx::fem::IntegralType;

/// Extract form integral types from a Form
template <typename T, std::floating_point U>
std::set<fem::IntegralType> get_integral_types_from_form(const dolfinx::fem::Form<T, U>& form)
{
  std::set<fem::IntegralType> all_integral_types{{
    fem::IntegralType::cell, fem::IntegralType::exterior_facet,
    fem::IntegralType::interior_facet
  }};
  std::set<fem::IntegralType> integral_types;
  for (auto& integral_type : all_integral_types)
      if (form.num_integrals(integral_type) > 0)
          integral_types.insert(integral_type);
  return integral_types;
}

} // namespace fem
} // namespace multiphenicsx

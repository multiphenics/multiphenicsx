# Copyright (C) 2016-2020 by the multiphenics authors
#
# This file is part of multiphenics.
#
# multiphenics is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

from ufl import Form
from dolfinx.cpp.fem import Form as cpp_Form
from multiphenics.cpp.compile_code import compile_code

zeros = (0, 0.)

def block_replace_zero(block_form, index, block_function_space):
    assert len(index) in (1, 2)
    if len(index) == 2:
        I = index[0]  # noqa: E741
        J = index[1]
        assert (
            isinstance(block_form[I][J], Form)
                or
            (isinstance(block_form[I][J], (float, int)) and block_form[I][J] in zeros)
        )
        if _is_zero(block_form[I][J]):
            return _get_zero_form(block_function_space, (I, J))
        else:
            return block_form[I][J]
    else:
        I = index[0]  # noqa: E741
        assert (
            isinstance(block_form[I], Form)
                or
            (isinstance(block_form[I], (float, int)) and block_form[I] in zeros)
        )
        if _is_zero(block_form[I]):
            return _get_zero_form(block_function_space, (I, ))
        else:
            return block_form[I]

def _is_zero(form_or_block_form):
    assert (
        isinstance(form_or_block_form, (cpp_Form, Form, list))
            or
        (isinstance(form_or_block_form, (float, int)) and form_or_block_form in zeros)
    )
    if isinstance(form_or_block_form, Form):
        return form_or_block_form.empty()
    elif isinstance(form_or_block_form, cpp_Form):
        _is_zero_form_cpp_code = """
            #include <dolfinx/fem/Form.h>
            #include <pybind11/pybind11.h>

            bool is_zero_form(std::shared_ptr<dolfinx::fem::Form> form)
            {
              return (
                form->integrals().num_integrals(dolfinx::fem::FormIntegrals::Type::cell) == 0
                    &&
                form->integrals().num_integrals(dolfinx::fem::FormIntegrals::Type::interior_facet) == 0
                    &&
                form->integrals().num_integrals(dolfinx::fem::FormIntegrals::Type::exterior_facet) == 0
              );
            }

            PYBIND11_MODULE(SIGNATURE, m)
            {
                m.def("is_zero_form", &is_zero_form);
            }
            """
        is_zero_form = compile_code("is_zero_form", _is_zero_form_cpp_code).is_zero_form
        return is_zero_form(form_or_block_form)
    elif isinstance(form_or_block_form, (float, int)) and form_or_block_form in zeros:
        return True
    else:
        raise AssertionError("Invalid case in _is_zero")

def _get_zero_form(block_function_space, index):
    assert len(index) in (1, 2)
    if len(index) == 2:
        return cpp_Form([block_function_space[0][index[0]]._cpp_object, block_function_space[1][index[1]]._cpp_object])
    else:
        return cpp_Form([block_function_space[0][index[0]]._cpp_object])

# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for multiphenicsx.io.compile_code module."""

import os

import dolfinx.jit
import mpi4py
import nbvalx.tempfile

import multiphenicsx.cpp
import multiphenicsx.io


def test_compile_code() -> None:
    """Compile a simple C++ function."""
    comm = mpi4py.MPI.COMM_WORLD

    with nbvalx.tempfile.TemporaryDirectory(comm) as tempdir:
        if comm.rank == 0:
            code = """
    #include <pybind11/pybind11.h>

    int multiply(int a, int b) {
        return a * b;
    }

    PYBIND11_MODULE(SIGNATURE, m)
    {
        m.def("multiply", &multiply);
    }
    """
            filename = os.path.join(tempdir, "test_compile_code_source.cpp")
            open(filename, "w").write(code)
            filename = comm.bcast(filename, root=0)
        else:
            filename = comm.bcast(None, root=0)

        compile_code = dolfinx.jit.mpi_jit_decorator(multiphenicsx.cpp.compile_code)
        print(comm.rank, os.path.exists(tempdir), os.path.exists(os.path.join(tempdir, "test_compile_code_source.cpp")))
        cpp_library = compile_code(comm, "test_compile_code", filename, output_dir=tempdir)
        assert cpp_library.multiply(2, 3) == 6


def test_compile_package() -> None:
    """Compile a simple C++ package."""
    comm = mpi4py.MPI.COMM_WORLD

    with nbvalx.tempfile.TemporaryDirectory(comm) as tempdir:
        if comm.rank == 0:
            package_root = os.path.join(tempdir, "test_compile_package")
            os.makedirs(os.path.join(package_root, "utilities"))
            os.makedirs(os.path.join(package_root, "wrappers"))

            multiply_header_code = """
    namespace utilities
    {
        int multiply(int a, int b);
    }
    """
            multiply_header_file = os.path.join("utilities", "multiply.h")
            open(os.path.join(package_root, multiply_header_file), "w").write(multiply_header_code)

            multiply_source_code = """
    #include <test_compile_package/utilities/multiply.h>

    int utilities::multiply(int a, int b)
    {
        return a * b;
    }
    """
            multiply_source_file = os.path.join("utilities", "multiply.cpp")
            open(os.path.join(package_root, multiply_source_file), "w").write(multiply_source_code)

            utilities_wrapper_code = """
    #include <pybind11/pybind11.h>

    #include <test_compile_package/utilities/multiply.h>

    namespace py = pybind11;

    namespace wrappers
    {
        void utilities(py::module& m)
        {
            m.def("multiply", &utilities::multiply);
        }
    }
    """
            utilities_wrapper_file = os.path.join("wrappers", "utilities.cpp")
            open(os.path.join(package_root, utilities_wrapper_file), "w").write(utilities_wrapper_code)

            main_wrapper_code = """
    #include <pybind11/pybind11.h>

    namespace py = pybind11;

    namespace wrappers
    {
        void utilities(py::module& m);
    }

    PYBIND11_MODULE(SIGNATURE, m)
    {
        py::module utilities = m.def_submodule("utilities", "utilities module");
        wrappers::utilities(utilities);
    }
    """
            main_wrapper_file = os.path.join("wrappers", "test_compile_package.cpp")
            open(os.path.join(package_root, main_wrapper_file), "w").write(main_wrapper_code)

            multiply_source_file = comm.bcast(multiply_source_file, root=0)
        else:
            multiply_source_file = comm.bcast(None, root=0)
        sources = [multiply_source_file]

        cpp_library = multiphenicsx.cpp.compile_package(
            comm, "test_compile_package", tempdir, *sources, output_dir=tempdir)
        assert cpp_library.utilities.multiply(2, 3) == 6

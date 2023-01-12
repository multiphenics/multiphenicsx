# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compile code in a C++ package using cppimport."""

import hashlib
import os
import sys
import types
import typing

import cppimport
import dolfinx.jit
import dolfinx.pkgconfig
import dolfinx.wrappers
import numpy as np
import petsc4py.PETSc


def compile_code(
    package_name: str, package_file: str, **kwargs: typing.Union[str, typing.List[str]]
) -> types.ModuleType:
    """Compile code in a C++ package."""
    dolfinx_pc = dict()
    has_petsc_complex = np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating)
    for (dolfinx_pc_package, scalar_type_check) in zip(
        ("dolfinx", "dolfinx_real", "dolfinx_complex"),
        (True, not has_petsc_complex, has_petsc_complex)
    ):
        if dolfinx.pkgconfig.exists(dolfinx_pc_package) and scalar_type_check:  # type: ignore[no-untyped-call]
            dolfinx_pc.update(dolfinx.pkgconfig.parse(dolfinx_pc_package))  # type: ignore[no-untyped-call]
            break
    assert len(dolfinx_pc) > 0

    # Set other sources
    sources = kwargs.get("sources", [])

    # Set include file dependencies
    dependencies = kwargs.get("dependencies", [])

    # Set include dirs
    include_dirs = kwargs.get("include_dirs", [])
    include_dirs.extend(dolfinx_pc["include_dirs"])
    include_dirs.append(str(dolfinx.wrappers.get_include_path()))

    # Set compiler arguments
    compiler_args = kwargs.get("compiler_args", [])
    compiler_args.append("-std=c++17")
    compiler_args.extend("-D" + dm for dm in dolfinx_pc["define_macros"] if "-NOTFOUND" not in dm)

    # Set libraries
    libraries = kwargs.get("libraries", [])
    libraries.extend(dolfinx_pc["libraries"])

    # Set library directories
    library_dirs = kwargs.get("library_dirs", [])
    library_dirs.extend(dolfinx_pc["library_dirs"])

    # Set linker arguments
    linker_args = kwargs.get("linker_args", [])

    # Set output directory
    jit_parameters = dolfinx.jit.get_parameters()
    output_dir = kwargs.get("output_dir", str(jit_parameters["cache_dir"]))

    # Prepare cpp import file
    package_cppimport_code = f"""
/*
<%
setup_pybind11(cfg)
cfg["sources"] += {str(sources)}
cfg["dependencies"] += {str(dependencies)}
cfg["include_dirs"] += {str(include_dirs)}
cfg["compiler_args"] += {str(compiler_args)}
cfg["libraries"] += {str(libraries)}
cfg["library_dirs"] += {str(library_dirs)}
cfg["linker_args"] += {str(linker_args)}
%>
*/
"""

    # Read in content of main package file
    package_code = open(package_file).read()

    # Compute hash from package code
    package_hash = hashlib.md5(package_code.encode("utf-8")).hexdigest()
    package_name_with_hash = package_name + "_" + package_hash

    # Write to output directory
    os.makedirs(output_dir, exist_ok=True)
    open(
        os.path.join(output_dir, package_name_with_hash + ".cpp"), "w"
    ).write(
        package_cppimport_code + package_code.replace("SIGNATURE", package_name_with_hash)
    )

    # Append output directory to path
    sys.path.append(output_dir)

    # Return module generated by cppimport
    return cppimport.imp(package_name_with_hash)

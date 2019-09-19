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

import glob
import mpi4py
import os
import petsc4py
from dolfin.jit import mpi_jit_decorator
from multiphenics.cpp.compile_code import compile_code

@mpi_jit_decorator
def compile_package(package_name, package_root, *args):
    # Remove extension from files
    files = [os.path.splitext(f)[0] for f in args]
    
    # Make sure that there are no duplicate files
    assert len(files) == len(set(files)), "There seems to be duplicate files. Make sure to include in the list only *.cpp files, not *.h ones."
    
    # Extract files
    package_headers = [os.path.join(package_root, package_name, f + ".h") for f in files]
    package_sources = [os.path.join(package_root, package_name, f + ".cpp") for f in files]
    assert len(package_headers) == len(package_sources)
    
    # Make sure that there are no files missing
    for (extension, typename, files_to_check) in zip(["h", "cpp"], ["headers", "sources"], [package_headers, package_sources]):
        all_package_files = set(glob.glob(os.path.join(package_root, package_name, "[!pybind11]*", "*." + extension)))
        sorted_package_files = set(files_to_check)
        if len(sorted_package_files) > len(all_package_files):
            raise AssertionError("Input " + typename + " list contains more files than ones present in the library. The files " + str(sorted_package_files - all_package_files) + " seem not to exist.")
        elif len(sorted_package_files) < len(all_package_files):
            raise AssertionError("Input " + typename + " list is not complete. The files " + str(all_package_files - sorted_package_files) + " are missing.")
        else:
            assert sorted_package_files == all_package_files, "Input " + typename + " list contains different files than ones present in the library. The files " + str(sorted_package_files - all_package_files) + " seem not to exist."
            
    # Extract submodules
    package_submodules = set([os.path.relpath(f, os.path.join(package_root, package_name)) for f in glob.glob(os.path.join(package_root, package_name, "[!_]*/"))])
    package_submodules.remove("pybind11")
    package_submodules = sorted(package_submodules)
    
    # Extract pybind11 files corresponding to each submodule
    package_pybind11_sources = list()
    for package_submodule in package_submodules:
        package_pybind11_sources.append(os.path.join(package_root, package_name, "pybind11", package_submodule + ".cpp"))
    if os.path.isfile(os.path.join(package_root, package_name, "pybind11", "MPICommWrapper.cpp")):
        package_pybind11_sources.append(os.path.join(package_root, package_name, "pybind11", "MPICommWrapper.cpp")) # TODO remove local copy of DOLFIN's pybind11 files
    
    # Read in content of main package file
    package_code = open(os.path.join(package_root, package_name, "pybind11", package_name + ".cpp")).read()
    
    # Prepare kwargs to be passed to cppimport
    kwargs = dict()
    
    # Setup sources for compilation
    kwargs["sources"] = package_sources + package_pybind11_sources
    
    # Setup headers for compilation
    kwargs["dependencies"] = package_headers
    
    # Setup include directories for compilation
    include_dirs = []
    include_dirs.append(package_root)
    include_dirs.append(mpi4py.get_include())
    include_dirs.append(petsc4py.get_include())
    kwargs["include_dirs"] = include_dirs
    
    # Compile C++ module and return
    return compile_code(package_name, package_code, **kwargs)

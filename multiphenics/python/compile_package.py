# Copyright (C) 2016-2018 by the multiphenics authors
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

import os
import glob
import mpi4py
import petsc4py
import dolfin
from dolfin import compile_cpp_code

def compile_package(package_name, package_root, *args, **kwargs):
    # Remove extension from files
    files = [os.path.splitext(f)[0] for f in args]
    
    # Make sure that there are no duplicate files
    assert len(files) == len(set(files)), "There seems to be duplicate files. Make sure to include in the list only *.cpp files, not *.h ones."
    
    # Extract folders
    package_folder = os.path.join(package_root, package_name)
    
    # Extract files
    package_headers = [os.path.join(package_folder, f + ".h") for f in files]
    package_sources = [os.path.join(package_folder, f + ".cpp") for f in files]
    assert len(package_headers) == len(package_sources)
    
    # Make sure that there are no files missing
    for (extension, typename, files_to_check) in zip(["h", "cpp"], ["headers", "sources"], [package_headers, package_sources]):
        all_package_files = set(glob.glob(os.path.join(package_folder, "[!python]*", "*." + extension)))
        sorted_package_files = set(files_to_check)
        if len(sorted_package_files) > len(all_package_files):
            raise AssertionError("Input " + typename + " list contains more files than ones present in the library. The files " + str(sorted_package_files - all_package_files) + " seem not to exist.")
        elif len(sorted_package_files) < len(all_package_files):
            raise AssertionError("Input " + typename + " list is not complete. The files " + str(all_package_files - sorted_package_files) + " are missing.")
        else:
            assert sorted_package_files == all_package_files, "Input " + typename + " list contains different files than ones present in the library. The files " + str(sorted_package_files - all_package_files) + " seem not to exist."
            
    # Extract submodules
    package_submodules = set([os.path.relpath(f, package_folder) for f in glob.glob(os.path.join(package_folder, "[!_]*"))])
    package_submodules.remove("python")
    package_submodules = sorted(package_submodules)
    
    # Extract pybind11 files corresponding to each submodule
    package_pybind11_sources = list()
    for package_submodule in package_submodules:
        package_pybind11_sources.append(os.path.join(package_folder, "python", package_submodule + ".cpp"))
    if os.path.isfile(os.path.join(package_folder, "python", "MPICommWrapper.cpp")):
        package_pybind11_sources.append(os.path.join(package_folder, "python", "MPICommWrapper.cpp")) # TODO remove local copy of DOLFIN's pybind11 files
    package_pybind11_sources.append(os.path.join(package_folder, "python", package_name + ".cpp"))
    
    # Read in the code
    package_code = ""
    package_code += "\n".join([open(h).read() for h in package_sources])
    package_code += "\n".join([open(h).read() for h in package_pybind11_sources])
    
    # Move all includes to the top
    package_code_includes = ""
    package_code_rest = ""
    for line in package_code.splitlines():
        if line.startswith("#include"):
            package_code_includes += line + "\n"
        else:
            package_code_rest += line + "\n"
    package_code = package_code_includes + package_code_rest
    
    # Patch dijitso
    include_dirs = list()
    include_dirs.append(package_root)
    if "include_dirs" in kwargs:
        include_dirs.extend(kwargs["include_dirs"])
    patch_dijitso(package_name, include_dirs)
    
    # Call DOLFIN's compile_cpp_code
    cpp = compile_cpp_code(package_code)
    
    # Restore original dijitso configuration
    undo_patch_dijitso()
    
    # Return compiled module
    return cpp
    
original_dijitso_jit = dolfin.jit.pybind11jit.dijitso_jit

def patch_dijitso(package_name, include_dirs):
    def dijitso_jit(jitable, name, params, generate=None, send=None, receive=None, wait=None):
        name = name.replace("dolfin", package_name)
        params["build"]["include_dirs"].append(mpi4py.get_include())
        params["build"]["include_dirs"].append(petsc4py.get_include())
        params["build"]["include_dirs"].extend(include_dirs)
        return original_dijitso_jit(jitable, name, params, generate, send, receive, wait)
    dolfin.jit.pybind11jit.dijitso_jit = dijitso_jit
    
def undo_patch_dijitso():
    dolfin.jit.pybind11jit.dijitso_jit = original_dijitso_jit

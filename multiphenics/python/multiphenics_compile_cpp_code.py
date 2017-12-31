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
import dolfin
from dolfin import compile_cpp_code

def multiphenics_compile_cpp_code(package_name, *args):
    # Remove extension from files
    files = [os.path.splitext(f)[0] for f in args]
    
    # Make sure that there are no duplicate files
    assert len(files) == len(set(files)), "There seems to be duplicate files. Make sure to include in the list only *.cpp files, not *.h ones."
    
    # Extract folders
    multiphenics_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    multiphenics_folder = os.path.join(multiphenics_root, package_name)
    
    # Extract files
    multiphenics_headers = [os.path.join(multiphenics_folder, f + ".h") for f in files]
    multiphenics_sources = [os.path.join(multiphenics_folder, f + ".cpp") for f in files]
    assert len(multiphenics_headers) == len(multiphenics_sources)
    
    # Make sure that there are no files missing
    for (extension, typename, multiphenics_list_to_check) in zip(["h", "cpp"], ["headers", "sources"], [multiphenics_headers, multiphenics_sources]):
        all_multiphenics_files = set(glob.glob(os.path.join(multiphenics_folder, "[!python]*", "*." + extension)))
        sorted_multiphenics_files = set(multiphenics_list_to_check)
        if len(sorted_multiphenics_files) > len(all_multiphenics_files):
            raise AssertionError("Input " + typename + " list contains more files than ones present in the library. The files " + str(sorted_multiphenics_files - all_multiphenics_files) + " seem not to exist.")
        elif len(sorted_multiphenics_files) < len(all_multiphenics_files):
            raise AssertionError("Input " + typename + " list is not complete. The files " + str(all_multiphenics_files - sorted_multiphenics_files) + " are missing.")
        else:
            assert sorted_multiphenics_files == all_multiphenics_files, "Input " + typename + " list contains different files than ones present in the library. The files " + str(sorted_multiphenics_files - all_multiphenics_files) + " seem not to exist."
            
    # Extract submodules
    multiphenics_submodules = set([os.path.relpath(f, multiphenics_folder) for f in glob.glob(os.path.join(multiphenics_folder, "[!_]*"))])
    multiphenics_submodules.remove("python")
    multiphenics_submodules = sorted(multiphenics_submodules)
    
    # Extract pybind11 files corresponding to each submodule
    multiphenics_pybind11_sources = list()
    for multiphenics_submodule in multiphenics_submodules:
        multiphenics_pybind11_sources.append(os.path.join(multiphenics_folder, "python", multiphenics_submodule + ".cpp"))
    multiphenics_pybind11_sources.append(os.path.join(multiphenics_folder, "python", "MPICommWrapper.cpp")) # TODO remove local copy of DOLFIN's pybind11 files
    multiphenics_pybind11_sources.append(os.path.join(multiphenics_folder, "python", package_name + ".cpp"))
    
    # Read in the code
    multiphenics_code = ""
    multiphenics_code += "\n".join([open(h).read() for h in multiphenics_sources])
    multiphenics_code += "\n".join([open(h).read() for h in multiphenics_pybind11_sources])
    
    # Move all includes to the top
    multiphenics_code_includes = ""
    multiphenics_code_rest = ""
    for line in multiphenics_code.splitlines():
        if line.startswith("#include"):
            multiphenics_code_includes += line + "\n"
        else:
            multiphenics_code_rest += line + "\n"
    multiphenics_code = multiphenics_code_includes + multiphenics_code_rest
    
    # Patch dijitso
    patch_dijitso(multiphenics_root, package_name)
    
    # Call DOLFIN's compile_cpp_code
    cpp = compile_cpp_code(multiphenics_code)
    
    # Restore original dijitso configuration
    undo_patch_dijitso()
    
    # Return compiled module
    return cpp
    
original_dijitso_jit = dolfin.jit.pybind11jit.dijitso_jit

def patch_dijitso(multiphenics_root, package_name):
    def dijitso_jit(jitable, name, params, generate=None, send=None, receive=None, wait=None):
        name = name.replace("dolfin", package_name)
        params["build"]["include_dirs"].append(multiphenics_root)
        params["build"]["include_dirs"].append(mpi4py.get_include())
        return original_dijitso_jit(jitable, name, params, generate, send, receive, wait)
    dolfin.jit.pybind11jit.dijitso_jit = dijitso_jit
    
def undo_patch_dijitso():
    dolfin.jit.pybind11jit.dijitso_jit = original_dijitso_jit

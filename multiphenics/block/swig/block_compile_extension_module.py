# Copyright (C) 2016-2017 by the multiphenics authors
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

import sys
import os
import io
import glob
import hashlib
import instant
import ffc
from ufl.utils.sorting import canonicalize_metadata
import dolfin
from dolfin import compile_extension_module
from dolfin.compilemodules.compilemodule import _interface_version
from dolfin_utils.cppparser import parse_and_extract_type_info

def block_compile_extension_module(*args, **kwargs):
    # Remove extension from files
    files = [os.path.splitext(f)[0] for f in args]
    
    # Make sure that there are no duplicate files
    assert len(files) == len(set(files)), "There seems to be duplicate files. Make sure to include in the list only *.cpp files, not *.h ones."
    
    # Process additional declarations
    if "additional_declarations" in kwargs:
        additional_declarations = kwargs["additional_declarations"]
        assert isinstance(additional_declarations, dict)
        assert "pre" in additional_declarations
        assert "post" in additional_declarations
        # Store pre additional declarations in a format compatible with dolfin signature
        kwargs["additional_declarations"] = additional_declarations["pre"]
        # Store post additional declaration in global variable
        if isinstance(additional_declarations["post"], bytes):
            additional_declarations["post"] = additional_declarations["post"].decode("utf8")
        extended_write_interfacefile.additional_declarations__post = additional_declarations["post"]
    
    # Extract folders
    block_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    block_folder = os.path.join(block_root, "block")
    
    # Extract files
    block_headers = [os.path.join(block_folder, f + ".h") for f in files]
    block_sources = [os.path.join(block_folder, f + ".cpp") for f in files]
    assert len(block_headers) == len(block_sources)
    
    # Make sure that there are no files missing
    for (extension, typename, block_list_to_check) in zip(["h", "cpp"], ["headers", "sources"], [block_headers, block_sources]):
        all_block_files = set(glob.glob(os.path.join(block_folder, "*", "*." + extension)))
        sorted_block_files = set(block_list_to_check)
        if len(sorted_block_files) > len(all_block_files):
            raise AssertionError("Input " + typename + " list contains more files than ones present in the library. The files " + str(sorted_block_files - all_block_files) + " seem not to exist.")
        elif len(sorted_block_files) < len(all_block_files):
            raise AssertionError("Input " + typename + " list is not complete. The files " + str(all_block_files - sorted_block_files) + " are missing.")
        else:
            assert sorted_block_files == all_block_files, "Input " + typename + " list contains different files than ones present in the library. The files " + str(sorted_block_files - all_block_files) + " seem not to exist."
        
    # Read in the code
    block_code = "\n".join([open(h).read() for h in block_headers])
    
    # Make sure to force recompilation if
    block_last_edit = max(
        # 1. dolfin has changed
        os.path.getmtime(dolfin.__file__),
        # 2. any block header file has changed
        #    (make would be smarter, forcing recompilation only if headers included in current sources have changed)
        max([os.path.getmtime(h) for h in block_headers]),
        # 3. source files have been changed
        max([os.path.getmtime(s) for s in block_sources])
    )
    
    # Mimic the module name assignment as done in
    #     dolfin/compilemodules/compilemodule.py,
    # but forcing recompilation if library code
    # has been modified
    block_module_signature = hashlib.sha1(
                              (
                                repr(block_code) +
                                # Standard dolfin
                                dolfin.__version__ +
                                str(_interface_version) +
                                ffc.ufc_signature() +
                                sys.version +
                                repr(canonicalize_metadata(kwargs)) +
                                # Customization
                                str(block_last_edit)
                              ).encode("utf-8")
                             ).hexdigest()
    block_module_name = "block_" + block_module_signature
    
    # Instant requires source files to be relative to the source directory
    block_sources = [os.path.relpath(s, block_folder) for s in block_sources]
    
    # Patch Instant
    patch_instant()
    
    # Call DOLFIN's compile_extension_module
    cpp = compile_extension_module(
        code=block_code, 
        source_directory=block_folder,
        sources=block_sources,
        include_dirs=[block_root],
        module_name = block_module_name,
        **kwargs
    )
    
    # Restore original Instant configuration
    undo_patch_instant()
    
    # Return compiled module
    return cpp
    
# Extend instant copy files to allow subfolders
original_copy_files = instant.build.copy_files
def extended_copy_files(source, dest, files):
    # Get all subfolders
    files_in_subfolders = dict()
    for f in files:
        subfolder = os.path.dirname(f)
        assert os.path.dirname(subfolder) == "", "We only handle one level of subfolders"
        if subfolder not in files_in_subfolders:
            files_in_subfolders[subfolder] = list()
        file_ = os.path.basename(f)
        files_in_subfolders[subfolder].append(file_)
    # Recursively copy them
    for (subfolder, files_in_subfolder) in files_in_subfolders.iteritems():
        source_subfolder = os.path.join(source, subfolder)
        dest_subfolder = os.path.join(dest, subfolder)
        if not os.path.exists(dest_subfolder):
            os.makedirs(dest_subfolder)
        original_copy_files(source_subfolder, dest_subfolder, files_in_subfolder)

# Extend instant write interface file to allow both pre and post code
original_write_interfacefile = instant.build.write_interfacefile
def extended_write_interfacefile(filename, modulename, code, init_code,
                                 additional_definitions, additional_declarations__pre,
                                 system_headers, local_headers, wrap_headers, arrays):
    
    # Generate standard interface file                         
    assert isinstance(additional_declarations__pre, str)
    original_write_interfacefile(filename, modulename, code, init_code,
                                 additional_definitions, additional_declarations__pre,
                                 system_headers, local_headers, wrap_headers, arrays)
    
    # Read the written file
    with io.open(filename, "r", encoding="utf8") as f:
        content = f.read().splitlines()
        
    # Enable directors, replacing the first four lines of the file which contain
    # module definition without directors
    content_new = list()
    content_new.append(u'%module(package="multiphenics.block.swig.cpp", directors="1") ' + modulename)
    content_new.extend(content[4:])
    
    # Write back to file, also appending post declarations
    with io.open(filename, "w", encoding="utf8") as f:
        f.write("\n".join(content_new))
        f.write(extended_write_interfacefile.additional_declarations__post)
        f.flush()
extended_write_interfacefile.additional_declarations__post = None

def patch_instant():
    instant.build.copy_files = extended_copy_files
    instant.build.write_interfacefile = extended_write_interfacefile
    
def undo_patch_instant():
    instant.build.copy_files = original_copy_files
    instant.build.write_interfacefile = original_write_interfacefile
    

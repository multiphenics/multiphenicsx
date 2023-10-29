cmake_minimum_required(VERSION 3.18)

project(multiphenicsx)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find python
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Find pybind11
find_package(
    pybind11 2.9.1 REQUIRED
    CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT} $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT}
)

# Check for DOLFINx C++ backend
find_package(DOLFINX REQUIRED CONFIG)

# Check for DOLFINx python wrappers
execute_process(
    COMMAND ${Python3_EXECUTABLE}
    -c "import os, sys, dolfinx; print(os.path.join(os.path.dirname(dolfinx.__file__), 'wrappers'))"
    OUTPUT_VARIABLE DOLFINX_PY_WRAPPERS_DIR
    RESULT_VARIABLE DOLFINX_PY_WRAPPERS_COMMAND_RESULT
    ERROR_VARIABLE DOLFINX_PY_WRAPPERS_COMMAND_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT DOLFINX_PY_WRAPPERS_COMMAND_RESULT)
    message(STATUS "Found DOLFINx python wrappers at ${DOLFINX_PY_WRAPPERS_DIR}")
else()
    message(FATAL_ERROR "DOLFINx python wrappers could not be found.")
endif()

# Check for petsc4py
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import petsc4py; print(petsc4py.get_include())"
    OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
    RESULT_VARIABLE PETSC4PY_INCLUDE_COMMAND_RESULT
    ERROR_VARIABLE PETSC4PY_INCLUDE_COMMAND_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PETSC4PY_INCLUDE_COMMAND_RESULT)
    message(STATUS "Found petsc4py include directory at ${PETSC4PY_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "petsc4py could not be found.")
endif()

# Check for mpi4py
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import mpi4py; print(mpi4py.get_include())"
    OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
    RESULT_VARIABLE MPI4PY_INCLUDE_COMMAND_RESULT
    ERROR_VARIABLE MPI4PY_INCLUDE_COMMAND_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT MPI4PY_INCLUDE_COMMAND_RESULT)
    message(STATUS "Found mpi4py include directory at ${MPI4PY_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "mpi4py could not be found.")
endif()

# Compile multiphenicsx C++ backend and pybind11 wrappers
pybind11_add_module(
    multiphenicsx_cpp
    MODULE
    multiphenicsx/fem/DofMapRestriction.cpp
    multiphenicsx/fem/sparsitybuild.cpp
    multiphenicsx/fem/petsc.cpp
    multiphenicsx/fem/utils.cpp
    multiphenicsx/la/petsc.cpp
    multiphenicsx/wrappers/fem.cpp
    multiphenicsx/wrappers/la.cpp
    multiphenicsx/wrappers/multiphenicsx.cpp
)

# Add current source directory to include directories
target_include_directories(multiphenicsx_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Add DOLFINx, petsc4py and mpi4py include directories
target_include_directories(multiphenicsx_cpp PRIVATE ${DOLFINX_PY_WRAPPERS_DIR})
target_include_directories(multiphenicsx_cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})
target_include_directories(multiphenicsx_cpp PRIVATE ${MPI4PY_INCLUDE_DIR})

# Add DOLFINx libraries
target_link_libraries(multiphenicsx_cpp PRIVATE dolfinx)

# Install the compiled library to the cpp subdirectory
install(TARGETS multiphenicsx_cpp LIBRARY DESTINATION multiphenicsx/cpp)

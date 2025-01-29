cmake_minimum_required(VERSION 3.18)

project(multiphenicsx)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Report build type
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")

# Find python
find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED
)

# Find nanobind
execute_process(
  COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
  OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
  RESULT_VARIABLE NANOBIND_CMAKE_DIR_COMMAND_RESULT
  ERROR_VARIABLE NANOBIND_CMAKE_DIR_COMMAND_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT NANOBIND_CMAKE_DIR_COMMAND_RESULT)
  list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
  find_package(nanobind CONFIG REQUIRED)
  message(STATUS "Found nanobind python wrappers at ${NANOBIND_CMAKE_DIR}")
else()
  message(FATAL_ERROR "nanobind could not be found.")
endif()

# Check for DOLFINx C++ backend
find_package(DOLFINX REQUIRED CONFIG)
if(DOLFINX_FOUND)
  message(STATUS "Found DOLFINx C++ backend at ${DOLFINX_DIR}")
endif()

# Check for DOLFINx python wrappers
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import dolfinx; print(dolfinx.get_include())"
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
  COMMAND ${Python_EXECUTABLE} -c
          "import petsc4py; print(petsc4py.get_include())"
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
  COMMAND ${Python_EXECUTABLE} -c "import mpi4py; print(mpi4py.get_include())"
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

# Compile multiphenicsx C++ backend and nanobind wrappers
nanobind_add_module(
  multiphenicsx_cpp
  NOMINSIZE
  multiphenicsx/fem/DofMapRestriction.cpp
  multiphenicsx/fem/sparsitybuild.cpp
  multiphenicsx/la/petsc.cpp
  multiphenicsx/wrappers/fem.cpp
  multiphenicsx/wrappers/la.cpp
  multiphenicsx/wrappers/multiphenicsx.cpp
)

# Add DOLFINx C++ libraries
target_link_libraries(multiphenicsx_cpp PRIVATE dolfinx)

# Add DOLFINx python, petsc4py and mpi4py include directories (with DOLFINx C++
# ones already being added by target_link_libraries)
target_include_directories(multiphenicsx_cpp PRIVATE ${DOLFINX_PY_WRAPPERS_DIR})
target_include_directories(multiphenicsx_cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})
target_include_directories(multiphenicsx_cpp PRIVATE ${MPI4PY_INCLUDE_DIR})

# Add current source directory to include directories
target_include_directories(
  multiphenicsx_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# Define HAS_PETSC4PY for compatibility with DOLFINx python wrappers
target_compile_definitions(multiphenicsx_cpp PRIVATE HAS_PETSC4PY)

# Install the compiled library to the cpp subdirectory
set_target_properties(
  multiphenicsx_cpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE
)
install(TARGETS multiphenicsx_cpp LIBRARY DESTINATION multiphenicsx/cpp)

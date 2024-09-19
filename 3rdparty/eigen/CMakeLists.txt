cmake_minimum_required(VERSION 3.10.0)

#==============================================================================
# CMake Policy issues.
#==============================================================================
# Allow overriding options in a parent project via `set` before including Eigen.
if (POLICY CMP0077)
  cmake_policy (SET CMP0077 NEW)
endif (POLICY CMP0077)

# NOTE Remove setting the policy once the minimum required CMake version is
# increased to at least 3.15. Retain enabling the export to package registry.
if (POLICY CMP0090)
  # The export command does not populate package registry by default
  cmake_policy (SET CMP0090 NEW)
  # Unless otherwise specified, always export to package registry to ensure
  # backwards compatibility.
  if (NOT DEFINED CMAKE_EXPORT_PACKAGE_REGISTRY)
    set (CMAKE_EXPORT_PACKAGE_REGISTRY ON)
  endif (NOT DEFINED CMAKE_EXPORT_PACKAGE_REGISTRY)
endif (POLICY CMP0090)

# Disable warning about find_package(CUDA).
# CUDA language support is lacking for clang as the CUDA compiler
# until at least cmake version 3.18.  Even then, there seems to be
# issues on Windows+Ninja in passing build flags.  Continue using
# the "old" way for now.
if (POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif ()

#==============================================================================
# CMake Project.
#==============================================================================

project(Eigen3)

# Remove this block after bumping CMake to v3.21.0
# PROJECT_IS_TOP_LEVEL is defined then by default
if(CMAKE_VERSION VERSION_LESS 3.21.0)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(PROJECT_IS_TOP_LEVEL ON)
  else()
    set(PROJECT_IS_TOP_LEVEL OFF)
  endif()
endif()

#==============================================================================
# Build ON/OFF Settings.
#==============================================================================
# Determine if we should build tests.
include(CMakeDependentOption)
cmake_dependent_option(BUILD_TESTING "Enable creation of tests." ON "PROJECT_IS_TOP_LEVEL" OFF)
option(EIGEN_LEAVE_TEST_IN_ALL_TARGET "Leaves tests in the all target, needed by ctest for automatic building." OFF)

# Determine if we should build BLAS/LAPACK implementations.
option(EIGEN_BUILD_BLAS "Toggles the building of the Eigen Blas library" ${PROJECT_IS_TOP_LEVEL})
option(EIGEN_BUILD_LAPACK "Toggles the building of the included Eigen LAPACK library" ${PROJECT_IS_TOP_LEVEL})
if (EIGEN_BUILD_BLAS OR EIGEN_BUILD_LAPACK)
  # BLAS and LAPACK currently need a fortran compiler.
  include(CMakeDetermineFortranCompiler)
  if (NOT CMAKE_Fortran_COMPILER)
    set(EIGEN_BUILD_BLAS OFF)
    set(EIGEN_BUILD_LAPACK OFF)
  else()
    # Determine if we should build shared libraries for BLAS/LAPACK on this platform.
    get_cmake_property(EIGEN_BUILD_SHARED_LIBS TARGET_SUPPORTS_SHARED_LIBS)
  endif()
endif()

option(EIGEN_BUILD_BTL "Build benchmark suite" OFF)
option(EIGEN_BUILD_SPBENCH "Build sparse benchmark suite" OFF)
# Avoid building docs if included from another project.
# Building documentation requires creating and running executables on the host
# platform.  We shouldn't do this if cross-compiling.
if (PROJECT_IS_TOP_LEVEL AND NOT CMAKE_CROSSCOMPILING)
  set(EIGEN_BUILD_DOC_DEFAULT ON)
endif()
option(EIGEN_BUILD_DOC "Enable creation of Eigen documentation" ${EIGEN_BUILD_DOC_DEFAULT})

option(EIGEN_BUILD_DEMOS "Toggles the building of the Eigen demos" ${PROJECT_IS_TOP_LEVEL})

# Disable pkgconfig only for native Windows builds
if(NOT WIN32 OR NOT CMAKE_HOST_SYSTEM_NAME MATCHES Windows)
  option(EIGEN_BUILD_PKGCONFIG "Build pkg-config .pc file for Eigen" ${PROJECT_IS_TOP_LEVEL})
endif()
option(EIGEN_BUILD_CMAKE_PACKAGE "Enables the creation of EigenConfig.cmake and related files" ${PROJECT_IS_TOP_LEVEL})

if (EIGEN_BUILD_BLAS OR EIGEN_BUILD_LAPACK OR EIGEN_BUILT_BTL OR EIGEN_BUILD_BTL OR EIGEN_BUILD_SPBENCH OR EIGEN_BUILD_DOC OR EIGEN_BUILD_DEMOS)
  set(EIGEN_IS_BUILDING_ ON)
endif()

#==============================================================================
# Version Info.
#==============================================================================

# Automatically parse the version number from header files.
file(READ "${PROJECT_SOURCE_DIR}/Eigen/src/Core/util/Macros.h" _eigen_version_header)
string(REGEX MATCH "define[ \t]+EIGEN_WORLD_VERSION[ \t]+([0-9]+)" _eigen_world_version_match "${_eigen_version_header}")
set(EIGEN_WORLD_VERSION "${CMAKE_MATCH_1}")
string(REGEX MATCH "define[ \t]+EIGEN_MAJOR_VERSION[ \t]+([0-9]+)" _eigen_major_version_match "${_eigen_version_header}")
set(EIGEN_MAJOR_VERSION "${CMAKE_MATCH_1}")
string(REGEX MATCH "define[ \t]+EIGEN_MINOR_VERSION[ \t]+([0-9]+)" _eigen_minor_version_match "${_eigen_version_header}")
set(EIGEN_MINOR_VERSION "${CMAKE_MATCH_1}")
set(EIGEN_VERSION_NUMBER ${EIGEN_WORLD_VERSION}.${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION})

# If we are in a git repo, extract a changeset.
if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/.git)
  # if the git program is absent or this will leave the EIGEN_GIT_REVNUM string empty,
  # but won't stop CMake.
  execute_process(COMMAND git ls-remote --refs -q ${CMAKE_SOURCE_DIR} HEAD OUTPUT_VARIABLE EIGEN_GIT_OUTPUT)
endif()
# extract the git rev number from the git output...
if(EIGEN_GIT_OUTPUT)
string(REGEX MATCH "^([0-9;a-f]+).*" EIGEN_GIT_CHANGESET_MATCH "${EIGEN_GIT_OUTPUT}")
set(EIGEN_GIT_REVNUM "${CMAKE_MATCH_1}")
endif()
#...and show it next to the version number
if(EIGEN_GIT_REVNUM)
  set(EIGEN_VERSION "${EIGEN_VERSION_NUMBER} (git rev ${EIGEN_GIT_REVNUM})")
else()
  set(EIGEN_VERSION "${EIGEN_VERSION_NUMBER}")
endif()

#==============================================================================
# Install Path Configuration.
#==============================================================================

# Unconditionally allow install of targets to support nested dependency
# installations.
#
# Note: projects that depend on Eigen should _probably_ exclude installing
# Eigen by default (e.g. by using EXCLUDE_FROM_ALL when using
# FetchContent_Declare or add_subdirectory) to avoid overwriting a previous
# installation.

include(GNUInstallDirs)
# Backward compatibility support for EIGEN_INCLUDE_INSTALL_DIR
if(EIGEN_INCLUDE_INSTALL_DIR)
  message(WARNING "EIGEN_INCLUDE_INSTALL_DIR is deprecated. Use INCLUDE_INSTALL_DIR instead.")
endif()

if(EIGEN_INCLUDE_INSTALL_DIR AND NOT INCLUDE_INSTALL_DIR)
  set(INCLUDE_INSTALL_DIR ${EIGEN_INCLUDE_INSTALL_DIR}
      CACHE PATH "The directory relative to CMAKE_INSTALL_PREFIX where Eigen header files are installed")
else()
  set(INCLUDE_INSTALL_DIR
      "${CMAKE_INSTALL_INCLUDEDIR}/eigen3"
      CACHE PATH "The directory relative to CMAKE_INSTALL_PREFIX where Eigen header files are installed"
      )
endif()
set(CMAKEPACKAGE_INSTALL_DIR
    "${CMAKE_INSTALL_DATADIR}/eigen3/cmake"
    CACHE PATH "The directory relative to CMAKE_INSTALL_PREFIX where Eigen3Config.cmake is installed"
    )
set(PKGCONFIG_INSTALL_DIR
    "${CMAKE_INSTALL_DATADIR}/pkgconfig"
    CACHE PATH "The directory relative to CMAKE_INSTALL_PREFIX where eigen3.pc is installed"
    )

foreach(var INCLUDE_INSTALL_DIR CMAKEPACKAGE_INSTALL_DIR PKGCONFIG_INSTALL_DIR)
  # If an absolute path is specified, make it relative to "{CMAKE_INSTALL_PREFIX}".
  if(IS_ABSOLUTE "${${var}}")
    file(RELATIVE_PATH "${var}" "${CMAKE_INSTALL_PREFIX}" "${${var}}")
  endif()
endforeach()

#==============================================================================
# Eigen Library.
#==============================================================================

set ( EIGEN_VERSION_STRING ${EIGEN_VERSION_NUMBER} )
set ( EIGEN_VERSION_MAJOR  ${EIGEN_WORLD_VERSION} )
set ( EIGEN_VERSION_MINOR  ${EIGEN_MAJOR_VERSION} )
set ( EIGEN_VERSION_PATCH  ${EIGEN_MINOR_VERSION} )

# Alias Eigen_*_DIR to Eigen3_*_DIR:
set(Eigen_SOURCE_DIR ${Eigen3_SOURCE_DIR})
set(Eigen_BINARY_DIR ${Eigen3_BINARY_DIR})

# Imported target support
add_library (eigen INTERFACE)
add_library (Eigen3::Eigen ALIAS eigen)
target_include_directories (eigen INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)

# Export as title case Eigen
set_target_properties (eigen PROPERTIES EXPORT_NAME Eigen)

#==============================================================================
# Install Rule Configuration.
#==============================================================================

install(FILES
  signature_of_eigen3_matrix_library
  DESTINATION ${INCLUDE_INSTALL_DIR} COMPONENT Devel
  )

if(EIGEN_BUILD_PKGCONFIG)
    configure_file(eigen3.pc.in eigen3.pc @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/eigen3.pc
        DESTINATION ${PKGCONFIG_INSTALL_DIR})
endif()

install(DIRECTORY Eigen DESTINATION ${INCLUDE_INSTALL_DIR} COMPONENT Devel)

install(TARGETS eigen EXPORT Eigen3Targets)

if(EIGEN_BUILD_CMAKE_PACKAGE)
  include (CMakePackageConfigHelpers)
  configure_package_config_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Eigen3Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/Eigen3Config.cmake
    INSTALL_DESTINATION ${CMAKEPACKAGE_INSTALL_DIR}
    NO_SET_AND_CHECK_MACRO # Eigen does not provide legacy style defines
    NO_CHECK_REQUIRED_COMPONENTS_MACRO # Eigen does not provide components
  )

  # NOTE Remove the first code path once the minimum required CMake version is
  # bumped to 3.14 or above.
  if (CMAKE_VERSION VERSION_LESS 3.14)
    # Remove CMAKE_SIZEOF_VOID_P from Eigen3ConfigVersion.cmake since Eigen does
    # not depend on architecture specific settings or libraries. More
    # specifically, an Eigen3Config.cmake generated from a 64 bit target can be
    # used for 32 bit targets as well (and vice versa).
    set (_Eigen3_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
    unset (CMAKE_SIZEOF_VOID_P)
    write_basic_package_version_file (Eigen3ConfigVersion.cmake
                                      VERSION ${EIGEN_VERSION_NUMBER}
                                      COMPATIBILITY SameMajorVersion)
    set (CMAKE_SIZEOF_VOID_P ${_Eigen3_CMAKE_SIZEOF_VOID_P})
  else (CMAKE_VERSION VERSION_LESS 3.14)
    write_basic_package_version_file (Eigen3ConfigVersion.cmake
                                      VERSION ${EIGEN_VERSION_NUMBER}
                                      COMPATIBILITY SameMajorVersion
                                      ARCH_INDEPENDENT)
  endif (CMAKE_VERSION VERSION_LESS 3.14)

  # The Eigen target will be located in the Eigen3 namespace. Other CMake
  # targets can refer to it using Eigen3::Eigen.
  export (TARGETS eigen NAMESPACE Eigen3:: FILE Eigen3Targets.cmake)
  # Export Eigen3 package to CMake registry such that it can be easily found by
  # CMake even if it has not been installed to a standard directory.
  export (PACKAGE Eigen3)

  install (EXPORT Eigen3Targets NAMESPACE Eigen3:: DESTINATION ${CMAKEPACKAGE_INSTALL_DIR})

  install (FILES ${CMAKE_CURRENT_BINARY_DIR}/Eigen3Config.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/Eigen3ConfigVersion.cmake
          DESTINATION ${CMAKEPACKAGE_INSTALL_DIR})

  # Add uninstall target
  if(NOT TARGET uninstall)
    add_custom_target ( uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EigenUninstall.cmake)
  endif()
endif()

#==============================================================================
# General Build Configuration.
#==============================================================================

# Avoid setting the standard in a parent if unset.
if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_CXX_STANDARD 14 CACHE STRING "Default C++ standard")
  set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require C++ standard")
  set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "Allow C++ extensions")
endif()

# Guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

# Guard against bad build-type strings
if (PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Only try to figure out how to link the math library if we are building something.
# Otherwise, let the parent project deal with dependencies.
if (EIGEN_IS_BUILDING_)
  # Use Eigen's cmake files.
  set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

  set(CMAKE_INCLUDE_CURRENT_DIR OFF)

  find_package(StandardMathLibrary)
  set(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO "")
  if(NOT STANDARD_MATH_LIBRARY_FOUND)
    message(FATAL_ERROR
      "Can't link to the standard math library. Please report to the Eigen developers, telling them about your platform.")
  else()
    if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
      set(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO "${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO} ${STANDARD_MATH_LIBRARY}")
    else()
      set(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO "${STANDARD_MATH_LIBRARY}")
    endif()
  endif()
  if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
    message(STATUS "Standard libraries to link to explicitly: ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO}")
  else()
    message(STATUS "Standard libraries to link to explicitly: none")
  endif()

  # Default tests/examples/libraries to row-major.
  option(EIGEN_DEFAULT_TO_ROW_MAJOR "Use row-major as default matrix storage order" OFF)
  if(EIGEN_DEFAULT_TO_ROW_MAJOR)
    add_definitions("-DEIGEN_DEFAULT_TO_ROW_MAJOR")
  endif()
endif()


#==============================================================================
# Other Build Configurations.
#==============================================================================
add_subdirectory(unsupported)

if(EIGEN_BUILD_BLAS)
  add_subdirectory(blas)
endif()

if (EIGEN_BUILD_LAPACK)
  add_subdirectory(lapack)
endif()

if(EIGEN_BUILD_DOC)
  add_subdirectory(doc EXCLUDE_FROM_ALL)
endif()

# TODO: consider also replacing EIGEN_BUILD_BTL by a custom target "make btl"?
if(EIGEN_BUILD_BTL)
  add_subdirectory(bench/btl EXCLUDE_FROM_ALL)
endif()

if(NOT WIN32 AND EIGEN_BUILD_SPBENCH)
  add_subdirectory(bench/spbench EXCLUDE_FROM_ALL)
endif()

if (EIGEN_BUILD_DEMOS)
  add_subdirectory(demos EXCLUDE_FROM_ALL)
endif()

if (PROJECT_IS_TOP_LEVEL)
  # must be after test and unsupported, for configuring buildtests.in
  add_subdirectory(scripts EXCLUDE_FROM_ALL)
  configure_file(scripts/cdashtesting.cmake.in cdashtesting.cmake @ONLY)
endif()

#==============================================================================
# Summary.
#==============================================================================

if(PROJECT_IS_TOP_LEVEL)
  string(TOLOWER "${CMAKE_GENERATOR}" cmake_generator_tolower)
  if(cmake_generator_tolower MATCHES "makefile")
    message(STATUS "Available targets (use: make TARGET):")
  else()
    message(STATUS "Available targets (use: cmake --build . --target TARGET):")
  endif()
  message(STATUS "---------+--------------------------------------------------------------")
  message(STATUS "Target   |   Description")
  message(STATUS "---------+--------------------------------------------------------------")
  message(STATUS "install  | Install Eigen. Headers will be installed to:")
  message(STATUS "         |     <CMAKE_INSTALL_PREFIX>/<INCLUDE_INSTALL_DIR>")
  message(STATUS "         |   Using the following values:")
  message(STATUS "         |     CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "         |     INCLUDE_INSTALL_DIR:  ${INCLUDE_INSTALL_DIR}")
  message(STATUS "         |   Change the install location of Eigen headers using:")
  message(STATUS "         |     cmake . -DCMAKE_INSTALL_PREFIX=yourprefix")
  message(STATUS "         |   Or:")
  message(STATUS "         |     cmake . -DINCLUDE_INSTALL_DIR=yourdir")
  message(STATUS "uninstall| Remove files installed by the install target")
  if (EIGEN_BUILD_DOC)
    message(STATUS "doc      | Generate the API documentation, requires Doxygen & LaTeX")
  endif()
  if (EIGEN_BUILD_BLAS)
    message(STATUS "blas     | Build BLAS library (not the same thing as Eigen)")
  endif()
  if (EIGEN_BUILD_LAPACK)
    message(STATUS "lapack   | Build LAPACK subset library (not the same thing as Eigen)")
  endif()
  message(STATUS "---------+--------------------------------------------------------------")
  message(STATUS "")
endif()

message(STATUS "")
message(STATUS "Configured Eigen ${EIGEN_VERSION_NUMBER}")
message(STATUS "")

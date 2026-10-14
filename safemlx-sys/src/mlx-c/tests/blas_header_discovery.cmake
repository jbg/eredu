cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED MLX_SOURCE_DIR OR NOT DEFINED TEST_ROOT)
  message(FATAL_ERROR "MLX_SOURCE_DIR and TEST_ROOT are required")
endif()

# Resolve the optional tool before isolating header lookup from host packages.
find_package(PkgConfig QUIET)
set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)
unset(ENV{BLAS_HOME})
unset(ENV{CMAKE_INCLUDE_PATH})
set(CMAKE_IGNORE_PATH /usr/include /usr/local/include
                      /usr/local/opt/openblas/include)
set(header_module "${MLX_SOURCE_DIR}/cmake/blas_headers.cmake")

if(PkgConfig_FOUND)
  # Only the .pc file identifies this nonstandard header directory.
  set(metadata_headers "${TEST_ROOT}/metadata/vendor-headers")
  file(MAKE_DIRECTORY "${metadata_headers}" "${TEST_ROOT}/pkgconfig")
  file(WRITE "${metadata_headers}/cblas.h" "/* CBLAS fixture */\n")
  file(WRITE "${metadata_headers}/lapacke.h" "/* LAPACKE fixture */\n")
  file(WRITE "${TEST_ROOT}/pkgconfig/openblas.pc"
       "Name: OpenBLAS\nDescription: Header discovery fixture\nVersion: 1.0\nCflags: -I${metadata_headers}\nLibs:\n")
  set(ENV{PKG_CONFIG_PATH} "${TEST_ROOT}/pkgconfig")
  set(ENV{PKG_CONFIG_LIBDIR} "${TEST_ROOT}/pkgconfig")
  set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH FALSE)
  include("${header_module}")
  if(NOT BLAS_INCLUDE_DIRS STREQUAL metadata_headers OR
     NOT LAPACK_INCLUDE_DIRS STREQUAL metadata_headers)
    message(FATAL_ERROR "pkg-config metadata did not locate BLAS/LAPACK headers")
  endif()
  message(STATUS "BLAS/LAPACK pkg-config header discovery passed")
else()
  message(STATUS "pkg-config unavailable; skipping metadata header discovery test")
endif()

# Without metadata, normal prefix discovery finds the nested OpenBLAS layout.
unset(BLAS_INCLUDE_DIRS CACHE)
unset(LAPACK_INCLUDE_DIRS CACHE)
unset(MLX_CBLAS_HEADERS_INCLUDE_DIRS)
unset(MLX_LAPACKE_HEADERS_INCLUDE_DIRS)
set(CMAKE_DISABLE_FIND_PACKAGE_PkgConfig TRUE)
set(PkgConfig_FOUND FALSE)
set(CMAKE_PREFIX_PATH "${TEST_ROOT}/fallback")
set(fallback_headers "${CMAKE_PREFIX_PATH}/include/openblas")
file(MAKE_DIRECTORY "${fallback_headers}")
file(WRITE "${fallback_headers}/cblas.h" "/* CBLAS fixture */\n")
file(WRITE "${fallback_headers}/lapacke.h" "/* LAPACKE fixture */\n")
include("${header_module}")
if(NOT BLAS_INCLUDE_DIRS STREQUAL fallback_headers OR
   NOT LAPACK_INCLUDE_DIRS STREQUAL fallback_headers)
  message(FATAL_ERROR "OpenBLAS nested header fallback failed")
endif()
message(STATUS "BLAS/LAPACK nested header fallback passed")

# Explicit choices continue to override discovery.
set(BLAS_INCLUDE_DIRS "${TEST_ROOT}/explicit-blas")
set(LAPACK_INCLUDE_DIRS "${TEST_ROOT}/explicit-lapack")
include("${header_module}")
if(NOT BLAS_INCLUDE_DIRS STREQUAL "${TEST_ROOT}/explicit-blas" OR
   NOT LAPACK_INCLUDE_DIRS STREQUAL "${TEST_ROOT}/explicit-lapack")
  message(FATAL_ERROR "Explicit BLAS/LAPACK header settings were overwritten")
endif()
message(STATUS "Explicit BLAS/LAPACK header overrides passed")

# Copyright © 2023-2024 Apple Inc.

cmake_minimum_required(VERSION 3.16)

# ----------------------------- Setup -----------------------------
project(mlx.data LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)

# ----------------------------- Configuration -----------------------------
if(NOT MLX_C_VERSION)
  set(MLX_C_VERSION 0.6.0)
endif()

option(BUILD_SHARED_LIBS "Build mlx C as a shared library" OFF)
option(MLX_C_BUILD_EXAMPLES "Build examples for mlx C" ON)
option(MLX_C_USE_SYSTEM_MLX "Use system MLX" OFF)
option(MLX_C_BUILD_PATCH_TESTS
       "Build opt-in tests carried by Eredu's vendored MLX patches" OFF)

if(MLX_C_BUILD_PATCH_TESTS AND MLX_C_USE_SYSTEM_MLX)
  message(FATAL_ERROR
          "MLX_C_BUILD_PATCH_TESTS requires Eredu's vendored patched MLX")
endif()

# ----------------------------- mlx -----------------------------

include(FetchContent)
# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# mlx
set(MLX_BUILD_TESTS ${MLX_C_BUILD_PATCH_TESTS})
set(MLX_BUILD_EXAMPLES OFF)
set(MLX_BUILD_BENCHMARKS OFF)
set(MLX_BUILD_PYTHON_BINDINGS OFF)
set(MLX_BUILD_GGUF OFF)

if(MLX_C_USE_SYSTEM_MLX)
  find_package(MLX REQUIRED)
else()
  # Keep common native builds offline and reproducible. These archives are
  # shipped in the safemlx-sys crate; Cargo verifies the crate checksum and
  # CMake verifies each upstream archive before extracting it. FetchContent's
  # first-declaration-wins rule makes MLX use these local dependency archives.
  function(safemlx_declare_vendored name filename sha256)
    set(archive "${CMAKE_CURRENT_LIST_DIR}/../../vendor/${filename}")
    if(NOT EXISTS "${archive}")
      message(FATAL_ERROR "Vendored source archive is missing: ${archive}")
    endif()
    FetchContent_Declare(${name} URL "${archive}"
                         URL_HASH "SHA256=${sha256}" ${ARGN})
  endfunction()

  safemlx_declare_vendored(
    json json-v3.11.3.tar.xz
    d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
  safemlx_declare_vendored(
    fmt fmt-12.1.0.tar.gz
    ea7de4299689e12b6dddd392f9896f08fb0777ac7168897a244a6d6085043fea
    EXCLUDE_FROM_ALL)
  safemlx_declare_vendored(
    metal_cpp metal-cpp-26.zip
    4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4)
  # Patch contents, not only patch paths, identify the extracted native source.
  # FetchContent otherwise reuses its successful patch stamp after an edit,
  # silently compiling an older revision. Keep the binary path stable for the
  # Rust packaging/metallib integration, but never patch an unknown old tree.
  set(mlx_patch_files
      mlx-apple-mobile-platforms.patch
      mlx-cuda-global-encoder-lifetime.patch
      mlx-affine-group16.patch
      mlx-completion-events.patch
      mlx-variable-all-to-all.patch
      mlx-host-transfer-buffer.patch
      mlx-host-transfer-umbrella.patch
      mlx-timing-events.patch
      mlx-cuda-jit-include-dirs.patch
      mlx-embedded-metallib.patch
      mlx-host-transfer-noexcept-move.patch
      mlx-cpu-submission-progress.patch
      mlx-gpu-submission-progress.patch
      mlx-submission-recovery.patch)
  set(mlx_patch_applier "${CMAKE_CURRENT_LIST_DIR}/patches/apply-idempotent.cmake")
  file(SHA256 "${mlx_patch_applier}" mlx_patch_identity)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${mlx_patch_applier}")
  set(mlx_patch_commands)
  foreach(mlx_patch IN LISTS mlx_patch_files)
    set(mlx_patch_path "${CMAKE_CURRENT_LIST_DIR}/patches/${mlx_patch}")
    file(SHA256 "${mlx_patch_path}" mlx_patch_hash)
    string(APPEND mlx_patch_identity "\n${mlx_patch}:${mlx_patch_hash}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${mlx_patch_path}")
    if(mlx_patch_commands)
      list(APPEND mlx_patch_commands COMMAND)
    endif()
    list(APPEND mlx_patch_commands
         ${CMAKE_COMMAND} "-DPATCH_FILE=${mlx_patch_path}" -P "${mlx_patch_applier}")
  endforeach()
  string(SHA256 mlx_patch_set_hash "${mlx_patch_identity}")
  safemlx_declare_vendored(
    mlx mlx-v0.32.0.tar.gz
    8e74a0eb613861ce50c09402dd99dbc42b65a762e7fdd291caa39f611db978ec
    SOURCE_DIR "${FETCHCONTENT_BASE_DIR}/mlx-${mlx_patch_set_hash}-src"
    SUBBUILD_DIR "${FETCHCONTENT_BASE_DIR}/mlx-${mlx_patch_set_hash}-subbuild"
    BINARY_DIR "${FETCHCONTENT_BASE_DIR}/mlx-build"
    PATCH_COMMAND ${mlx_patch_commands})
  FetchContent_MakeAvailable(mlx)
endif()

if(MLX_C_BUILD_PATCH_TESTS)
  add_executable(
    mlx-patch-test-binary
    ${CMAKE_CURRENT_LIST_DIR}/tests/event_submission_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/submission_recovery_tests.cpp
    ${mlx_SOURCE_DIR}/tests/tests.cpp
    ${mlx_SOURCE_DIR}/tests/completion_tests.cpp
    ${mlx_SOURCE_DIR}/tests/distributed_all_to_all_v_tests.cpp
    ${mlx_SOURCE_DIR}/tests/host_transfer_tests.cpp)
  target_sources(mlx-patch-test-binary PRIVATE
                 ${mlx_SOURCE_DIR}/tests/scheduler_recovery_tests.cpp
                 ${mlx_SOURCE_DIR}/tests/gpu_submission_progress_tests.cpp)
  target_link_libraries(mlx-patch-test-binary PRIVATE mlxc mlx doctest)

  add_custom_target(
    mlx-patch-tests
    COMMAND
      ${CMAKE_COMMAND} -E env DEVICE=cpu
      $<TARGET_FILE:mlx-patch-test-binary>
      "--test-case=*completion*,*failed CPU event*,*all_to_all_v*,*host transfer*,*CPU recovery*,*GPU recovery*,*submission recovery*"
    DEPENDS mlx-patch-test-binary
    USES_TERMINAL
    COMMENT "Running tests carried by Eredu's vendored MLX patches")
endif()

# When MLX is a FetchContent dependency its own install rules are not included
# in this project's install target. Windows uses a shared-library boundary, so
# install the MLX DLL and import library beside the mlx-c artifacts explicitly.
if(WIN32 AND BUILD_SHARED_LIBS)
  include(GNUInstallDirs)
  install(TARGETS mlx
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# ----------------------------- lib -----------------------------

set(mlxc-src
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/array.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/closure.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/compile.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/cuda.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/device.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/distributed.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/distributed_group.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/error.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/event.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/export.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/fast.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/fft.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/graph_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/host_transfer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/io.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/io_types.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/linalg.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/map.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/memory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/metal.cpp
    # ${CMAKE_CURRENT_LIST_DIR}/mlx/c/object.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/ops.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/random.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/stream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/submission.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/string.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/transforms.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/transforms_impl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/vector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/version.cpp)

add_library(mlxc ${mlxc-src})

target_link_libraries(mlxc PRIVATE mlx)
target_include_directories(mlxc
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
set_property(TARGET mlxc PROPERTY POSITION_INDEPENDENT_CODE ON)

# Windows DLL symbol exports
if(WIN32)
  set_target_properties(mlxc PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

if(MLX_C_BUILD_EXAMPLES)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/examples)
endif()

# ----------------------------- Installation -----------------------------
include(GNUInstallDirs)

# Install library
install(
  TARGETS mlxc
  EXPORT MLXCTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install headers
install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/mlx
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT headers
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "private" EXCLUDE)

# Install cmake config
set(MLX_C_CMAKE_BUILD_CONFIG ${CMAKE_BINARY_DIR}/MLXCConfig.cmake)
set(MLX_C_CMAKE_BUILD_VERSION_CONFIG
    ${CMAKE_BINARY_DIR}/MLXCConfigVersion.cmake)
set(MLX_C_CMAKE_INSTALL_MODULE_DIR share/cmake/MLXC)

install(
  EXPORT MLXCTargets
  FILE MLXCTargets.cmake
  DESTINATION ${MLX_C_CMAKE_INSTALL_MODULE_DIR})

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${MLX_C_CMAKE_BUILD_VERSION_CONFIG}
  COMPATIBILITY SameMajorVersion
  VERSION ${MLX_C_VERSION})

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/mlx-c.pc.in ${MLX_C_CMAKE_BUILD_CONFIG}
  INSTALL_DESTINATION ${MLX_C_CMAKE_INSTALL_MODULE_DIR}
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
  PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR
            MLX_C_CMAKE_INSTALL_MODULE_DIR)

install(FILES ${MLX_C_CMAKE_BUILD_CONFIG} ${MLX_C_CMAKE_BUILD_VERSION_CONFIG}
        DESTINATION ${MLX_C_CMAKE_INSTALL_MODULE_DIR})

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
      mlx-cpu-bf16-row-reduction.patch
      mlx-cpu-cascade-sum.patch
      mlx-cpu-softmax-reduction.patch
      mlx-cpu-argpartition-equal-keys.patch
      mlx-gpu-submission-progress.patch
      mlx-submission-recovery.patch
      mlx-retained-sdpa-settings.patch
      mlx-host-transfer-allocation-identity.patch
      mlx-allocation-owner-retention.patch
      mlx-allocation-lifetime-generations.patch
      mlx-deferred-allocation-owner-retention.patch
      mlx-view-descriptor-normalization.patch
      mlx-distributed-terminal.patch
      mlx-prepared-allocation-owner.patch
      mlx-record-storage-quota.patch
      mlx-arraydesc-retirement.patch
      mlx-graph-metadata-quota.patch
      mlx-cpu-f32-argsort-inplace.patch
      mlx-original-prepared-input.patch
      mlx-host-transfer-policy-test.patch
      mlx-original-model-input-handle.patch
      mlx-prefill-root-descriptor-bound.patch
      mlx-prefill-scoped-completion.patch
      mlx-original-event-task-controls.patch
      mlx-original-scoped-evaluation.patch
      mlx-original-operation-observation.patch
      mlx-original-prompt-input.patch
      mlx-resident-graph-construction.patch
      mlx-resident-view-axes.patch
      mlx-resident-reduction-axes.patch
      mlx-custom-kernel-borrowed-call.patch
      mlx-metal-library-builder.patch
      mlx-resident-gated-product.patch
      mlx-resident-softplus.patch
      mlx-resident-grouped-axes.patch
      mlx-resident-convolution-axes.patch
      mlx-resident-constant-pad.patch
      mlx-record-capacity-causes.patch
      mlx-record-request-extent.patch
      mlx-custom-kernel-fixed-invocation.patch
      mlx-resident-router.patch
      mlx-prepared-metal-definition.patch
      mlx-record-lifetime-fit.patch
      mlx-admitted-gpu-stream.patch
      mlx-prepared-kernel-family.patch
      mlx-stream-sync-retirement.patch
      mlx-graph-lifetime-fit.patch
      mlx-resident-dispatch-graph.patch
      mlx-concatenate-worker-graph.patch
      mlx-request-pipeline-cache.patch
      mlx-kernel-family-64.patch
      mlx-resident-hybrid-dispatch.patch
      mlx-resident-grouped-dispatch.patch
      mlx-grouped-indexed-sources.patch
      mlx-resident-router-dispatch.patch
      mlx-sampling-filter-sources.patch
      mlx-standard-sampling-random.patch
      mlx-resident-nested-completion.patch
      mlx-resident-reset-readiness.patch
      mlx-mirostat-mask-source.patch
      mlx-original-worker-borrow.patch
      mlx-resident-host-copy.patch
      mlx-original-host-source.patch
      mlx-quantized-supplied-stream.patch
      mlx-resident-cumulative-frontiers.patch
      mlx-graph-quota-next-fit.patch
      mlx-original-buffer-contention.patch
      mlx-packed-embedding-fallback.patch
      mlx-graph-construction-class-index.patch
      mlx-pooled-mask-gather.patch
      mlx-implicit-convolution-receipt.patch
      mlx-fence-header-guard.patch
      mlx-distributed-storage-inventory.patch
      mlx-ring-socket-tasks.patch
      mlx-ring-captured-graph.patch
      mlx-ring-pool-tasks.patch
      mlx-ring-reduction-plan.patch
      mlx-ring-worker-storage.patch
      mlx-ring-outer-dispatch.patch
      mlx-distributed-original-constructor.patch
      mlx-cpu-copy-storage-source.patch
      mlx-distributed-worker-graph-loan.patch
      mlx-distributed-cpu-eval-source.patch
      mlx-distributed-persistent-source.patch
      mlx-distributed-persistent-managed-boundary.patch
      mlx-distributed-cold-operation-source.patch
      mlx-distributed-workspace-layout-source.patch
      mlx-distributed-retained-buffer-source.patch
      mlx-prepared-host-destination.patch
      mlx-original-control-child-scope.patch
      mlx-distributed-resident-constructor-loan.patch
      mlx-indexed-rank-and-signed-sources.patch
      mlx-cpu-copy-eval-source.patch
      mlx-cpu-unary-eval-source.patch
      mlx-cpu-binary-eval-source.patch
      mlx-cpu-cast-eval-source.patch
      mlx-cpu-slice-eval-source.patch
      mlx-cpu-softmax-eval-source.patch
      mlx-cpu-greedy-eval-source.patch
      mlx-cpu-reshape-eval-source.patch
      mlx-cpu-random-bits-eval-source.patch
      mlx-cpu-matmul-eval-source.patch
      mlx-prepared-input-scalars.patch
      mlx-cpu-matmul-selection.patch
      mlx-cpu-matmul-selected-dtype.patch
      mlx-cpu-alias-eval-source.patch
      mlx-resident-byte-view-source.patch
      mlx-cpu-matmul-selected-empty.patch
      mlx-cpu-pointwise-logical-source.patch
      mlx-cpu-gather-source.patch
      mlx-cpu-reduction-source.patch
      mlx-cpu-lookup-source.patch
      mlx-cpu-rms-source.patch
      mlx-cpu-view-alias.patch
      mlx-cpu-concatenate-source.patch
      mlx-cpu-gemma-rope.patch
      mlx-cpu-arange-promotion.patch
      mlx-cpu-gemma-attention.patch
      mlx-cpu-ordered-partition.patch
      mlx-cpu-ordered-gather.patch
      mlx-cpu-ordered-mask.patch
      mlx-cpu-ordered-scatter.patch
      mlx-cpu-ordered-aliases.patch
      mlx-cpu-reshape-inference.patch
      mlx-cpu-all-sum-source.patch
      mlx-cpu-causal-mask-source.patch
      mlx-cpu-broadcast-select-source.patch
      mlx-cpu-masked-attention-source.patch
      mlx-cpu-strided-row-reduction.patch
      mlx-cpu-matmul-copy-source.patch
      mlx-cpu-reshape-copy-source.patch
      mlx-eval-ready-input-publication.patch
      mlx-cpu-preview-slice-source.patch
      mlx-cpu-positive-slice-source.patch
      mlx-host-transfer-nested-stream-capacity.patch
      mlx-cpu-flat-reduction-source.patch
      mlx-cpu-token-update-source.patch
      mlx-cpu-candidate-sort-source.patch
      mlx-cpu-static-update-source.patch
      mlx-cpu-typed-static-update-source.patch
      mlx-cpu-half-gather-source.patch
      mlx-cpu-half-squeeze-source.patch
      mlx-cpu-half-softmax-source.patch
      mlx-cpu-bf16-matmul-copy-source.patch
      mlx-cpu-f16-selected-matmul.patch
      mlx-cpu-half-attention-selection.patch
      mlx-cpu-byte-view-source.patch
      mlx-cpu-byte-frame-source.patch
      mlx-cpu-shard-selection-source.patch
      mlx-cpu-member-concatenate-source.patch
      mlx-cpu-f16-rms-source.patch
      mlx-cpu-half-row-sum-source.patch
      mlx-prepared-input-boolean-source.patch
      mlx-cpu-sampling-filter-leaves.patch
      mlx-cpu-sampling-gather-axis.patch
      mlx-cpu-sampling-argsort-axis.patch
      mlx-ring-variable-source.patch
      mlx-cpu-strided-select-source.patch
      mlx-broadcast-result-reservation.patch
      mlx-resident-mxfp4-grouped-source.patch
      mlx-ring-variable-envelope.patch
    mlx-prepared-zero-source.patch
      mlx-masked-scatter-source.patch
      mlx-scatter-axis-sum-source.patch
      mlx-cpu-flat-scatter-source.patch
      mlx-kernel-name-fold.patch
      mlx-cpu-row-maximum-source.patch
      mlx-cpu-host-transfer-source.patch
      mlx-host-transfer-source-witness.patch
      mlx-cpu-grouped-sum-source.patch
      mlx-cpu-tiled-gather-mm-source.patch
      mlx-cpu-grouped-argsort-source.patch
      mlx-cpu-grouped-scatter-source.patch
      mlx-cpu-grouped-gather-axis-source.patch
      mlx-cpu-empty-typed-joins-source.patch
      mlx-cpu-empty-structural-alias-source.patch
      mlx-cpu-row-movement-source.patch
      mlx-cpu-routing-selection-source.patch
      mlx-cpu-empty-slice-source.patch
      mlx-cpu-affine-quantized-source.patch
      mlx-cpu-quantization-unary-source.patch
      mlx-cpu-quantization-integer-source.patch
      mlx-cpu-quantization-half-source.patch
      mlx-cpu-affine-converter-source.patch
      mlx-cpu-failed-eval-cleanup.patch
      mlx-affine-quantize-construction.patch
      mlx-cpu-empty-buffer-birth.patch
      mlx-cpu-event-dispatch-layout.patch
      mlx-cpu-mxfp4-construction.patch
      mlx-cpu-ranked-reshape.patch
      mlx-cpu-mxfp4-payload-layout.patch
      mlx-cpu-prepared-host-storage.patch
      mlx-physical-memory-placement.patch
      mlx-physical-memory-access.patch
      mlx-physical-memory-default-placement.patch
      mlx-physical-memory-gpu-envelope.patch
      mlx-original-buffer-lifetime.patch
      mlx-physical-backing-custody.patch
      mlx-scoped-physical-funding.patch
      mlx-cpu-mirostat-selection-source.patch
      mlx-cpu-controlled-router-source.patch
      mlx-worker-source-identity.patch
      mlx-cpu-boolean-gather-source.patch
      mlx-cpu-packed-decode-source.patch
      mlx-cpu-addmm-source.patch
      mlx-cpu-fp8-decode-source.patch
      mlx-completed-foreign-readback.patch
      mlx-cpu-copy-alias-source.patch
      mlx-cpu-wide-position-source.patch
      mlx-cpu-wide-trigonometry.patch
      mlx-cpu-layer-norm-source.patch
      mlx-cpu-indexed-histogram-source.patch
      mlx-ordinary-host-buffer-source.patch
      mlx-eval-traversal-capacity-causes.patch
      mlx-ordinary-control-observation.patch
      mlx-ordinary-metal-control-observation.patch
      mlx-temporary-physical-cache-policy.patch
      mlx-ordinary-cpu-fence-observation.patch
      mlx-ordinary-metal-router-observation.patch
      mlx-scoped-fresh-physical-backing.patch)
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
  set(mlx_archive_hash
      8e74a0eb613861ce50c09402dd99dbc42b65a762e7fdd291caa39f611db978ec)
  set(mlx_prepared_source "${FETCHCONTENT_BASE_DIR}/mlx-${mlx_patch_set_hash}-src")
  set(mlx_population_options)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
    # SOURCE_SUBDIR prevents MakeAvailable from adding the native project before
    # its verified after-images have reached their stable compilation paths.
    set(mlx_compile_source "${FETCHCONTENT_BASE_DIR}/mlx-compile-src")
    set(mlx_compile_state "${FETCHCONTENT_BASE_DIR}/mlx-compile-state")
    # Invalidate before archive/patch preparation too. A failed configure must
    # never leave a completed marker for the attempted compilation revision.
    file(REMOVE "${mlx_compile_state}/completed.txt")
    set(mlx_population_subdir "__safemlx_population_only")
    if(EXISTS "${mlx_prepared_source}/${mlx_population_subdir}")
      message(FATAL_ERROR "MLX population-only subdirectory unexpectedly exists")
    endif()
    list(APPEND mlx_population_options SOURCE_SUBDIR "${mlx_population_subdir}")
  endif()
  safemlx_declare_vendored(
    mlx mlx-v0.32.0.tar.gz "${mlx_archive_hash}"
    SOURCE_DIR "${mlx_prepared_source}"
    SUBBUILD_DIR "${FETCHCONTENT_BASE_DIR}/mlx-${mlx_patch_set_hash}-subbuild"
    BINARY_DIR "${FETCHCONTENT_BASE_DIR}/mlx-build"
    PATCH_COMMAND ${mlx_patch_commands} ${mlx_population_options})
  FetchContent_MakeAvailable(mlx)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
    if(NOT mlx_SOURCE_DIR STREQUAL mlx_prepared_source OR
       EXISTS "${mlx_SOURCE_DIR}/${mlx_population_subdir}")
      message(FATAL_ERROR "MLX mirror requires the verified vendored preparation")
    endif()
    set(mlx_mirror_helper "${CMAKE_CURRENT_LIST_DIR}/cmake/sync-mlx-source.cmake")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${mlx_mirror_helper}")
    file(SHA256 "${mlx_mirror_helper}" mlx_mirror_helper_hash)
    string(SHA256 mlx_compile_identity
           "${mlx_archive_hash}\n${mlx_patch_set_hash}\n${mlx_mirror_helper_hash}")
    execute_process(
      COMMAND ${CMAKE_COMMAND} "-DSOURCE_DIR=${mlx_SOURCE_DIR}"
              "-DDESTINATION_DIR=${mlx_compile_source}"
              "-DSTATE_DIR=${mlx_compile_state}" "-DIDENTITY=${mlx_compile_identity}"
              -P "${mlx_mirror_helper}"
      RESULT_VARIABLE mlx_mirror_result)
    if(NOT mlx_mirror_result STREQUAL "0")
      message(FATAL_ERROR "MLX compilation source refresh failed")
    endif()
    set(mlx_SOURCE_DIR "${mlx_compile_source}")
    add_subdirectory("${mlx_SOURCE_DIR}" "${mlx_BINARY_DIR}")
  endif()
  if(APPLE AND MLX_BUILD_METAL)
    # The vendored embedded-metallib worker lives in mlx itself. Every static
    # consumer must inherit its libcompression dependency, including new tools.
    target_link_libraries(mlx PUBLIC compression)
  endif()
endif()

# Explicit cold test target; not built or installed with production artifacts.
if(NOT MLX_C_USE_SYSTEM_MLX)
  add_executable(mlx-operation-observer-allocation-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/operation_observer_allocation_test.cpp)
  target_link_libraries(mlx-operation-observer-allocation-test PRIVATE mlxc mlx)
  add_executable(mlx-cpu-argsort-benchmark EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/cpu_argsort_benchmark.cpp)
  target_link_libraries(mlx-cpu-argsort-benchmark PRIVATE mlx)
  add_executable(mlx-arraydesc-retirement-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/arraydesc_retirement_test.cpp)
  target_link_libraries(mlx-arraydesc-retirement-test PRIVATE mlx)
  add_executable(mlx-cold-sdpa-settings-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/sdpa_settings_test.cpp)
  target_link_libraries(mlx-cold-sdpa-settings-test PRIVATE mlx)
  add_executable(mlx-allocation-generations-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/allocation_generations_test.cpp)
  target_link_libraries(mlx-allocation-generations-test PRIVATE mlxc mlx)
  add_executable(mlx-prepared-allocation-owner-test EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_LIST_DIR}/tests/prepared_allocation_owner_test.cpp)
  target_link_libraries(mlx-prepared-allocation-owner-test PRIVATE mlxc mlx)
  add_executable(mlx-deferred-allocation-owner-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/deferred_allocation_owner_test.cpp)
  target_link_libraries(mlx-deferred-allocation-owner-test PRIVATE mlxc mlx)
  # Isolated process: exact registry geometry must not inherit other tests.
  add_executable(mlx-submission-retirement-fairness-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/submission_retirement_fairness_test.cpp)
  target_link_libraries(mlx-submission-retirement-fairness-test PRIVATE mlx)
  add_executable(mlx-submission-retirement-pass-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/submission_retirement_pass_test.cpp)
  target_link_libraries(mlx-submission-retirement-pass-test PRIVATE mlxc mlx)
  add_executable(mlx-distributed-terminal-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/distributed_terminal_test.cpp)
  target_link_libraries(mlx-distributed-terminal-test PRIVATE mlxc mlx)
  add_executable(mlx-distributed-original-ring-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/distributed_original_ring_test.cpp)
  target_link_libraries(mlx-distributed-original-ring-test PRIVATE mlxc mlx)
  add_executable(mlx-distributed-variable-original-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/distributed_variable_original_test.cpp)
  target_link_libraries(mlx-distributed-variable-original-test PRIVATE mlxc mlx)
  add_executable(mlx-shared-metal-device-init-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/shared_metal_device_init_test.cpp)
  target_link_libraries(mlx-shared-metal-device-init-test PRIVATE mlxc mlx)
  if(MLX_BUILD_METAL)
    target_compile_definitions(mlx-shared-metal-device-init-test PRIVATE MLX_C_PATCH_TEST_METAL)
  endif()
  add_executable(mlx-shared-scheduler-init-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/shared_scheduler_init_test.cpp)
  target_link_libraries(mlx-shared-scheduler-init-test PRIVATE mlxc mlx)
  add_executable(mlx-stream-registration-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/stream_registration_test.cpp)
  target_link_libraries(mlx-stream-registration-test PRIVATE mlxc mlx)
  add_executable(mlx-cpu-worker-startup-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/cpu_worker_startup_test.cpp)
  target_link_libraries(mlx-cpu-worker-startup-test PRIVATE mlxc mlx ${CMAKE_DL_LIBS})
  add_executable(mlx-cpu-worker-directory-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/cpu_worker_directory_test.cpp)
  target_link_libraries(mlx-cpu-worker-directory-test PRIVATE mlxc mlx)
  if(MLX_BUILD_METAL)
    target_compile_definitions(mlx-shared-scheduler-init-test PRIVATE MLX_C_PATCH_TEST_METAL)
  endif()
  add_executable(mlx-shared-input-allocator-init-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/shared_input_allocator_init_test.cpp)
  target_link_libraries(mlx-shared-input-allocator-init-test PRIVATE mlxc mlx)
  if(MLX_BUILD_METAL)
    target_compile_definitions(mlx-shared-input-allocator-init-test PRIVATE MLX_C_PATCH_TEST_METAL)
  endif()
  add_executable(mlx-submission-registry-init-test EXCLUDE_FROM_ALL
                 ${CMAKE_CURRENT_LIST_DIR}/tests/submission_registry_init_test.cpp)
  target_link_libraries(mlx-submission-registry-init-test PRIVATE mlxc mlx)
  if(APPLE AND MLX_BUILD_METAL)
    target_link_libraries(mlx-operation-observer-allocation-test PRIVATE compression)
    target_link_libraries(mlx-cpu-argsort-benchmark PRIVATE compression)
    target_link_libraries(mlx-arraydesc-retirement-test PRIVATE compression)
    target_link_libraries(mlx-cold-sdpa-settings-test PRIVATE compression)
    target_link_libraries(mlx-allocation-generations-test PRIVATE compression)
    target_link_libraries(mlx-deferred-allocation-owner-test PRIVATE compression)
    target_link_libraries(mlx-prepared-allocation-owner-test PRIVATE compression)
    target_link_libraries(mlx-submission-retirement-fairness-test PRIVATE compression)
    target_link_libraries(mlx-submission-retirement-pass-test PRIVATE compression)
    target_link_libraries(mlx-submission-registry-init-test PRIVATE compression)
    target_link_libraries(mlx-shared-input-allocator-init-test PRIVATE compression)
    target_link_libraries(mlx-shared-metal-device-init-test PRIVATE compression)
    target_link_libraries(mlx-shared-scheduler-init-test PRIVATE compression)
    target_link_libraries(mlx-stream-registration-test PRIVATE compression)
    target_link_libraries(mlx-cpu-worker-directory-test PRIVATE compression)
    target_link_libraries(mlx-cpu-worker-startup-test PRIVATE compression)
    target_link_libraries(mlx-distributed-terminal-test PRIVATE compression)
    target_link_libraries(mlx-distributed-original-ring-test PRIVATE compression)
  endif()
endif()

if(MLX_C_BUILD_PATCH_TESTS)
  add_executable(
    mlx-patch-test-binary
    ${CMAKE_CURRENT_LIST_DIR}/tests/event_submission_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/submission_recovery_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/submission_record_readiness_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/submission_scope_owner_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/ordinary_control_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/submission_record_quota_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/graph_metadata_quota_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/prepared_input_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/input_allocator_slot_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/original_buffer_owner_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/prefill_roots_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/scoped_observation_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/scoped_roots_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/native_controls_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/operation_observer_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/array_descriptor_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/cpu_argsort_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/kernel_name_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/intrusive_index_tests.cpp
    ${mlx_SOURCE_DIR}/tests/tests.cpp
    ${mlx_SOURCE_DIR}/tests/completion_tests.cpp
    ${mlx_SOURCE_DIR}/tests/distributed_all_to_all_v_tests.cpp
    ${mlx_SOURCE_DIR}/tests/host_transfer_tests.cpp)
  target_sources(mlx-patch-test-binary PRIVATE
                 ${mlx_SOURCE_DIR}/tests/scheduler_recovery_tests.cpp
                 ${mlx_SOURCE_DIR}/tests/gpu_submission_progress_tests.cpp)
  if(MLX_BUILD_METAL)
    target_compile_definitions(mlx-patch-test-binary PRIVATE MLX_C_PATCH_TEST_METAL)
    target_sources(mlx-patch-test-binary PRIVATE
                   ${CMAKE_CURRENT_LIST_DIR}/tests/prepared_input_metal_tests.cpp
                   ${CMAKE_CURRENT_LIST_DIR}/tests/scoped_failure_metal_tests.cpp
                   ${CMAKE_CURRENT_LIST_DIR}/tests/metal_receipt_controls_tests.cpp
                   ${CMAKE_CURRENT_LIST_DIR}/tests/metal_persistent_storage_tests.cpp)
  endif()
  target_link_libraries(mlx-patch-test-binary PRIVATE mlxc mlx doctest)
  if(APPLE AND MLX_BUILD_METAL)
    target_link_libraries(mlx-patch-test-binary PRIVATE compression)
  endif()

  add_custom_target(
    mlx-patch-tests
    COMMAND
      ${CMAKE_COMMAND} -E env DEVICE=cpu
      $<TARGET_FILE:mlx-arraydesc-retirement-test>
    COMMAND
      ${CMAKE_COMMAND} -E env DEVICE=cpu
      $<TARGET_FILE:mlx-patch-test-binary>
      "--test-case=*completion*,*failed CPU event*,*all_to_all_v*,*host transfer*,*CPU recovery*,*GPU recovery*,*submission recovery*,*graph quota*,*prepared input*,*persistent encoder lists*,*persistent encoder index*,*CPU argsort*"
    # Separate process: the fairness regression requires an initially empty
    # submission registry and must not share the doctest binary's history.
    COMMAND
      ${CMAKE_COMMAND} -E env DEVICE=cpu
      $<TARGET_FILE:mlx-submission-retirement-fairness-test>
    COMMAND
      ${CMAKE_COMMAND} -E env DEVICE=cpu
      $<TARGET_FILE:mlx-submission-retirement-pass-test>
    COMMAND
      ${CMAKE_COMMAND} -E env DEVICE=cpu
      $<TARGET_FILE:mlx-submission-registry-init-test>
    COMMAND
      ${CMAKE_COMMAND} -E env DEVICE=cpu
      $<TARGET_FILE:mlx-distributed-terminal-test>
    DEPENDS mlx-arraydesc-retirement-test mlx-patch-test-binary mlx-submission-retirement-fairness-test
            mlx-submission-retirement-pass-test mlx-submission-registry-init-test
            mlx-distributed-terminal-test
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
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/prepared_input.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/device_initialization.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/scheduler_initialization.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/original_buffer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/owned_host_copy.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/prefill_roots.cpp
    mlx/c/prefill_failure.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/closure.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/compile.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/cuda.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/device.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/distributed.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/distributed_group.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/distributed_eval.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/distributed_completion.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/distributed_persistent.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/error.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/event.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/export.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/fast.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/fft.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/graph_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/host_transfer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/ordinary_host_transfer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/ordinary_control.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/ordinary_array_controls.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/ordinary_fast_controls.cpp
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
  ${CMAKE_CURRENT_LIST_DIR}/mlx/c/stream_copy.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/stream_registration.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/gpu_stream_registration.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/cpu_worker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/submission.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/string.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/transforms.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/transforms_impl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/vector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlx/c/graph_rows.cpp
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

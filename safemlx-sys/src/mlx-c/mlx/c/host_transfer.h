// Copyright © 2026 Apple Inc.

#ifndef MLX_HOST_TRANSFER_H
#define MLX_HOST_TRANSFER_H

#include <stddef.h>

#include "mlx/c/array.h"
#include "mlx/c/event.h"
#include "mlx/c/stream.h"
#include "mlx/c/prepared_input.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_host_transfer Host transfer buffers
 * Backend-independent host transfer storage and asynchronous copies.
 */
/**@{*/

/** An owning, host-addressable transfer allocation. */
typedef struct mlx_host_transfer_buffer_ {
  void* ctx;
  // Null for ordinary wrappers; closed original Graph-owned destruction only.
  void* prepared_owner;
} mlx_host_transfer_buffer;

/** Source-owned immutable buffer construction. Separate from operation scopes. */
typedef struct mlx_prepared_host_transfer_layout {
  size_t metadata_bytes, backing_bytes, logical_bytes, controls;
} mlx_prepared_host_transfer_layout;
unsigned mlx_prepared_host_transfer_layout_for(mlx_prepared_host_transfer_layout*,
    mlx_prepared_input_runtime, const int*, size_t, mlx_dtype, size_t array_handles);
unsigned mlx_prepared_host_transfer_new(mlx_host_transfer_buffer*, void**,
    mlx_prepared_input_runtime, mlx_submission_graph_quota, const int*, size_t, mlx_dtype, size_t array_handles);
unsigned mlx_prepared_host_transfer_array(mlx_array*, void*);
void mlx_prepared_host_transfer_free(void*);
void* mlx_prepared_host_transfer_data(void*);
uint64_t mlx_prepared_host_transfer_identity(void*);

/** Requested semantics for host transfer storage. */
typedef enum mlx_host_transfer_policy_ {
  MLX_HOST_TRANSFER_POLICY_TRANSFER,
  MLX_HOST_TRANSFER_POLICY_MANAGED,
} mlx_host_transfer_policy;

unsigned mlx_copy_from_host_operation(mlx_array*, mlx_operation_event,
    mlx_host_transfer_buffer, mlx_stream);
unsigned mlx_copy_to_host_operation(mlx_host_transfer_buffer*, mlx_operation_event,
    mlx_array, mlx_host_transfer_policy, mlx_stream);
// Exact exclusive destination created by PreparedHostTransferSource; no mutable
// host view or Array alias may be outstanding during this one-use submission.
unsigned mlx_copy_to_prepared_host_operation(mlx_array*, mlx_operation_event,
    mlx_array, void* prepared_destination, mlx_stream);
bool mlx_original_host_store_layout(size_t rank, mlx_dtype dtype,
    size_t* controls, size_t* direct_graph_extent);
// Only original wrappers have an embedded preallocated deferred-cleanup node.
void mlx_host_transfer_buffer_defer_original(mlx_host_transfer_buffer);
size_t mlx_operation_host_transfer_control_bytes(void);

/** Physical storage selected by the active MLX backend. */
typedef enum mlx_host_transfer_storage_kind_ {
  MLX_HOST_TRANSFER_STORAGE_CPU,
  MLX_HOST_TRANSFER_STORAGE_METAL_SHARED,
  MLX_HOST_TRANSFER_STORAGE_CUDA_PINNED,
  MLX_HOST_TRANSFER_STORAGE_CUDA_MANAGED,
} mlx_host_transfer_storage_kind;

/** Process-wide charged backing-allocation telemetry for one storage kind. */
typedef struct mlx_host_transfer_memory_stats_ {
  size_t active_bytes;
  size_t peak_bytes;
  size_t active_allocations;
  size_t peak_allocations;
} mlx_host_transfer_memory_stats;

/** Read current and peak charged host-transfer backing telemetry. */
int mlx_host_transfer_memory_stats_get(
    mlx_host_transfer_memory_stats* stats,
    mlx_host_transfer_storage_kind kind);

/** Reset one storage kind's peaks to its current occupancy. */
int mlx_host_transfer_memory_stats_reset_peak(
    mlx_host_transfer_storage_kind kind);

/** Return the charged backing bound for pre-allocation admission. */
int mlx_host_transfer_capacity_upper_bound(
    size_t* capacity,
    size_t nbytes,
    mlx_host_transfer_policy policy);

/** Ordinary constructor controls; no native allocation or execution grant. */
unsigned mlx_ordinary_host_buffer_capacity(
    size_t* capacity, mlx_memory_placement* placement,
    mlx_prepared_input_runtime runtime, size_t bytes);

bool mlx_ordinary_host_buffer_observed_controls(size_t*, mlx_prepared_input_runtime, size_t rank);
size_t mlx_ordinary_host_buffer_wrapper_controls(size_t rank);
/** Only the ordinary copy wrapper; graph/evaluation facts remain separate. */
size_t mlx_ordinary_host_copy_wrapper_controls(void);
size_t mlx_ordinary_array_to_host_wrapper_controls(void);
size_t mlx_ordinary_event_wrapper_controls(void);
size_t mlx_ordinary_stream_clone_wrapper_controls(void);

/** Allocate an uninitialized typed host transfer buffer. */
int mlx_host_transfer_buffer_new(
    mlx_host_transfer_buffer* buffer,
    const int* shape,
    int dim,
    mlx_dtype dtype,
    mlx_host_transfer_policy policy);

/** Free a host transfer buffer. Submitted work retains its allocation. */
int mlx_host_transfer_buffer_free(mlx_host_transfer_buffer buffer);

/**
 * Allocate a host transfer buffer and asynchronously copy `source` into it.
 *
 * `event` covers the complete copy. The buffer must not be read until the
 * event completes.
 */
int mlx_copy_to_host(
    mlx_host_transfer_buffer* buffer,
    mlx_event* event,
    mlx_array source,
    mlx_host_transfer_policy policy,
    mlx_stream stream);

/**
 * Asynchronously copy a host transfer buffer into a new array on `stream`.
 * `event` covers the complete copy.
 */
int mlx_copy_from_host(
    mlx_array* array,
    mlx_event* event,
    mlx_host_transfer_buffer source,
    mlx_stream stream);

int mlx_host_transfer_buffer_data(
    const void** data,
    mlx_host_transfer_buffer buffer);
int mlx_host_transfer_buffer_data_mut(
    void** data,
    mlx_host_transfer_buffer buffer);
int mlx_host_transfer_buffer_shape(
    const int** shape,
    mlx_host_transfer_buffer buffer);
int mlx_host_transfer_buffer_ndim(
    size_t* ndim,
    mlx_host_transfer_buffer buffer);
int mlx_host_transfer_buffer_size(
    size_t* size,
    mlx_host_transfer_buffer buffer);
int mlx_host_transfer_buffer_nbytes(
    size_t* nbytes,
    mlx_host_transfer_buffer buffer);
/** Return the charged backing-allocation extent. */
int mlx_host_transfer_buffer_capacity(
    size_t* capacity,
    mlx_host_transfer_buffer buffer);
/** Non-reused generation shared with certified host-backed arrays; zero is unknown. */
int mlx_host_transfer_buffer_allocation_identity(
    uint64_t* identity,
    mlx_host_transfer_buffer buffer);
/* Same ownership/outcome contract as the prepared Array attachment. */
int mlx_host_transfer_buffer_attach_prepared_allocation_owner(
    int* outcome, mlx_host_transfer_buffer buffer, void* node,
    void* payload, void (*release)(void*));
/* Fixed call transports only; prepared owner nodes are separately supplied. */
size_t mlx_host_transfer_buffer_prepared_owner_control_bytes(void);

/**
 * Attach directly to shared host storage without creating/evaluating an array.
 * Ownership transfers only on success with attached=true. Otherwise payload
 * remains caller-owned. The nonthrowing release callback may run on any native
 * thread and must not block, allocate or reenter MLX. Caller serializes attachment.
 */
int mlx_host_transfer_buffer_retain_allocation_owner(
    bool* attached,
    mlx_host_transfer_buffer buffer,
    void* payload,
    void (*release)(void*));
int mlx_host_transfer_buffer_dtype(
    mlx_dtype* dtype,
    mlx_host_transfer_buffer buffer);
int mlx_host_transfer_buffer_policy(
    mlx_host_transfer_policy* policy,
    mlx_host_transfer_buffer buffer);
int mlx_host_transfer_buffer_storage_kind(
    mlx_host_transfer_storage_kind* kind,
    mlx_host_transfer_buffer buffer);

// Actual fixed copy controls and the existing direct C output Graph shell.
bool mlx_original_host_copy_layout(size_t rank, mlx_dtype dtype,
    size_t* controls, size_t* direct_graph_extent);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif

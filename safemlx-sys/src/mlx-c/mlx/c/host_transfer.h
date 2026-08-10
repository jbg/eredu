// Copyright © 2026 Apple Inc.

#ifndef MLX_HOST_TRANSFER_H
#define MLX_HOST_TRANSFER_H

#include <stddef.h>

#include "mlx/c/array.h"
#include "mlx/c/event.h"
#include "mlx/c/stream.h"

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
} mlx_host_transfer_buffer;

/** Requested semantics for host transfer storage. */
typedef enum mlx_host_transfer_policy_ {
  MLX_HOST_TRANSFER_POLICY_TRANSFER,
  MLX_HOST_TRANSFER_POLICY_MANAGED,
} mlx_host_transfer_policy;

/** Physical storage selected by the active MLX backend. */
typedef enum mlx_host_transfer_storage_kind_ {
  MLX_HOST_TRANSFER_STORAGE_CPU,
  MLX_HOST_TRANSFER_STORAGE_METAL_SHARED,
  MLX_HOST_TRANSFER_STORAGE_CUDA_PINNED,
  MLX_HOST_TRANSFER_STORAGE_CUDA_MANAGED,
} mlx_host_transfer_storage_kind;

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
int mlx_host_transfer_buffer_capacity(
    size_t* capacity,
    mlx_host_transfer_buffer buffer);
int mlx_host_transfer_buffer_dtype(
    mlx_dtype* dtype,
    mlx_host_transfer_buffer buffer);
int mlx_host_transfer_buffer_policy(
    mlx_host_transfer_policy* policy,
    mlx_host_transfer_buffer buffer);
int mlx_host_transfer_buffer_storage_kind(
    mlx_host_transfer_storage_kind* kind,
    mlx_host_transfer_buffer buffer);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif

/* Copyright © 2026 Apple Inc. */

#ifndef MLX_EVENT_H
#define MLX_EVENT_H

#include <stdbool.h>

#include "mlx/c/device.h"
#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_event Completion event
 * Backend-independent, single-shot completion events.
 */
/**@{*/

/** An owning opaque completion event. */
typedef struct mlx_event_ {
  void* ctx;
} mlx_event;

/** Backend identity of a completion event. */
typedef enum mlx_event_backend_ {
  MLX_EVENT_BACKEND_NONE,
  MLX_EVENT_BACKEND_CPU,
  MLX_EVENT_BACKEND_METAL,
  MLX_EVENT_BACKEND_CUDA,
} mlx_event_backend;

/** Return a new, already-complete event without a device identity. */
mlx_event mlx_event_new(void);

/** Free an event handle. Queued producer and consumer work retains it. */
int mlx_event_free(mlx_event event);

/** Block the host until completion and report retained async errors. */
int mlx_event_synchronize(mlx_event event);

/**
 * Nonblocking completion query.
 *
 * On success, writes a monotonic completion value to `complete`. A completed
 * event with a retained asynchronous error returns failure instead.
 */
int mlx_event_query(bool* complete, mlx_event event);

/**
 * Order later work on `stream` after `event` without blocking the host.
 *
 * Producer and consumer devices must match. This orders backend work which is
 * subsequently submitted on the stream; constructing lazy MLX operations does
 * not itself submit them.
 */
int mlx_stream_wait_event(mlx_stream stream, mlx_event event);

/** Return whether this event has a producer device identity. */
int mlx_event_has_device(bool* has_device, mlx_event event);

/**
 * Return the producer device.
 *
 * Fails for identity-free events, such as events produced for empty or
 * already-available output sets.
 */
int mlx_event_get_device(mlx_device* device, mlx_event event);

/** Return the producer backend. */
int mlx_event_get_backend(mlx_event_backend* backend, mlx_event event);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif

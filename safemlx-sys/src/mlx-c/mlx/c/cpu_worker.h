#pragma once
#include <stddef.h>
#include <stdint.h>
#include "mlx/c/stream.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mlx_cpu_worker_target_ {
  uint64_t scheduler_identity;
  int stream_index;
  int device_index;
  const void* stream_birth;
} mlx_cpu_worker_target;
typedef struct mlx_cpu_worker_result_ {
  unsigned cause;
  int system_value;
  unsigned system_category;
} mlx_cpu_worker_result;
typedef struct mlx_cpu_worker_layout_ {
  size_t entry_bytes, entry_alignment;
  size_t fallback_control_bytes, fallback_char_bytes;
  size_t thread_handle_bytes, thread_implementation_bytes, thread_packet_bytes;
  size_t failure_requests, controls;
} mlx_cpu_worker_layout;
typedef struct mlx_cpu_worker_static_layout_ {
  size_t bytes;
  unsigned qualified;
} mlx_cpu_worker_static_layout;
void mlx_cpu_worker_static_layout_for(mlx_cpu_worker_static_layout*);
mlx_cpu_worker_result mlx_cpu_worker_layout_for(mlx_cpu_worker_layout*);
/* Captures scalar identity only; no owning wrapper or worker allocation. */
mlx_cpu_worker_result mlx_cpu_worker_target_for(mlx_cpu_worker_target*,
    uint64_t scheduler_identity, mlx_stream, const void* stream_birth);
/* Failed construction preserves output/source after all native prefixes retire.
   Success keeps source permanently with the process worker. Birth only: future
   tasks, error() Strings and mutable diagnostic aliases require separate bounds. */
mlx_cpu_worker_result mlx_cpu_worker_initialize(const void** worker_birth,
    mlx_cpu_worker_layout, mlx_cpu_worker_target, void* owner, void (*retire)(void*));
mlx_cpu_worker_result mlx_cpu_worker_borrow(mlx_cpu_worker_target, const void* worker_birth);
/* Same source authentication plus terminal FIFO/capture observation; never waits. */
mlx_cpu_worker_result mlx_cpu_worker_observe_idle(mlx_cpu_worker_target, const void* worker_birth);
#ifdef __cplusplus
}
#endif

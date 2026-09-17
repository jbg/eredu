#pragma once
#include <stddef.h>
#include <stdint.h>
#include "mlx/c/stream.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mlx_gpu_stream_target_ {
  uint64_t device_identity, scheduler_identity;
} mlx_gpu_stream_target;
typedef struct mlx_gpu_stream_registration_layout_ {
  size_t bytes[6], alignments[6];
  size_t wrapper_bytes, wrapper_alignment, controls;
  size_t platform_queues, platform_buffers, autorelease_pools;
} mlx_gpu_stream_registration_layout;
unsigned mlx_gpu_stream_registration_layout_for(mlx_gpu_stream_registration_layout*);
unsigned mlx_stream_register_gpu(mlx_stream*, mlx_gpu_stream_registration_layout,
    mlx_gpu_stream_target, void*, void (*)(void*));
unsigned mlx_stream_gpu_registration_borrow(mlx_stream, const void*, mlx_gpu_stream_target);
/* Read-only terminal observation; no encoder construction, progress or wait.
   Caller serializes mutation. Success grants no right to mutate a session. */
unsigned mlx_stream_gpu_registration_observe_idle(mlx_stream, const void*, mlx_gpu_stream_target);
#ifdef __cplusplus
}
#endif

#pragma once
#include <stddef.h>
#include <stdbool.h>
#include "mlx/c/stream.h"
#ifdef __cplusplus
extern "C" {
#endif
/* An exact scalar snapshot of an existing Stream value. No queue or worker. */
typedef struct mlx_stream_copy_value_ {
  int index;
  int device_index;
  unsigned device_kind;
  unsigned cpu_matmul;
} mlx_stream_copy_value;
typedef struct mlx_stream_copy_layout_ {
  size_t wrapper_bytes;
  size_t wrapper_alignment;
  size_t controls;
} mlx_stream_copy_layout;
/* Descriptive CPU implementation facts. No stream, queue or source grant. */
typedef struct mlx_cpu_matmul_facts_ {
  size_t tile_edge, reduction_lanes, max_rank, max_elements, controls;
  bool float16_tiles, platform_float16_tiles;
} mlx_cpu_matmul_facts;
unsigned mlx_cpu_matmul_facts_for(mlx_cpu_matmul_facts*);
/* Select only on an already captured CPU scalar value, no allocation. */
unsigned mlx_stream_copy_select_cpu_matmul(mlx_stream_copy_value*, unsigned kernel);
/* Fixed status: success0, invalid2, allocation_failed4. Unchanged outputs on refusal. */
unsigned mlx_stream_copy_snapshot(mlx_stream_copy_value*, mlx_stream);
unsigned mlx_stream_copy_layout_for(mlx_stream_copy_layout*);
unsigned mlx_stream_copy_new(mlx_stream*, mlx_stream_copy_value);
/* Only a wrapper returned by copy_new; no runtime, error text or callback. */
void mlx_stream_copy_free(mlx_stream);
#ifdef __cplusplus
}
#endif

#pragma once
#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#include "mlx/c/submission.h"
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
// Additional metadata allocated by the existing GPU reshape/concatenate/cast
// workers at the supplied source rank. This is a cold profile, not an allocator.
typedef struct mlx_graph_copy_worker_layout_ {
  size_t allocation_extents, named_control_bytes;
} mlx_graph_copy_worker_layout;
bool mlx_graph_copy_worker_inspect(mlx_graph_copy_worker_layout*, size_t rank,
    size_t reshapes, size_t concatenations, size_t concatenate_inputs);
// The same rank-dependent worker union for one byte reinterpretation, including
// its possible general-copy fallback. No source/dtype or admission authority.
bool mlx_graph_byte_view_worker_inspect(mlx_graph_copy_worker_layout*, size_t rank);
bool mlx_reshape_like_prefix_control_bytes(size_t*);
// Cold geometry/control proof for the same selected convolution constructor.
// Normalized source shapes are [N,H,W,C], with W=1 for one spatial dimension.
typedef struct mlx_original_convolution_source_ {
  size_t spatial_dimensions;
  int input[4], weight[4], stride[2], padding[2], dilation[2], groups;
} mlx_original_convolution_source;
typedef struct mlx_original_convolution_layout_ {
  size_t primitives, edges, backing_births, named_control_bytes;
  int output[4];
} mlx_original_convolution_layout;
bool mlx_original_convolution_inspect(mlx_original_convolution_layout*,
    const mlx_original_convolution_source*);
// A finite construction-only row owner. Header and elements use the selected
// Graph allocator; no independent Scope, Record, execution or budget is created.
typedef struct mlx_graph_array_rows_ { void* ctx; } mlx_graph_array_rows;
typedef struct mlx_graph_array_rows_layout_ {
  size_t capacity, maximum_operands, resident_controls, named_control_bytes;
} mlx_graph_array_rows_layout;
bool mlx_graph_array_rows_inspect(mlx_graph_array_rows_layout*, size_t capacity, size_t rank);
unsigned mlx_graph_array_rows_new(mlx_graph_array_rows*, mlx_submission_observer, size_t capacity);
unsigned mlx_graph_array_rows_push(mlx_graph_array_rows, mlx_submission_observer, mlx_array);
unsigned mlx_graph_array_rows_concatenate(mlx_array*, mlx_graph_array_rows,
    mlx_submission_observer, int axis, mlx_stream);
void mlx_graph_array_rows_free(mlx_graph_array_rows);
// Same ordinary reshape constructor, with a borrowed source prefix. The Shape
// destination uses the native Graph allocator when entered under original work.
int mlx_reshape_like_prefix(mlx_array*, mlx_array input, mlx_array shape_source,
    int final_dimension, mlx_stream);
#ifdef __cplusplus
}
#endif

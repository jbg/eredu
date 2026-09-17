#ifndef MLX_OWNED_HOST_COPY_C_H
#define MLX_OWNED_HOST_COPY_C_H
#include "mlx/c/prepared_input.h"
#include "mlx/c/original_buffer.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Explicit copied immutable source storage, never ordinary pointer import. */
enum { MLX_OWNED_HOST_COPY_STRATEGY = 1 };
typedef struct mlx_owned_host_copy_layout {
  size_t metadata_bytes;
  size_t backing_bytes;
  size_t copy_bytes;
  size_t handle_bytes;
  size_t handle_alignment;
  size_t controls;
  unsigned strategy;
} mlx_owned_host_copy_layout;
typedef struct mlx_owned_host_copy_slot { void* ctx; } mlx_owned_host_copy_slot;
/* Fixed statuses: success0 invalid1 busy2 allocation3 capacity4 identity5
 * domain6 spent7 unsupported8 failed9. No callback consumes caller input. */
unsigned mlx_owned_host_copy_layout_for(mlx_owned_host_copy_layout*,
    mlx_prepared_input_runtime, const int* shape, size_t rank, mlx_dtype);
/* Cold metadata/final handle only; no native payload allocation or evaluation. */
unsigned mlx_owned_host_copy_new(mlx_owned_host_copy_slot*,
    mlx_prepared_input_runtime, mlx_submission_graph_quota,
    const int* shape, size_t rank, mlx_dtype);
void mlx_owned_host_copy_free(mlx_owned_host_copy_slot);
/* Borrow exact initialized bytes for the call. Failure never publishes output.
 * Preflight refusal leaves Ready; the physical allocation attempt spends slot.
 * Success publishes the cold arena-owned array handle using the existing
 * mlx_array prepared-owner destruction contract. */
unsigned mlx_owned_host_copy_fill(mlx_array*, mlx_owned_host_copy_slot,
    mlx_submission_observer, const void* data, size_t bytes);
/* Same worker, with successful actual Data birth facts. Facts remain unchanged
 * on refusal; they are descriptive and carry no funding authority. */
unsigned mlx_owned_host_copy_fill_completed(mlx_array*, mlx_original_buffer_info*,
    mlx_owned_host_copy_slot, mlx_submission_observer, const void* data, size_t bytes);
/* Read-only: 0 Ready, 1 attempted/spent, 2 invalid handle. */
unsigned mlx_owned_host_copy_state(mlx_owned_host_copy_slot);
#ifdef __cplusplus
}
#endif
#endif

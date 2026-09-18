/* Fixed root populations for one source-bound prefill completion site. */
#ifndef MLX_PREFILL_ROOTS_H
#define MLX_PREFILL_ROOTS_H
#include <stddef.h>
#include "mlx/c/prefill_failure.h"
#include "mlx/c/array.h"
#include "mlx/c/event.h"
#include "mlx/c/submission.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlx_prefill_roots_ { void* ctx; } mlx_prefill_roots;
typedef struct mlx_prefill_roots_layout_ {
  size_t capacity;
  size_t owner_bytes;
  size_t owner_alignment;
  size_t graph_bytes;
  size_t native_controls;
  size_t validation_descriptor_minimum;
} mlx_prefill_roots_layout;
/* Fixed status: 0 success; 1 invalid; 2 capacity; 3 allocation; 4 domain;
 * 5 spent; 6 pending. 7 preserves the actual native exception via mlx_error.
 * Source/runtime preparation, original admission and retirement authority are
 * supplied by the closed caller, never inferred from these diagnostics. */
unsigned mlx_prefill_roots_layout_for(mlx_prefill_roots_layout* out, size_t capacity);
unsigned mlx_prefill_roots_new(mlx_prefill_roots* out, size_t capacity,
                             mlx_submission_graph_quota graph);
unsigned mlx_prefill_roots_append(mlx_prefill_roots roots, mlx_array value);
// Exact same-scope completed root; spent append capacity is never refunded.
unsigned mlx_prefill_roots_retire_completed_current(mlx_prefill_roots roots, mlx_array value);
/* Original validation append authenticates the descriptor's Graph domain and
 * retains each descriptor once. Ordinary arbitrary append has no such bound. */
unsigned mlx_prefill_roots_append_validation(mlx_prefill_roots roots, mlx_array value);
unsigned mlx_prefill_roots_submit(mlx_prefill_roots roots);
unsigned mlx_prefill_roots_wait(mlx_prefill_roots roots);
unsigned mlx_prefill_roots_query(mlx_prefill_roots roots);
unsigned mlx_prefill_roots_validate(mlx_prefill_roots roots);
size_t mlx_prefill_roots_size(mlx_prefill_roots roots);
/* Called only under the closed runtime/retirement owner. No wait, callback,
 * evaluation or error translation occurs here. Source/Graph/Record authority
 * remains with the caller until the native values and this shell retire. */
/* Same fixed roots, with a preconstructed failure owner. No post-hoc grant. */
unsigned mlx_prefill_roots_new_owned(
    mlx_prefill_roots* out, size_t capacity, mlx_submission_graph_quota graph,
    mlx_prefill_failure failure);
unsigned mlx_prefill_roots_bind(mlx_prefill_roots roots, mlx_submission_scope scope);
unsigned mlx_prefill_roots_submit_scoped(mlx_prefill_roots roots, mlx_submission_scope scope);
unsigned mlx_prefill_roots_query_scoped(mlx_prefill_roots roots, mlx_submission_scope scope);
unsigned mlx_prefill_roots_wait_scoped(mlx_prefill_roots roots, mlx_submission_scope scope);
unsigned mlx_prefill_roots_validate_scoped(mlx_prefill_roots roots, mlx_submission_scope scope);
/* Installed transaction-only completion. Actual current scope must match its
 * one-shot binding. Sealed observation uses the explicit retained-scope APIs. */
unsigned mlx_prefill_roots_complete_current(mlx_prefill_roots roots);
/* Same prepared roots and carrier; also closes the exact selected stream frontier.
 * A wrong owner or unprepared stream refuses before the collector is consumed. */
unsigned mlx_prefill_roots_complete_current_on_stream(mlx_prefill_roots roots, mlx_stream stream);
/* The same collector/selected stream with caller-owned finite traversal limits.
 * Limits are checked before submitted state or roots change. No count is
 * inferred from arena headroom; caller retains the complete lowering source. */
unsigned mlx_prefill_roots_complete_current_on_stream_prepared(
    mlx_prefill_roots roots, mlx_stream stream,
    const mlx_operation_eval_traversal_limits* limits);
/* Current original-required array evaluation. error receives one owned alias
 * of the already-bound carrier on every outcome when available; caller frees
 * it. No fresh error/carrier/control allocation or legacy textual handler. */
unsigned mlx_array_eval_scoped(mlx_array value, mlx_prefill_failure* error);
/* TLS-only existing-carrier retention for a runtime-busy fixed refusal. */
unsigned mlx_array_eval_scoped_failure(mlx_prefill_failure* error);
/* Exact named native synchronous evaluation controls; no construction. */
size_t mlx_array_eval_scoped_control_bytes(void);
/* Completed contiguous source only. Uses the ordinary typed data initializer
 * under existing original controls; never evaluates or selects another stream. */
unsigned mlx_array_deep_copy_scoped(mlx_array* out, mlx_array source,
                                  mlx_prefill_failure* error);
size_t mlx_array_deep_copy_scoped_control_bytes(size_t maximum_rank);
void mlx_prefill_roots_free(mlx_prefill_roots roots);
#ifdef __cplusplus
}
#endif
#endif

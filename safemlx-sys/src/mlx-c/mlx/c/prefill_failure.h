#ifndef MLX_PREFILL_FAILURE_H
#define MLX_PREFILL_FAILURE_H
#include <stddef.h>
#include "mlx/c/submission.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mlx_prefill_failure_ { void* ctx; } mlx_prefill_failure;
typedef struct mlx_prefill_failure_layout_ {
  size_t owner_bytes;
  size_t owner_alignment;
  size_t native_controls;
  size_t retirement_controls;
} mlx_prefill_failure_layout;
typedef struct mlx_prefill_failure_view_ {
  unsigned kind;
  const char* message;
  size_t message_size;
  const char* source_type;
  size_t source_type_size;
  long long native_code;
} mlx_prefill_failure_view;
/* No construction, callback, TLS access or native work. */
unsigned mlx_prefill_failure_layout_for(mlx_prefill_failure_layout* out);
/* Failure leaves owner unchanged; success consumes it exactly once. */
unsigned mlx_prefill_failure_new_retaining(
    mlx_prefill_failure* out, void* owner, void (*retire)(void*));
unsigned mlx_submission_observer_failure(
    mlx_prefill_failure* out, mlx_submission_observer value);
void mlx_prefill_failure_retain(mlx_prefill_failure value);
void mlx_prefill_failure_free(mlx_prefill_failure value);
/* 0=immutable retained cause, 1=not captured, 2=publication in progress.
 * Borrowed views never outlive a retained handle; no formatting/error channel. */
unsigned mlx_prefill_failure_view_get(mlx_prefill_failure_view* out, mlx_prefill_failure value);
/* Native description/domain use copied UTF-16, never autoreleased C pointers.
 * field 0=description, 1=domain. Caller owns output; no native string allocation. */
unsigned mlx_prefill_failure_native_text(
    size_t* used, size_t* remaining, mlx_prefill_failure value, unsigned field,
    size_t offset, uint16_t* output, size_t capacity);
/* Bind a preallocated final-index/operation failure to its actual empty Scope.
 * No root collector or completion is manufactured by this binding. */
unsigned mlx_prefill_failure_bind_original_scope(
    mlx_prefill_failure failure, mlx_submission_scope scope);

#ifdef __cplusplus
}
#endif
#endif

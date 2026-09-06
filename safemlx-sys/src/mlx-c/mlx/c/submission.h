#ifndef MLX_SUBMISSION_H
#define MLX_SUBMISSION_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlx_submission_scope_ {
  void* ctx;
} mlx_submission_scope;

typedef enum mlx_submission_activity_ {
  MLX_SUBMISSION_ACTIVITY_NONE = 0,
  MLX_SUBMISSION_ACTIVITY_PENDING = 1,
  MLX_SUBMISSION_ACTIVITY_TERMINAL = 2,
} mlx_submission_activity;

typedef struct mlx_submission_status_ {
  mlx_submission_activity activity;
  bool failed;
  bool blocked;
} mlx_submission_status;

// Begin before any operation that may submit work. Scopes are thread-affine,
// nestable, and may be sealed/freed out of order. The error channel is not used.
// Initialize the output handle to {NULL}; a failed constructor leaves it empty.
int mlx_submission_scope_new(mlx_submission_scope* out);
int mlx_submission_scope_seal(mlx_submission_scope scope);
int mlx_submission_scope_query(
    mlx_submission_status* out, mlx_submission_scope scope);
// Advances lifetime evidence without reclaiming native graph/resource owners.
// Reclamation is deferred to an ordinary native operation on the owner thread.
int mlx_submission_scope_progress(
    mlx_submission_status* out, mlx_submission_scope scope);
int mlx_submission_scope_free(mlx_submission_scope scope);

#ifdef __cplusplus
}
#endif
#endif

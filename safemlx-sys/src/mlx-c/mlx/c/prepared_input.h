#ifndef MLX_PREPARED_INPUT_C_H
#define MLX_PREPARED_INPUT_C_H
#include <stddef.h>
#include <stdint.h>
#include "mlx/c/array.h"
#include "mlx/c/submission.h"
#include "mlx/c/stream.h"
struct mlx_original_buffer_budget;
#ifdef __cplusplus
extern "C" {
#endif
// Exact borrowed target equality; no allocation, global runtime entry or errors.
int mlx_prepared_input_target_matches(mlx_stream, mlx_device);
size_t mlx_prepared_input_target_controls(void);
// Borrowed runtime context. It may follow ordinary or admitted initialization.
// No owned caller arrays enter it.
typedef struct mlx_prepared_input_runtime {
  void* allocator;
  size_t page_size;
  size_t maximum;
  unsigned storage_kind;
  size_t controls;
} mlx_prepared_input_runtime;
// Shared allocator initialization. Dynamic object storage is distinct from
// fixed module storage and from the borrowed Device ownership prerequisite.
typedef struct mlx_input_allocator_layout {
  size_t static_bytes;
  size_t object_bytes;
  size_t controls;
  unsigned qualified;
  unsigned requires_device;
} mlx_input_allocator_layout;
// Pure layout query; no Device/allocator initialization or runtime loan.
void mlx_input_allocator_layout_for(mlx_input_allocator_layout*);
// Fixed statuses: 0 success, 1 unsupported, 2 invalid/context, 3 busy,
// 4 allocation failure, 5 missing Device, 6 ordinary predecessor,
// 7 competing admitted predecessor, 8 identity mismatch, 9 identity exhaustion.
// Success alone consumes
// owner. Every refusal preserves output and owner; no error-handler call.
unsigned mlx_input_allocator_initialize(mlx_prepared_input_runtime*, uint64_t*,
    void* owner, void (*retire)(void*));
// Pure per-call transient controls; zero means this implementation is unqualified.
size_t mlx_input_allocator_borrow_controls(void);
// Borrows only the matching admitted singleton, including during an active role.
// Does not initialize, wait, allocate, or alter the supplied output on refusal.
unsigned mlx_input_allocator_borrow(mlx_prepared_input_runtime*, uint64_t identity);
typedef struct mlx_prepared_input_source {
  const void* data;
  const size_t* shape;
  size_t rank;
  size_t elements;
  unsigned kind; /* 0 U32, 1 I32, 2 F32, 3 Boolean; 4..9 initialized zeros: F16, BF16, F32, I32, U32, Bool */
} mlx_prepared_input_source;
typedef struct mlx_prepared_input_layout {
  size_t metadata_bytes;
  size_t backing_bytes;
  size_t controls;
} mlx_prepared_input_layout;
typedef struct mlx_prepared_input_leaf { void* ctx; } mlx_prepared_input_leaf;
// Initialization is an explicitly ordinary prerequisite, not numeric preflight.
int mlx_prepared_input_runtime_prepare(mlx_prepared_input_runtime*);
int mlx_prepared_input_layout_for(mlx_prepared_input_layout*, mlx_prepared_input_runtime,
    mlx_prepared_input_source);
unsigned mlx_prepared_input_leaf_new(mlx_prepared_input_leaf*, mlx_prepared_input_runtime,
    mlx_submission_graph_quota, mlx_prepared_input_source);
void mlx_prepared_input_leaf_free(mlx_prepared_input_leaf);
// Ordinary wrapper allocation; no numerical copy. No original construction claim.
int mlx_prepared_input_leaf_clone_array(mlx_array*, mlx_prepared_input_leaf);
// Pure concrete handle layout; closed arena construction with fixed status.
int mlx_prepared_input_array_layout(size_t* metadata_bytes, size_t* controls);
unsigned mlx_prepared_input_leaf_array(mlx_array*, mlx_prepared_input_leaf);
int mlx_prepared_input_leaf_info(uint64_t*, size_t*, mlx_prepared_input_leaf);
// Exact closed mutable U32[2] producer; this is not a generic tensor grant.
typedef struct mlx_original_mutable_pair_layout {
  size_t metadata_bytes;
  size_t backing_bytes;
  size_t copy_bytes;
  size_t record_minimum_capacity;
  size_t controls;
  size_t module_bytes;
  size_t thread_bytes;
  size_t graph_requests;
  unsigned constant_registry;
} mlx_original_mutable_pair_layout;
// Pure TLS check. No initialization, registry walk, allocator lookup or authority.
int mlx_original_mutable_pair_context_empty(void);
unsigned mlx_original_mutable_pair_layout_for(mlx_original_mutable_pair_layout*,
    mlx_prepared_input_runtime);
unsigned mlx_original_mutable_pair_new(mlx_array*, mlx_prepared_input_runtime,
    mlx_submission_graph_quota, mlx_submission_scope, struct mlx_original_buffer_budget,
    const uint32_t*);
typedef struct mlx_original_prompt_input_layout {
  size_t metadata_bytes;
  size_t backing_bytes;
  size_t controls;
  size_t ordinary_handle_bytes;
} mlx_original_prompt_input_layout;
// Pure layout for the selected runtime; no original authority is created.
unsigned mlx_original_prompt_input_layout_for(mlx_original_prompt_input_layout*,
    mlx_prepared_input_runtime, size_t elements);
// Exact one eager source in the current authenticated original Scope.
unsigned mlx_original_prompt_input_new(mlx_array*, const uint32_t*, size_t elements);
// Exact signed input through the same constructor and four-byte layout.
unsigned mlx_original_prediction_input_new(mlx_array*, const int32_t*, size_t elements);
#ifdef __cplusplus
}
#endif
#endif

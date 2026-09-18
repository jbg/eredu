#ifndef MLX_ORIGINAL_BUFFER_C_H
#define MLX_ORIGINAL_BUFFER_C_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "mlx/c/array.h"
#include "mlx/c/host_transfer.h"
#include "mlx/c/prepared_input.h"
#include "mlx/c/submission.h"
#ifdef __cplusplus
extern "C" {
#endif

// Fixed statuses. No error-handler allocation or completion claim.
enum {
  MLX_ORIGINAL_BUFFER_OK = 0,
  MLX_ORIGINAL_BUFFER_MISSING = 1,
  MLX_ORIGINAL_BUFFER_FOREIGN = 2,
  MLX_ORIGINAL_BUFFER_SCOPE = 3,
  MLX_ORIGINAL_BUFFER_BOUND = 4,
  MLX_ORIGINAL_BUFFER_LAYOUT = 5,
  MLX_ORIGINAL_BUFFER_CAPACITY = 6,
  MLX_ORIGINAL_BUFFER_BUSY = 7,
  MLX_ORIGINAL_BUFFER_ALLOCATION = 8,
  MLX_ORIGINAL_BUFFER_UNSUPPORTED = 9,
  MLX_ORIGINAL_BUFFER_IDENTITY = 10,
  MLX_ORIGINAL_BUFFER_UNCERTIFIED = 11,
  MLX_ORIGINAL_BUFFER_CHANGED = 12,
  MLX_ORIGINAL_BUFFER_NATIVE_CONTROL_BASE = 256,
  MLX_ORIGINAL_BUFFER_UNEXPECTED = 512
};
typedef struct mlx_original_buffer_budget { void* ctx; } mlx_original_buffer_budget;
typedef struct mlx_original_buffer_layout {
  size_t owner_bytes;
  size_t control_bytes;
} mlx_original_buffer_layout;
typedef struct mlx_original_buffer_population_layout {
  size_t capacity;
  size_t control_bytes;
} mlx_original_buffer_population_layout;
typedef struct mlx_original_buffer_info {
  bool known;
  uint64_t identity;
  size_t charged_bytes;
} mlx_original_buffer_info;

// Positive storage classifications; UNKNOWN never authorizes ordinary fallback.
enum {
  MLX_ORDINARY_BUFFER_UNKNOWN = 0,
  MLX_ORDINARY_BUFFER_EMPTY = 1,
  MLX_ORDINARY_BUFFER_ALLOCATION = 2
};
// Reuses the descriptor kernel. EMPTY proves no physical allocation; CPU zero
// original backing still has a real birth and is not EMPTY. Input/host/custom
// storage has no ordinary witness and must retain its own source proof.
unsigned mlx_ordinary_buffer_array_info(uint32_t* kind,
    mlx_original_buffer_info*, mlx_array);
// Same exclusive node contract as original attachment. Rechecks positive ordinary
// kind AND identity/capacity before append, using one serialized Data observation.
unsigned mlx_ordinary_buffer_array_attach(mlx_array,
    const mlx_original_buffer_info*, void* node, void* payload, void (*release)(void*));

// Immutable prepared source observations never authorize new accounting rows.
// UNKNOWN/EMPTY/ALLOCATION use the existing fixed classification values above.
// EMPTY includes the actual zero-copy Data with no generation or native buffer.
unsigned mlx_immutable_source_array_info(uint32_t* kind,
    mlx_original_buffer_info*, mlx_array);
unsigned mlx_immutable_source_array_attach(mlx_array,
    const mlx_original_buffer_info*, void* node, void* payload, void (*release)(void*));

// Positive immutable Host constructor provenance. Neither kind authorizes a birth.
typedef struct mlx_immutable_host_transfer_info {
  mlx_original_buffer_info backing;
  bool prepared_source;
} mlx_immutable_host_transfer_info;
unsigned mlx_immutable_host_transfer_inspect(mlx_immutable_host_transfer_info*, mlx_host_transfer_buffer);
unsigned mlx_immutable_host_transfer_attach(mlx_host_transfer_buffer,
    const mlx_immutable_host_transfer_info*, void* node, void* payload, void (*release)(void*));
size_t mlx_immutable_host_transfer_control_bytes(void);

// Settled Host-transfer aliases retain their separate Host allocation namespace.
// No source account is granted: publication requires the existing exact prepaid
// immutable source. Every failed attachment preserves both exclusive nodes.
unsigned mlx_host_transfer_array_alias_info(mlx_immutable_host_transfer_info*, mlx_array);
unsigned mlx_host_transfer_array_alias_attach(mlx_array,
    const mlx_immutable_host_transfer_info*, void* node, void* payload, void (*release)(void*));
size_t mlx_host_transfer_array_alias_control_bytes(void);

// Exact completed native Data view of a certified Host-transfer backing. The
// nonrecycled Data generation distinguishes independent reload wrappers of the
// same immutable Host allocation. A view attachment follows this Data and its
// shared views/recovery pins, not unrelated owners of the immutable Host buffer.
typedef struct mlx_host_transfer_view_info {
  mlx_original_buffer_info backing;
  uint64_t view_identity;
} mlx_host_transfer_view_info;
unsigned mlx_host_transfer_array_view_info(mlx_host_transfer_view_info*, mlx_array);
unsigned mlx_host_transfer_array_view_attach(mlx_array,
    const mlx_host_transfer_view_info*, void* node, void* payload, void (*release)(void*));
size_t mlx_host_transfer_array_view_control_bytes(void);

// Pure owner/transport layouts; excludes physical capacity and Graph births.
unsigned mlx_original_buffer_layout_for(mlx_original_buffer_layout*);
// Pure original Metal population bound using this retained allocator. Input
// bytes bounds the sum of requested payloads; births bounds positive attempts.
// Each physical birth rounds separately. CPU headers are a separate mechanism.
unsigned mlx_original_buffer_metal_population_layout_for(
    mlx_original_buffer_population_layout*, mlx_prepared_input_runtime,
    size_t requested_bytes, size_t maximum_births);
// Exact one-birth capacity from this actual retained allocator, using the same
// malloc_original physical worker. Includes CPU zero-byte headers and Metal's
// empty allocation. Query only; false/refusal leaves out unchanged.
size_t mlx_original_buffer_request_control_bytes(void);
unsigned mlx_original_buffer_request_layout_for(
    mlx_original_buffer_population_layout*, mlx_prepared_input_runtime, size_t requested_bytes);
// Ordinary construction using an already prepared, actual allocator. Success
// alone consumes owner. Its callback must enqueue rather than destroy payload.
unsigned mlx_original_buffer_budget_new_retaining(mlx_original_buffer_budget*,
    mlx_prepared_input_runtime, size_t, void*, void (*)(void*));
void mlx_original_buffer_budget_retain(mlx_original_buffer_budget);
void mlx_original_buffer_budget_release(mlx_original_buffer_budget);
size_t mlx_original_buffer_budget_capacity(mlx_original_buffer_budget);
size_t mlx_original_buffer_budget_occupied(mlx_original_buffer_budget);
// Explicit only, on the current empty role after original-control enablement.
unsigned mlx_original_buffer_budget_bind(mlx_submission_scope, mlx_original_buffer_budget);
// Completed same-budget Data/birth facts. No evaluation, polling, clone or
// registration; a foreign birth rejects, ordinary/unknown backing stays unknown.
unsigned mlx_original_buffer_array_info(mlx_original_buffer_info*,
    mlx_array, mlx_original_buffer_budget);
// Authenticates only an actual settled mutable birth. No budget pointer or
// authority is returned; a neutral existing-entry lookup must establish origin.
unsigned mlx_original_buffer_array_alias_info(mlx_original_buffer_info*, mlx_array);
// Named native transport only, excluding the separately prepared owner nodes.
size_t mlx_original_buffer_inspection_control_bytes(void);
size_t mlx_original_buffer_attachment_control_bytes(void);
// Caller serializes the whole call and retains source and expected budget.
// Every nonzero status preserves the exclusive fresh node and payload. Zero
// appends to the same Data inspected/compared in this call, consumes both once,
// and cannot subsequently fail. No allocation, polling, handler or callback.
// Callback/node custody requirements match the existing prepared-owner API.
unsigned mlx_original_buffer_array_attach(mlx_array, mlx_original_buffer_budget,
    const mlx_original_buffer_info*, void* node, void* payload, void (*release)(void*));
unsigned mlx_original_buffer_array_alias_attach(mlx_array,
    const mlx_original_buffer_info*, void* node, void* payload, void (*release)(void*));
#ifdef __cplusplus
}
#endif
#endif

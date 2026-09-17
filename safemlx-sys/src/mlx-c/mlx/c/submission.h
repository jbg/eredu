#ifndef MLX_SUBMISSION_H
#define MLX_SUBMISSION_H

#include <stdbool.h>
#include <stddef.h>
#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlx_submission_scope_ {
  void* ctx;
} mlx_submission_scope;

// An alias of an already configured original role, never a child Scope. Its
// release does not seal the role. Values are owned and host-thread affine.
typedef struct mlx_submission_observer_ { void* ctx; } mlx_submission_observer;
// 0 retained; 11 ordinary context; 4 invalid original context. No allocation.
unsigned mlx_submission_observer_current(mlx_submission_observer* out);
unsigned mlx_submission_observer_retain(mlx_submission_observer value);
size_t mlx_submission_observer_control_bytes(void);
void mlx_submission_observer_release(mlx_submission_observer value);

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

unsigned mlx_submission_observer_query(
    mlx_submission_status* out, mlx_submission_observer value);
// ScopedEvaluation fixed status: 0 observed, 8 funded, 9 unknown, 10 busy.
// Query/progress need no current role; new native work always needs exact TLS.
unsigned mlx_submission_observer_progress(
    mlx_submission_status* out, mlx_submission_observer value);
// No progress or waits; exact-owner terminal record destruction outside locks.
unsigned mlx_submission_observer_retire(mlx_submission_observer value);
// Completed-only descriptor validation, including after role seal. No work is
// submitted, progressed or waited for. Unscheduled arrays and foreign pending
// descriptors are unchanged on refusal. A completed matching event is detached.
// Uses ScopedEvaluation statuses; the caller serializes native array access.
unsigned mlx_submission_observer_validate_array(
    mlx_submission_observer value, mlx_array array);

// Begin before any operation that may submit work. Scopes are thread-affine,
// nestable, and may be sealed/freed out of order. The error channel is not used.
// Initialize the output handle to {NULL}; a failed constructor leaves it empty.
int mlx_submission_scope_new(mlx_submission_scope* out);
// Constructor-only ownership. Both owner and retire must be non-null. Success
// consumes owner once; any failure leaves it with the caller and out unchanged.
// Final intrusive release deletes the complete Scope before invoking retire.
// Child storage is deleted before releasing its durable parent reference.
// retire must not throw, wait, allocate or run arbitrary destructors: it should
// only publish a preallocated retirement node. It must support any release thread.
// No source/backing/completion/funding guarantee is conferred by this handoff.
int mlx_submission_scope_new_retaining(
    mlx_submission_scope* out, void* owner, void (*retire)(void*));
/** Closed shared arena; capacity bounds only Record/container storage. */
typedef struct mlx_submission_record_quota_ { void* ctx; } mlx_submission_record_quota;
typedef struct mlx_submission_record_quota_layout_ {
  size_t capacity;
  size_t allocation_bytes;
  size_t alignment;
  size_t retirement_controls;
} mlx_submission_record_quota_layout;
/** Cold, no allocation. Zero succeeds, nonzero rejects the exact layout. */
// Pure minimum physical arena accepted by the existing native layout worker.
int mlx_submission_record_quota_minimum_capacity(size_t* out);
// Fresh-arena capacity for an authenticated whole-lifetime extent population.
int mlx_submission_record_quota_fresh_capacity(size_t* out, size_t extents);
/** Exact one-block physical storage, including allocator headers and rounding.
 * Unsupported alignment/overflow refuses without changing out; no allocation. */
int mlx_submission_record_quota_allocation_extent(
    size_t* out, size_t requested, size_t alignment);
int mlx_submission_record_quota_layout_for(
    mlx_submission_record_quota_layout* out, size_t capacity);
/** Failure consumes neither owner nor callback; success owns both. */
int mlx_submission_record_quota_new_retaining(
    mlx_submission_record_quota* out, size_t capacity,
    void* owner, void (*retire)(void*));
void mlx_submission_record_quota_retain(mlx_submission_record_quota quota);
void mlx_submission_record_quota_release(mlx_submission_record_quota quota);
size_t mlx_submission_record_quota_occupied(mlx_submission_record_quota quota);
/** Closed shared arena; capacity bounds only migrated graph-metadata storage. */
typedef struct mlx_submission_graph_quota_ { void* ctx; } mlx_submission_graph_quota;
typedef struct mlx_submission_graph_quota_layout_ {
  size_t capacity;
  size_t allocation_bytes;
  size_t alignment;
  size_t retirement_controls;
} mlx_submission_graph_quota_layout;
/** Cold, no allocation. Zero succeeds, nonzero rejects the exact layout. */
int mlx_submission_graph_quota_layout_for(
    mlx_submission_graph_quota_layout* out, size_t capacity);
/** Pure all-attempt arithmetic; zero succeeds. No arena or runtime is touched. */
int mlx_submission_graph_quota_population_extent(
    size_t* out, size_t requested_bytes, size_t nonempty_attempts);
int mlx_submission_graph_quota_fresh_capacity(size_t* out, size_t extents);
/** Failure consumes neither owner nor callback; success owns both. */
int mlx_submission_graph_quota_new_retaining(
    mlx_submission_graph_quota* out, size_t capacity,
    void* owner, void (*retire)(void*));
void mlx_submission_graph_quota_retain(mlx_submission_graph_quota quota);
void mlx_submission_graph_quota_release(mlx_submission_graph_quota quota);
size_t mlx_submission_graph_quota_occupied(mlx_submission_graph_quota quota);
/** Constructor-only arena, inherited by nested ordinary scopes. */
int mlx_submission_scope_new_retaining_with_quota(
    mlx_submission_scope* out, void* owner, void (*retire)(void*),
    mlx_submission_record_quota quota);
/** Both original arenas are constructor-owned; nested scopes inherit each. */
int mlx_submission_scope_new_retaining_with_arenas(
    mlx_submission_scope* out, void* owner, void (*retire)(void*),
    mlx_submission_record_quota records, mlx_submission_graph_quota graph);
/** Separate paid arenas, with exact current original parent and durable ancestry.
 * No active Record/worker is allowed; failure consumes no owner or arena.
 * Status 2: foreign Record arena; 3: foreign Graph arena; 4: invalid parent.
 */
int mlx_submission_scope_new_original_child(
    mlx_submission_scope* out, void* owner, void (*retire)(void*),
    mlx_submission_record_quota records, mlx_submission_graph_quota graph,
    mlx_submission_observer parent);
size_t mlx_submission_scope_original_child_control_bytes(void);
int mlx_submission_scope_seal(mlx_submission_scope scope);
int mlx_submission_scope_query(
    mlx_submission_status* out, mlx_submission_scope scope);
// Exact record-only cold observation for this scope and its durable descendants.
// An entered but unfinished Record is pending even before it accepts native work.
// Live empty children do not count. Failure/blocked flags stay sticky. This is
// not scope settlement, graph retirement, a write exclusion or admission grant.
typedef struct mlx_submission_record_status_ {
  bool pending;
  bool failed;
  bool blocked;
} mlx_submission_record_status;
// Owner-thread only, allocation/lock/progress-free. Invalid arguments return 1
// without writing the output or using the textual native error channel.
int mlx_submission_scope_query_records(
    mlx_submission_record_status* out, mlx_submission_scope scope);
// Actual inline native Scope allocation size; no registry/graph/allocator cost.
size_t mlx_submission_scope_control_bytes(void);
// Exact inline extraction controls used after final release, not a heap node.
size_t mlx_submission_scope_retirement_control_bytes(void);

// Advances lifetime evidence without reclaiming native graph/resource owners.
// Reclamation is deferred to an ordinary native operation on the owner thread.
int mlx_submission_scope_progress(
    mlx_submission_status* out, mlx_submission_scope scope);
/* Fixed observation-only transport. 0=observed, 1=needs funded progress,
 * 2=unobservable, 3=busy. None is terminal evidence by itself. */
unsigned mlx_submission_scope_enable_scoped(mlx_submission_scope scope);
/* Original configure-once mode. Fixed NativeControlFailure status; no error
 * handler, allocation, scope replacement or ordinary fallback on refusal. */
unsigned mlx_submission_scope_require_original_controls(mlx_submission_scope scope);
unsigned mlx_submission_scope_enable_original_controls(mlx_submission_scope scope);
/* Per-object control facts only. Event count, concrete F payload and combined
 * arena producer fit require their own source geometry; zero does not prove it. */
typedef struct mlx_submission_native_control_layout_ {
  size_t event_object_bytes;
  size_t event_shared_bytes;
  size_t event_shared_alignment;
  size_t event_graph_extent;
  size_t event_platform_objects;
  size_t task_header_bytes;
  size_t task_header_alignment;
  size_t fixed_controls;
} mlx_submission_native_control_layout;
unsigned mlx_submission_native_control_layout_for(mlx_submission_native_control_layout* out);
/* Pure fixed owner query, separate from selected-runtime preparation below.
 * thread_bytes covers submission TLS only, not all safe/worker TLS. */
typedef struct mlx_submission_static_layout_ {
  size_t module_bytes;
  size_t thread_bytes;
  size_t dynamic_registry_bytes;
  bool constant_registry;
} mlx_submission_static_layout;
void mlx_submission_static_layout_for(mlx_submission_static_layout* out);

/* Ordinary selected runtime initialization, never a request grant. Fixed
 * headers/counts are separate from unpriced table/string/thread/device payloads.
 * 0=success, 1=ordinary native error, 2=active original scope, 3=invalid pointers. */
typedef struct mlx_submission_runtime_baseline_ {
  size_t selected_streams;
  size_t cpu_workers;
  size_t scheduler_object_bytes;
  size_t worker_object_bytes;
  size_t event_runtime_object_bytes;
  size_t native_threads;
  size_t unpriced_populations;
} mlx_submission_runtime_baseline;
/* TLS-only no-hooks precheck; the constructor repeats it after ordinary entry. */
bool mlx_submission_runtime_preparation_allowed(void);
unsigned mlx_submission_prepare_runtime(
    mlx_submission_runtime_baseline* out, mlx_stream operation, mlx_stream weights);


unsigned mlx_submission_scope_progress_scoped(
    mlx_submission_status* out, mlx_submission_scope scope);
/* Identity has authority only alongside the actual borrowed live scope. */

int mlx_submission_scope_free(mlx_submission_scope scope);

// One ordinary owner-thread retirement attempt, never completion evidence.
typedef enum mlx_submission_retirement_ {
  MLX_SUBMISSION_RETIREMENT_COMPLETE_SNAPSHOT = 0,
  MLX_SUBMISSION_RETIREMENT_BUSY = 1,
} mlx_submission_retirement;

// Does not progress, evaluate, wait for work, or retry a contended registry.
// COMPLETE_SNAPSHOT leaves unresolved and foreign-owner records retained.
// Native resource destructors run outside the registry lock. Their cost is
// not bounded; this is not a scope-progress operation or a freed-byte report.
// Returns 0 with the result, or 1 through the existing native error channel.
// Failure leaves a non-null output untouched.
int mlx_submission_retire_completed(mlx_submission_retirement* out);

// One physical request-owned row per certified default-library lookup attempt.
// Owner callback is consumed only by successful creation and runs after storage.
typedef struct mlx_pipeline_cache_ { void* ctx; } mlx_pipeline_cache;
typedef struct mlx_pipeline_cache_layout_ {
  size_t allocation_bytes, alignment, control_bytes;
} mlx_pipeline_cache_layout;
unsigned mlx_pipeline_cache_layout_for(mlx_pipeline_cache_layout*, size_t attempts);
unsigned mlx_pipeline_cache_new_retaining(mlx_pipeline_cache*, size_t attempts,
    void* owner, void (*retire)(void*));
unsigned mlx_pipeline_cache_install(mlx_pipeline_cache, mlx_submission_graph_quota);
void mlx_pipeline_cache_free(mlx_pipeline_cache);

#ifdef __cplusplus
}
#endif
#endif

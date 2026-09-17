/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_DISTRIBUTED_GROUP_H
#define MLX_DISTRIBUTED_GROUP_H

#include <stdbool.h>
#include <stddef.h>

#include "mlx/c/stream.h"
#include "mlx/c/array.h"
#include "mlx/c/submission.h"
#include "mlx/c/event.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_distributed_group MLX distributed
 */
/**@{*/

/**
 * A MLX distributed group object.
 */
typedef struct mlx_distributed_group_ {
  void* ctx;
} mlx_distributed_group;

/**
 * Create an empty group.
 */
mlx_distributed_group mlx_distributed_group_new(void);

/**
 * Free the group.
 */
int mlx_distributed_group_free(mlx_distributed_group group);

/**
 * Initialize distributed.
 */
int mlx_distributed_init(
    mlx_distributed_group* res,
    bool strict,
    const char* bk /* may be null */);

/** Mark a live canonical communicator terminal for future submission. This is
 * monotonic and does not allocate, lock, set an error, wait, cancel or complete.
 * Null is ignored. The caller retains the immutable live group wrapper.
 */
void mlx_distributed_group_mark_terminal_submission(mlx_distributed_group group);
/** Atomic-only terminal query; null is conservatively terminal. */
bool mlx_distributed_group_terminal_submission(mlx_distributed_group group);

/** Partial immutable storage of the actual native implementation. All byte
 * fields exclude unresolved domains. This is not a complete allocation bound.
 */
typedef struct mlx_distributed_storage_inventory_ {
  unsigned kind, unresolved;
  size_t wrapper_bytes, implementation_bytes;
  size_t socket_handles, socket_capacity_bytes, buffer_bytes;
  size_t pool_workers, pool_worker_capacity_bytes;
  size_t socket_workers, socket_map_buckets;
} mlx_distributed_storage_inventory;
/** Fixed read-only query, no runtime/error channel. Null returns false. */
bool mlx_distributed_group_storage_inventory(
    mlx_distributed_storage_inventory* out, mlx_distributed_group group);
/** Exact implementation equality across distinct wrappers; null is false. */
bool mlx_distributed_group_same_implementation(
    mlx_distributed_group left, mlx_distributed_group right);
/** Fixed host frame census for these allocation-free queries. */
size_t mlx_distributed_group_storage_inventory_controls(void);

/** Partial exact worker destinations. Other operation/communicator owners
 * remain outside this query; it never grants a Scope or total allocation fit.
 */
typedef struct mlx_distributed_worker_storage_ {
  size_t pool_jobs, socket_attempts, destination_arrays;
  size_t task_graph_extent, destination_graph_extent, controls;
} mlx_distributed_worker_storage;
/** operation: 0=sum, 1=max, 2=min, 3=gather, 4=send, 5=recv.
 * Reads only actual retained group/input metadata, with no runtime/error path.
 * False leaves output unchanged (including unsupported backend or ABI).
 */
bool mlx_distributed_group_worker_storage(mlx_distributed_worker_storage* out,
    mlx_distributed_group group, mlx_array input, unsigned operation, int peer);
size_t mlx_distributed_group_worker_storage_controls(mlx_distributed_group group);

/** Actual Ring outer CPU TaskNode; not a whole-operation allocation bound. */
typedef struct mlx_distributed_dispatch_storage_ {
  size_t task_bytes, task_alignment, graph_extent, controls;
} mlx_distributed_dispatch_storage;
/** Same operation/source inputs as the worker query; output changes only on success. */
bool mlx_distributed_group_dispatch_storage(mlx_distributed_dispatch_storage* out,
    mlx_distributed_group group, mlx_array input, unsigned operation, int peer);
size_t mlx_distributed_group_dispatch_storage_controls(mlx_distributed_group group);

/** Source-derived lazy constructor only. These classes exclude evaluation,
 * data/backing, communication tasks, events and persistent group ownership.
 */
typedef struct mlx_distributed_constructor_storage_ {
  size_t output_rank, output_elements, primitives, input_edges;
  size_t blocks, header_bytes, header_alignment, slots_bytes, slots_alignment;
  size_t reserved_alignment, requested_bytes, allocation_extents, named_control_bytes;
  size_t request_bytes[10], request_alignments[10], request_counts[10];
} mlx_distributed_constructor_storage;
bool mlx_distributed_group_constructor_storage(mlx_distributed_constructor_storage* out,
    mlx_distributed_group group, mlx_array input, unsigned operation, int peer);
size_t mlx_distributed_group_constructor_storage_controls(void);
/** Recompute and reserve the exact constructor, invoke the same ordinary ops,
 * publish one fresh original handle, then retire unused slots before return.
 * This does not evaluate, submit communication, or establish its total fit.
 */
unsigned mlx_distributed_construct_original(mlx_array* out, mlx_submission_observer observer,
    mlx_distributed_group group, mlx_array input, unsigned operation, int peer, mlx_stream stream);

/** The exact native primitive and its retained actual Ring source. Host bank,
 * asynchronous worker and logical backing populations are deliberately separate.
 * This is not a physical-buffer budget or a persistent communicator fit.
 */
typedef struct mlx_distributed_cpu_eval_storage_ {
  unsigned operation;
  int peer;
  size_t input_rank, output_rank, inputs;
  bool tracer, possible_copy;
  size_t backing_births, data_captures, temporary_batches;
  size_t logical_backing_bytes, copy_backing_bytes, output_backing_bytes;
  size_t copy_worker_graph_extent, communication_worker_graph_extent;
  mlx_distributed_worker_storage communication;
  size_t blocks, header_bytes, header_alignment, slots_bytes, slots_alignment;
  size_t reserved_alignment, requested_bytes, allocation_extents, named_control_bytes;
  size_t request_bytes[10], request_alignments[10], request_counts[10];
} mlx_distributed_cpu_eval_storage;
/** Read-only exact dynamic primitive/source query, no initialization or error
 * channel. False leaves out unchanged. Accepted graphs do not recheck fencing.
 */
bool mlx_distributed_query_cpu_eval_storage(mlx_distributed_cpu_eval_storage* out, mlx_array output);
/** Actual source query frames payable before the query. SIZE_MAX on overflow. */
size_t mlx_distributed_cpu_eval_storage_controls(mlx_array output);
/** Same Eval worker census before lazy graph construction. Actual retained
 * Group/input and ordinary constructor geometry are required; no proxy array.
 */
bool mlx_distributed_query_cpu_source_storage(mlx_distributed_cpu_eval_storage* out,
    mlx_distributed_group group,mlx_array input,unsigned operation,int peer);
size_t mlx_distributed_cpu_source_storage_controls(mlx_distributed_group group);
/** Layout-only same-worker quotation from a real retained native Group. No
 * array/primitive is constructed. Both outputs stay unchanged on refusal;
 * actual input binding and original role admission remain separate. */
bool mlx_distributed_query_cpu_layout_storage(mlx_distributed_cpu_eval_storage* out,
    mlx_distributed_constructor_storage* constructor_out,mlx_distributed_group group,
    const int* shape,size_t rank,mlx_dtype dtype,unsigned operation,int peer);
size_t mlx_distributed_cpu_layout_storage_controls(mlx_distributed_group group);


/** Actual settled-leaf Ring operation plus its single-CPU-stream completion.
 * Includes constructor, Eval, worker, exact root and Synchronizer/Event owners.
 * Runtime startup, physical payloads and host source/custody remain separate. */
typedef struct mlx_distributed_cpu_completion_storage_ {
  mlx_operation_eval_traversal_layout traversal;
  size_t graph_allocation_extents, graph_capacity;
  size_t record_allocation_extents, record_capacity;
  size_t synchronizer_graph_extent, signal_graph_extent;
  size_t platform_events, named_control_bytes;
} mlx_distributed_cpu_completion_storage;
bool mlx_distributed_query_cpu_completion_storage(mlx_distributed_cpu_completion_storage* out,
    mlx_distributed_group group, mlx_array input, unsigned operation, int peer);
size_t mlx_distributed_cpu_completion_storage_controls(mlx_distributed_group group);
/** Same single-operation completion, sourced from an immutable selected layout.
 * No Array, native task, storage grant or completion claim is constructed. */
bool mlx_distributed_query_cpu_completion_layout_storage(mlx_distributed_cpu_completion_storage* out,
    mlx_distributed_group group,const int* shape,size_t rank,mlx_dtype dtype,unsigned operation,int peer);
size_t mlx_distributed_cpu_completion_layout_storage_controls(mlx_distributed_group group);
/** Same-stream send/receive pair from one actual settled input. Both native
 * edges execute before the one shared completion; no per-edge wait is inserted. */
bool mlx_distributed_query_cpu_exchange_storage(mlx_distributed_cpu_completion_storage* out,
    mlx_distributed_group group,mlx_array input,int destination,int source);
size_t mlx_distributed_cpu_exchange_storage_controls(mlx_distributed_group group);
/** Exact same paired completion, sourced from an immutable logical input layout
 * and the actual Group. No proxy Array or native execution is created. */
bool mlx_distributed_query_cpu_exchange_layout_storage(mlx_distributed_cpu_completion_storage* out,
    mlx_distributed_group group,const int* shape,size_t rank,mlx_dtype dtype,int destination,int source);
size_t mlx_distributed_cpu_exchange_layout_storage_controls(mlx_distributed_group group);
/** Same pair and shared completion with independently authenticated send and
 * receive prototypes. The receive prototype is not a primitive input edge. */
bool mlx_distributed_query_cpu_exchange_sources_storage(mlx_distributed_cpu_completion_storage* out,
    mlx_distributed_group group,mlx_array input,mlx_array receive_like,int destination,int source);
bool mlx_distributed_query_cpu_exchange_layouts_storage(mlx_distributed_cpu_completion_storage* out,
    mlx_distributed_group group,const int* send_shape,size_t send_rank,
    const int* receive_shape,size_t receive_rank,mlx_dtype dtype,int destination,int source);





/** Actual retained native owner requests. Shared owner bytes include the
 * implementation. Missing managed/inherited sources remain unresolved; OS-private
 * resources follow the existing native runtime boundary.
 */
typedef struct mlx_distributed_persistent_storage_ {
  uint64_t storage_identity;
  unsigned kind, unresolved;
  size_t wrapper_bytes, shared_owner_bytes, shared_owner_alignment;
  size_t socket_vector_bytes, communication_buffer_bytes, pool_vector_bytes;
  size_t socket_nodes, socket_node_bytes, socket_bucket_bytes;
  size_t thread_count, thread_cpp_runtime_bytes, retained_cpp_bytes, controls;
} mlx_distributed_persistent_storage;
bool mlx_distributed_group_persistent_storage(mlx_distributed_persistent_storage* out,mlx_distributed_group);
size_t mlx_distributed_group_persistent_storage_controls(mlx_distributed_group);

/**
 * Get the rank.
 */
int mlx_distributed_group_rank(mlx_distributed_group group);

/**
 * Get the group size.
 */
int mlx_distributed_group_size(mlx_distributed_group group);

/** Get the native communicator's selected transport stream. */
int mlx_distributed_group_communication_stream(
    mlx_stream* res,
    mlx_distributed_group group);

/**
 * Split the group.
 */
int mlx_distributed_group_split(
    mlx_distributed_group* res,
    mlx_distributed_group group,
    int color,
    int key);

/**
 * Check if distributed is available.
 */
bool mlx_distributed_is_available(const char* bk /* may be null */);

/**@}*/

/** Exact borrowed matrix source for the existing variable Ring worker. */
size_t mlx_distributed_variable_storage_controls(mlx_distributed_group group);
bool mlx_distributed_query_variable_layout_storage(mlx_distributed_cpu_eval_storage* out,
    mlx_distributed_constructor_storage* constructor_out, mlx_distributed_group group,
    const int* shape, size_t rank, mlx_dtype dtype, const size_t* matrix,
    size_t matrix_entries, bool transposed);
unsigned mlx_distributed_construct_variable_original(mlx_array* out, mlx_submission_observer observer,
    mlx_distributed_group group, mlx_array input, const size_t* matrix,
    size_t matrix_entries, bool transposed, mlx_stream stream);

size_t mlx_distributed_variable_completion_storage_controls(mlx_distributed_group group);
bool mlx_distributed_query_variable_completion_layout_storage(
    mlx_distributed_cpu_completion_storage* out, mlx_distributed_group group,
    const int* shape, size_t rank, mlx_dtype dtype, const size_t* matrix,
    size_t entries, bool transposed);
bool mlx_distributed_query_variable_completion_storage(
    mlx_distributed_cpu_completion_storage* out, mlx_distributed_group group,
    mlx_array input, const size_t* matrix, size_t entries, bool transposed);

// Descriptive global send/receive row ceilings. No count matrix or constructor
// permission is created; actual execution still requires the complete matrix.
size_t mlx_distributed_variable_envelope_storage_controls(mlx_distributed_group group);
bool mlx_distributed_query_variable_envelope_layout_storage(
    mlx_distributed_cpu_eval_storage* out,mlx_distributed_constructor_storage* constructor,
    mlx_distributed_group group,const int* maximum_send_shape,size_t rank,
    mlx_dtype dtype,size_t maximum_receive_rows);
bool mlx_distributed_query_variable_envelope_storage(
    mlx_distributed_cpu_eval_storage* out,mlx_distributed_constructor_storage* constructor,
    mlx_distributed_cpu_completion_storage* completion,mlx_distributed_group group,
    const int* maximum_send_shape,size_t rank,mlx_dtype dtype,size_t maximum_receive_rows);

#ifdef __cplusplus
}
#endif

#endif

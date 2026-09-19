/* Copyright © 2026 Apple Inc. */

#ifndef MLX_EVENT_H
#define MLX_EVENT_H
#include "mlx/c/array.h"
#include "mlx/c/submission.h"
#include "mlx/c/prefill_failure.h"

#include <stdbool.h>

#include "mlx/c/device.h"
#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_event Completion event
 * Backend-independent, single-shot completion events.
 */
/**@{*/

/** An owning opaque completion event. */
typedef struct mlx_event_ {
  void* ctx;
} mlx_event;

// Selected original operation companion. All statuses are ScopedEvaluation;
// no textual error channel or replacement failure owner is used.
typedef struct mlx_operation_event_ { void* ctx; } mlx_operation_event;
unsigned mlx_operation_event_new(mlx_operation_event*, mlx_submission_observer);
// Qualified host Add/Multiply constructor envelope. Physical reservation uses
// the existing current-role Graph arena. No Eval/worker/whole-fit acceptance.
typedef struct mlx_pointwise_graph_layout_ {
  size_t operations;
  size_t maximum_rank;
  size_t blocks;
  size_t header_bytes;
  size_t header_alignment;
  size_t slots_bytes;
  size_t slots_alignment;
  size_t reserved_alignment;
  size_t requested_bytes;
  size_t allocation_extents;
  size_t named_control_bytes;
  size_t request_bytes[10];
  size_t request_alignments[10];
  size_t request_counts[10];
} mlx_pointwise_graph_layout;
bool mlx_operation_event_pointwise_graph_layout(mlx_pointwise_graph_layout*, size_t operations, size_t maximum_rank);
// Output remains null on refusal. Success owns only the header/slots/unused
// blocks; consumed blocks belong to their actual Graph allocator owners.
unsigned mlx_operation_event_prepare_pointwise_graph(void** out, mlx_submission_observer, size_t operations, size_t maximum_rank);
void mlx_operation_event_finish_pointwise_graph(void*);

// Selected fixed GPU Eval host prologue. Pure query only: no public API can
// install this bank. The actual fixed Record/current array selects construction.
typedef struct mlx_gpu_eval_prologue_layout_ {
  size_t inputs, siblings, tracer_inputs;
  size_t request_bytes[5], request_alignments[5];
  size_t header_bytes, header_alignment, slots_bytes, slots_alignment;
  size_t blocks, reserved_alignment, requested_bytes, allocation_extents;
  size_t data_slot_bytes, data_slot_alignment, array_slot_bytes, array_slot_alignment;
  size_t named_control_bytes;
} mlx_gpu_eval_prologue_layout;
bool mlx_operation_event_gpu_eval_prologue_layout(
    mlx_gpu_eval_prologue_layout*, size_t inputs, size_t siblings);
// Descriptive tracing variant; actual bank creation reads the genuine array.
bool mlx_operation_event_gpu_eval_prologue_layout_with_tracing(
    mlx_gpu_eval_prologue_layout*, size_t inputs, size_t siblings, bool tracer);

// Exact CPU Eval cleanup node and empty Data-reference backing. No public
// caller can install the bank: the genuine Fixed Eval/current array owns it.
typedef struct mlx_cpu_eval_cleanup_layout_ {
  size_t inputs, siblings;
  size_t request_bytes[2], request_alignments[2];
  size_t header_bytes, header_alignment, slots_bytes, slots_alignment;
  size_t blocks, reserved_alignment, requested_bytes, allocation_extents;
  size_t data_slot_bytes, data_slot_alignment, named_control_bytes;
} mlx_cpu_eval_cleanup_layout;
// Pure query; invalid/overflow input leaves the whole output unchanged.
bool mlx_operation_event_cpu_eval_cleanup_layout(
    mlx_cpu_eval_cleanup_layout*, size_t inputs, size_t siblings);

// Exact rank-two/last-axis CPU ArgPartition Eval producer. The physical bank
// is installed only by the authenticated native Eval; this is a pure query.
typedef struct mlx_cpu_argpartition_layout_ {
  size_t request_bytes[8], request_alignments[8], request_counts[8];
  size_t header_bytes, header_alignment, slots_bytes, slots_alignment;
  size_t blocks, reserved_alignment, requested_bytes, allocation_extents;
  size_t named_control_bytes;
} mlx_cpu_argpartition_layout;
bool mlx_operation_event_cpu_argpartition_layout(mlx_cpu_argpartition_layout*, bool tracer);
bool mlx_operation_event_cpu_argpartition_source_layout(mlx_cpu_argpartition_layout*, size_t rank, size_t elements, bool tracer);

// Existing CPU Contiguous or private completion Eval: scalar source facts only.
// The actual native Eval authenticates the primitive before installing storage.
typedef struct mlx_cpu_copy_eval_layout_ {
  size_t graph_extents, worker_graph_extents, backing_births, named_control_bytes;
  size_t signal_graph_extents;
} mlx_cpu_copy_eval_layout;
bool mlx_operation_event_cpu_host_transfer_eval_layout(mlx_cpu_copy_eval_layout*,
    mlx_dtype,size_t rank,bool store,bool tracer);
// Three-output affine converter Eval, with optional input compaction.
bool mlx_operation_event_cpu_affine_quantize_eval_layout(mlx_cpu_copy_eval_layout*,
    mlx_dtype,size_t rank,size_t rows,size_t columns,int group_size,int bits,bool copy,bool tracer);
// Typed F16/BF16/F32 ArgMin/ArgMax source; numerical task and U32 output are shared.
bool mlx_operation_event_cpu_typed_arg_reduce_eval_layout(mlx_cpu_copy_eval_layout*,
    mlx_dtype,size_t rank,size_t columns,size_t rows,bool tracer);
bool mlx_operation_event_cpu_greedy_eval_layout(mlx_cpu_copy_eval_layout*,
    size_t rank,size_t columns,size_t rows,bool reduction,bool tracer);
bool mlx_operation_event_cpu_reshape_alias_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank, size_t output_rank, bool tracer);
bool mlx_operation_event_cpu_reshape_copy_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank, size_t output_rank, bool tracer);
bool mlx_operation_event_cpu_reshape_eval_layout(mlx_cpu_copy_eval_layout* out,
    const int* source_shape,const int64_t* source_strides,size_t rank,
    const int* output_shape,size_t output_rank,bool tracer);
bool mlx_operation_event_cpu_sdpa_array_mask_control_bytes(size_t* out,
    size_t rank,size_t queries,size_t keys,size_t values,size_t scores);
bool mlx_operation_event_cpu_sdpa_fallback_control_bytes(size_t* out,
    size_t rank,size_t queries,size_t keys,size_t values,size_t scores);
bool mlx_operation_event_cpu_arange_float_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t elements,bool tracer);
bool mlx_operation_event_cpu_arange_int_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype dtype,size_t elements,bool tracer);
bool mlx_operation_event_cpu_tiled_gather_mm_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t lhs_rank,size_t rhs_rank,size_t index_rank,size_t m,size_t n,size_t k,
    size_t batches,bool tracer);
bool mlx_operation_event_cpu_rope_fallback_control_bytes(size_t* out,
    size_t rank,size_t dimensions,size_t elements);
bool mlx_operation_event_cpu_concatenate_many_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype dtype,size_t rank,size_t inputs,size_t elements,bool tracer);
bool mlx_operation_event_cpu_concatenate_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype dtype,size_t rank,size_t left_elements,size_t right_elements,bool tracer);
bool mlx_operation_event_cpu_rms_fallback_control_bytes(
    size_t* out, mlx_dtype dtype, size_t rank, size_t width, size_t rows);
bool mlx_operation_event_cpu_select_broadcast_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t elements,bool tracer);
bool mlx_operation_event_cpu_typed_select_broadcast_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype dtype,size_t rank,size_t elements,bool tracer);
bool mlx_operation_event_cpu_selection_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype dtype,size_t rank,size_t elements,bool full,bool tracer);
bool mlx_operation_event_cpu_row_full_eval_layout(mlx_cpu_copy_eval_layout*,
    size_t rank,size_t width,size_t rows,bool tracer);
bool mlx_operation_event_cpu_reduction_eval_layout(mlx_cpu_copy_eval_layout* out,
    unsigned operation,size_t rank,size_t width,size_t rows,bool tracer);
bool mlx_operation_event_cpu_scatter_eval_layout(mlx_cpu_copy_eval_layout*,
    mlx_dtype source,mlx_dtype index,size_t rank,size_t output_elements,size_t update_elements,bool tracer);
bool mlx_operation_event_cpu_scatter_axis_eval_layout(mlx_cpu_copy_eval_layout*,
    mlx_dtype index,size_t rank,size_t output_elements,size_t update_elements,bool tracer);
bool mlx_operation_event_cpu_scatter_add_rows_eval_layout(mlx_cpu_copy_eval_layout*,
    mlx_dtype,size_t,size_t,bool);
bool mlx_operation_event_cpu_gather_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype source, mlx_dtype index, size_t source_rank, size_t index_rank,
    size_t source_elements, size_t index_elements, size_t slice_elements, bool tracer);
bool mlx_operation_event_cpu_alias_eval_layout(mlx_cpu_copy_eval_layout* out,
    unsigned operation, size_t rank, size_t output_rank, bool tracer);
bool mlx_operation_event_cpu_tiled_matmul_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank, size_t m, size_t n, size_t k, size_t batches, bool tracer);
bool mlx_operation_event_cpu_tiled_matmul_copy_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank, size_t m, size_t n, size_t k, size_t batches, size_t copies, bool tracer);
bool mlx_operation_event_cpu_bf16_matmul_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank, size_t m, size_t n, size_t k, size_t batches, bool tracer);
bool mlx_operation_event_cpu_bf16_matmul_copy_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t m,size_t n,size_t k,size_t batches,size_t copies,bool tracer);
bool mlx_operation_event_cpu_f16_matmul_copy_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t m,size_t n,size_t k,size_t batches,size_t copies,bool tracer);
bool mlx_operation_event_cpu_random_bits_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank, size_t elements, bool tracer);
bool mlx_operation_event_cpu_softmax_eval_layout(mlx_cpu_copy_eval_layout*,
    size_t rank,size_t columns,size_t rows,bool tracer);
bool mlx_operation_event_cpu_typed_softmax_eval_layout(mlx_cpu_copy_eval_layout*,
    mlx_dtype dtype,bool precise,size_t rank,size_t columns,size_t rows,bool tracer);
bool mlx_operation_event_cpu_scalar_update_eval_layout(mlx_cpu_copy_eval_layout*,size_t elements,bool tracer);
bool mlx_operation_event_cpu_static_update_eval_layout(mlx_cpu_copy_eval_layout*,size_t rank,size_t elements,size_t update_elements,bool tracer);
bool mlx_operation_event_cpu_argsort_eval_layout(mlx_cpu_copy_eval_layout*,mlx_dtype source,size_t rank,size_t columns,size_t rows,bool tracer);
bool mlx_operation_event_cpu_gather_axis_row_eval_layout(mlx_cpu_copy_eval_layout*,size_t rank,size_t elements,bool tracer);
bool mlx_operation_event_cpu_partition_row_eval_layout(mlx_cpu_copy_eval_layout*,size_t rank,size_t elements,bool tracer);
bool mlx_operation_event_cpu_scan_sum_row_eval_layout(mlx_cpu_copy_eval_layout*,size_t rank,size_t elements,bool tracer);
bool mlx_operation_event_cpu_maximum_row_eval_layout(mlx_cpu_copy_eval_layout*,size_t rank,size_t elements,bool tracer);

bool mlx_operation_event_cpu_slice_eval_layout(mlx_cpu_copy_eval_layout*,size_t rank,bool empty,bool tracer);
bool mlx_operation_event_cpu_cast_eval_layout(mlx_cpu_copy_eval_layout*,
    mlx_dtype source, mlx_dtype destination, size_t rank, size_t elements, bool tracer);
bool mlx_operation_event_cpu_byte_view_eval_layout(mlx_cpu_copy_eval_layout*,
    mlx_dtype source, mlx_dtype destination, size_t rank, size_t bytes, bool copy, bool tracer);
bool mlx_operation_event_cpu_copy_eval_layout(mlx_cpu_copy_eval_layout*,
    size_t rank, size_t inputs, bool copy, bool tracer);

// Closed existing unary worker selector, matched again against the real primitive.
typedef struct mlx_cpu_unary_eval_layout_ {
  size_t graph_extents, worker_graph_extents, backing_births, named_control_bytes;
} mlx_cpu_unary_eval_layout;
bool mlx_operation_event_cpu_unary_eval_layout(mlx_cpu_unary_eval_layout*,
    uint32_t operation, mlx_dtype dtype, size_t rank, bool tracer);

// Actual two-input binary/comparison task, with checked native int loop extents.
typedef struct mlx_cpu_binary_eval_layout_ {
  size_t graph_extents, worker_graph_extents, backing_births, named_control_bytes;
} mlx_cpu_binary_eval_layout;
bool mlx_operation_event_cpu_binary_eval_layout(mlx_cpu_binary_eval_layout*,
    uint32_t operation, mlx_dtype dtype, size_t rank, size_t elements, bool tracer);

// Pure requested storage facts, not an arena-fit or admission certificate.
// On invalid/overflow input returns false without changing *out.
typedef struct mlx_operation_root_storage_layout_ {
  size_t root_count;
  size_t object_bytes;
  size_t object_alignment;
  size_t object_graph_extent;
  size_t roots_bytes;
  size_t roots_alignment;
  size_t roots_graph_extent;
  size_t graph_blocks;
  size_t graph_request_extent;
} mlx_operation_root_storage_layout;
// Pure qualified fixed Record recipe. An eventless completion allocates none;
// host polling allocates none. Counts are supplied by an admitted caller bank.
// Unknown qualification/overflow returns false without writing out. This is not
// an arena-fit certificate and excludes Graph/backend wait dispatch owners.
typedef struct mlx_operation_wait_record_layout_ {
  size_t wait_count;
  size_t object_bytes;
  size_t object_alignment;
  size_t capture_slots;
  size_t capture_bytes;
  size_t capture_alignment;
  size_t stream_receipts;
  size_t stream_bytes;
  size_t stream_alignment;
  size_t allocations_per_wait;
  size_t requested_bytes_per_wait;
  size_t total_record_allocations;
  size_t total_record_requested_bytes;
  size_t named_control_bytes;
} mlx_operation_wait_record_layout;
bool mlx_operation_event_wait_record_layout(
    mlx_operation_wait_record_layout* out, size_t wait_count);

// Pure qualified selected-Eval reservation slice. Positive T/S/O describes
// actual tape/stream/output populations; this is not a cold admission proof.
// Unknown qualification, invalid shape or overflow leaves *out unchanged.
typedef struct mlx_operation_eval_record_layout_ {
  size_t tape_entries;
  size_t stream_count;
  size_t output_slots;
  size_t object_bytes;
  size_t object_alignment;
  size_t capture_slots;
  size_t capture_bytes;
  size_t capture_alignment;
  size_t stream_state_bytes;
  size_t stream_state_alignment;
  size_t stream_receipt_bytes;
  size_t stream_receipt_alignment;
  size_t primitive_owner_bytes;
  size_t primitive_owner_alignment;
  size_t output_pin_bytes;
  size_t output_pin_alignment;
  size_t record_allocations;
  size_t record_requested_bytes;
  size_t named_control_bytes;
  // Fixed Synchronizer/descriptor/Event Graph owner recipe, separate from Record.
  size_t host_graph_request_bytes[4];
  size_t host_graph_request_alignments[4];
  size_t host_graph_blocks;
  size_t host_graph_header_bytes;
  size_t host_graph_header_alignment;
  size_t host_graph_slots_bytes;
  size_t host_graph_slots_alignment;
  size_t host_graph_reserved_alignment;
  size_t host_graph_requested_bytes;
  size_t host_graph_allocation_extents;
  size_t host_graph_event_controls;
  size_t host_graph_platform_events;
} mlx_operation_eval_record_layout;
bool mlx_operation_event_eval_record_layout(
    mlx_operation_eval_record_layout* out,
    size_t tape_entries, size_t streams, size_t output_slots);

// Physically reserved host Graph storage for a closed resident lowering recipe.
// Counts never authorize a missing lowering or any ordinary host containers.
typedef struct mlx_resident_graph_layout_ {
  size_t primitives, seeds, maximum_rank, maximum_operands;
  size_t request_bytes[10], request_alignments[10], request_counts[10];
  size_t blocks, header_bytes, header_alignment, slots_bytes, slots_alignment;
  size_t reserved_alignment, requested_bytes, allocation_extents, named_control_bytes;
} mlx_resident_graph_layout;
bool mlx_operation_event_resident_graph_layout(mlx_resident_graph_layout*,
    size_t primitives, size_t seeds, size_t maximum_rank);
unsigned mlx_operation_event_prepare_resident_graph(void**, mlx_submission_observer,
    size_t primitives, size_t seeds, size_t maximum_rank);
bool mlx_operation_event_resident_graph_layout_with_operands(mlx_resident_graph_layout*,
    size_t primitives, size_t seeds, size_t maximum_rank, size_t maximum_operands);
unsigned mlx_operation_event_prepare_resident_graph_with_operands(void**, mlx_submission_observer,
    size_t primitives, size_t seeds, size_t maximum_rank, size_t maximum_operands);
// Additional immutable C handle shells, independently counted from primitives.
bool mlx_operation_event_resident_graph_layout_with_shells(mlx_resident_graph_layout*,
    size_t primitives, size_t seeds, size_t maximum_rank, size_t maximum_operands,
    size_t additional_shells);
unsigned mlx_operation_event_prepare_resident_graph_with_shells(void**, mlx_submission_observer,
    size_t primitives, size_t seeds, size_t maximum_rank, size_t maximum_operands,
    size_t additional_shells);
void mlx_operation_event_finish_resident_graph(void*);



// A finite prepared Record destination, including the selected Synchronizer.
// These counts confer no source authenticity, submission authority or Graph fit.
typedef struct mlx_operation_eval_traversal_limits_ {
  size_t root_count;
  size_t array_nodes;
  size_t tape_entries;
  size_t input_edges;
  size_t output_slots;
  size_t stream_count;
  size_t capture_slots;
} mlx_operation_eval_traversal_limits;

// Install a finite cold nested-Eval population before any host constructor.
unsigned mlx_operation_event_configure_nested_graph(
    void* owner, mlx_submission_observer observer,
    const mlx_operation_eval_traversal_limits* limits, size_t attempts);
unsigned mlx_operation_event_validate_nested_graph(
    mlx_submission_observer observer, size_t roots);
// Completes and retires only the actual current Scope's entered records. The
// outer Scope/custody remains live and its host bank resumes on every exit.
unsigned mlx_operation_event_complete_nested_graph(
    mlx_submission_observer observer, mlx_stream stream,
    const mlx_array* roots, size_t count);
size_t mlx_operation_event_nested_graph_control_bytes(void);
typedef struct mlx_operation_eval_traversal_layout_ {
  mlx_operation_eval_traversal_limits limits;
  size_t request_bytes[11];
  size_t request_alignments[11];
  size_t record_allocations;
  size_t record_requested_bytes;
  size_t named_control_bytes;
} mlx_operation_eval_traversal_layout;
// Qualification/invalid/overflow leaves out unchanged.
bool mlx_operation_event_eval_traversal_layout(
    mlx_operation_eval_traversal_layout* out,
    const mlx_operation_eval_traversal_limits* limits);
// Exact current original context; refuses active tracing/export/retained graphs.
unsigned mlx_operation_event_validate_traversal_context(mlx_submission_observer);
// Read only graph-membership checks plus the existing completed-event detach.
// Rejects lazy/traced/non-leaf input before any new graph work or allocation.
unsigned mlx_operation_event_validate_traversal_leaf(mlx_submission_observer, mlx_array);
// Reuses the same authenticated selected evaluator. All finite Record buffers
// precede traversal; capacity refusal never falls back to growing containers.
unsigned mlx_operation_event_submit_on_stream_prepared(
    mlx_operation_event, mlx_stream, const mlx_operation_eval_traversal_limits*);

bool mlx_operation_event_root_storage_layout(
    mlx_operation_root_storage_layout* out, size_t root_count);
// Validate owner, selected stream and count before native allocation. A reserve
// failure may return a NONNULL owning *out: caller must free/defer that wrapper
// even when status != 0. Never overwrite or refill an existing output.
unsigned mlx_operation_event_new_exact(
    mlx_operation_event* out, mlx_submission_observer observer,
    mlx_stream selected_stream, size_t root_count);
unsigned mlx_operation_event_append(mlx_operation_event, mlx_array);
unsigned mlx_operation_event_submit(mlx_operation_event);
// Explicit selected-stream frontier; empty/evaluated roots still submit it.
unsigned mlx_operation_event_submit_on_stream(mlx_operation_event, mlx_stream);
unsigned mlx_operation_event_query(mlx_operation_event);
unsigned mlx_operation_event_wait(mlx_operation_event);
unsigned mlx_operation_event_wait_stream(mlx_operation_event, mlx_stream);
// Guarded destruction, or allocation-free deferral when that guard is busy.
void mlx_operation_event_free(mlx_operation_event);
void mlx_operation_event_defer(mlx_operation_event);
unsigned mlx_operation_event_retire_deferred(mlx_submission_observer);
size_t mlx_operation_event_control_bytes(void);

/** Backend identity of a completion event. */
typedef enum mlx_event_backend_ {
  MLX_EVENT_BACKEND_NONE,
  MLX_EVENT_BACKEND_CPU,
  MLX_EVENT_BACKEND_METAL,
  MLX_EVENT_BACKEND_CUDA,
} mlx_event_backend;

/** Return a new, already-complete event without a device identity. */
mlx_event mlx_event_new(void);

/** Free an event handle. Queued producer and consumer work retains it. */
int mlx_event_free(mlx_event event);

/** Block the host until completion and report retained async errors. */
int mlx_event_synchronize(mlx_event event);

/**
 * Nonblocking completion query.
 *
 * On success, writes a monotonic completion value to `complete`. A completed
 * event with a retained asynchronous error returns failure instead.
 */
int mlx_event_query(bool* complete, mlx_event event);

/**
 * Order later work on `stream` after `event` without blocking the host.
 *
 * Producer and consumer devices must match. This orders backend work which is
 * subsequently submitted on the stream; constructing lazy MLX operations does
 * not itself submit them.
 */
int mlx_stream_wait_event(mlx_stream stream, mlx_event event);

/** Return whether this event has a producer device identity. */
int mlx_event_has_device(bool* has_device, mlx_event event);

/**
 * Return the producer device.
 *
 * Fails for identity-free events, such as events produced for empty or
 * already-available output sets.
 */
int mlx_event_get_device(mlx_device* device, mlx_event event);

/** Return the producer backend. */
int mlx_event_get_backend(mlx_event_backend* backend, mlx_event event);

/** Return whether this completion event carries timestamp markers. */
int mlx_event_has_timing(bool* has_timing, mlx_event event);

/**
 * Resolve elapsed execution-timeline seconds, blocking until completion.
 * Repeated calls return the same cached value and retained async errors.
 */
int mlx_event_elapsed(double* seconds, mlx_event event);

/**
 * Query elapsed execution-timeline seconds without blocking.
 *
 * Sets `ready` false while work is outstanding. When ready, `seconds` receives
 * the same stable value as `mlx_event_elapsed`.
 */
int mlx_event_try_elapsed(double* seconds, bool* ready, mlx_event event);

// One mixed GPU/CPU router frontier: exact owning alternatives, no arena grant.
typedef struct mlx_router_receipt_layout_ {
  size_t graph_requested_bytes, graph_allocation_extents;
  size_t platform_events, fast_backing_bytes, fast_backing_births;
  size_t named_control_bytes;
} mlx_router_receipt_layout;
bool mlx_operation_event_router_receipt_layout(mlx_router_receipt_layout*);

// Concrete single-GPU worker population; unknown outside its native profile.
typedef struct mlx_resident_gpu_worker_layout_ {
  size_t allocation_extents, named_control_bytes;
  size_t kernel_attempts, encoding_intervals, buffer_universe;
} mlx_resident_gpu_worker_layout;
bool mlx_operation_event_resident_gpu_worker_layout(mlx_resident_gpu_worker_layout*,
    size_t entries, size_t input_edges, size_t output_slots, size_t array_nodes,
    size_t backing_births, size_t maximum_rank, size_t maximum_operands);
bool mlx_operation_event_resident_gpu_worker_layout_with_sorts(mlx_resident_gpu_worker_layout*,
    size_t entries, size_t input_edges, size_t output_slots, size_t array_nodes,
    size_t backing_births, size_t maximum_rank, size_t maximum_operands,
    size_t additional_sort_kernels);
bool mlx_operation_event_resident_gpu_worker_layout_with_router(mlx_resident_gpu_worker_layout*,
    size_t entries, size_t input_edges, size_t output_slots, size_t array_nodes,
    size_t backing_births, size_t maximum_rank, size_t maximum_operands,
    size_t additional_sort_kernels, size_t cpu_partitions);
bool mlx_operation_event_resident_gpu_worker_layout_with_frontiers(mlx_resident_gpu_worker_layout*,
    size_t entries, size_t input_edges, size_t output_slots, size_t array_nodes,
    size_t backing_births, size_t maximum_rank, size_t maximum_operands,
    size_t additional_sort_kernels, size_t cpu_partitions, size_t evaluations,
    size_t consumer_waits);
bool mlx_operation_event_resident_grouped_sort_additional_kernels(size_t selections, size_t* out);

/** Submit one prepared nested single-GPU frontier; restore the same host bank. */
unsigned mlx_operation_event_new_nested_scheduled(mlx_operation_event* out,
    mlx_submission_observer observer, mlx_stream stream, size_t roots);
unsigned mlx_operation_event_append_nested_scheduled(mlx_operation_event value, mlx_array root);
unsigned mlx_operation_event_submit_nested_scheduled(mlx_operation_event value, mlx_stream stream);
void mlx_operation_event_abort_nested(mlx_operation_event value);
/** Same prepared nested worker over an exact caller-owned root list. */
unsigned mlx_operation_event_submit_nested_graph_roots(
    mlx_operation_event* out, mlx_submission_observer observer,
    mlx_stream stream, const mlx_array* roots, size_t count);
unsigned mlx_operation_event_submit_nested_graph(
    mlx_operation_event* out, mlx_submission_observer observer,
    mlx_stream stream, mlx_array root);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif

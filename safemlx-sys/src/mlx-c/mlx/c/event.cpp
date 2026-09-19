#include "mlx/backend/cpu/sampling_storage.h"
/* Copyright © 2026 Apple Inc. */

#include "mlx/c/error.h"
#include "mlx/c/event.h"
#include "mlx/c/private/mlx.h"
#include "mlx/c/private/operation_event.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include "mlx/transforms_impl.h"
#include "mlx/graph_construction.h"
#include "mlx/backend/cpu/copy_storage.h"
#include "mlx/fence.h"
#include "mlx/original_buffer.h"
#include "mlx/event.h"
#include <algorithm>
#include <limits>
#include <utility>
#include <thread>

namespace {
using mlx_c_detail::OperationEvent;
using mlx_c_detail::operation_get;
using mlx_c_detail::operation_current;
using mlx_c_detail::operation_failure;
mlx::core::CompletionEvalTraversalLimits native_traversal_limits(
    const mlx_operation_eval_traversal_limits& value) noexcept {
  return {value.root_count, value.array_nodes, value.tape_entries, value.input_edges,
      value.output_slots, value.stream_count, value.capture_slots};
}
// Unavailable runtime teardown keeps the already-paid wrapper, completion and
// exact Scope reference. It is drained only at that owner's guarded retirement.
void destroy_operation(OperationEvent* event) noexcept {
  auto* scope = event->scope;
  auto* quota = event->quota;
  event->~OperationEvent();
  quota->deallocate(event, sizeof(OperationEvent), alignof(OperationEvent));
  scope->release();
}
}

extern "C" unsigned mlx_operation_event_new(
    mlx_operation_event* out, mlx_submission_observer observer) {
  using namespace mlx::core;
  auto* scope = static_cast<submission::Scope*>(observer.ctx);
  if (!out || out->ctx || !scope || !scope->valid_original_observer() ||
      submission::current_scope() != scope || !scope->active_on_owner_thread()) return 4;
  const auto controls = submission::current_native_controls();
  if (!controls.original || controls.failure != submission::NativeControlFailure::none ||
      controls.scope_identity != scope->identity()) return 4;
  if (scope->failure_owner().get()->borrow()) return 7;
  if (scope->failure_owner().get()->pending_publication()) return 10;
  void* memory = controls.graph->try_allocate(sizeof(OperationEvent), alignof(OperationEvent));
  if (!memory) return 2;
  scope->retain_observer(); // same already-validated owner; no new authority
  out->ctx = ::new (memory) OperationEvent(scope, controls.graph);
  return 0;
}
extern "C" bool mlx_operation_event_eval_record_layout(
    mlx_operation_eval_record_layout* out,
    size_t tape_entries, size_t streams, size_t output_slots) {
  if (!out) return false;
  mlx::core::CompletionEvalRecordLayout native;
  if (!mlx::core::completion_eval_record_layout(tape_entries, streams, output_slots, native))
    return false;
  const mlx_operation_eval_record_layout value{
      native.tape_entries,
      native.stream_count,
      native.output_slots,
      native.object_bytes,
      native.object_alignment,
      native.capture_slots,
      native.capture_bytes,
      native.capture_alignment,
      native.stream_state_bytes,
      native.stream_state_alignment,
      native.stream_receipt_bytes,
      native.stream_receipt_alignment,
      native.primitive_owner_bytes,
      native.primitive_owner_alignment,
      native.output_pin_bytes,
      native.output_pin_alignment,
      native.record_allocations,
      native.record_requested_bytes,
      native.named_control_bytes + sizeof(native) + sizeof(mlx_operation_eval_record_layout) +
          sizeof(mlx_operation_eval_record_layout*) + 3 * sizeof(size_t),
      {native.host_graph.request_bytes[0], native.host_graph.request_bytes[1],
       native.host_graph.request_bytes[2], native.host_graph.request_bytes[3]},
      {native.host_graph.request_alignments[0], native.host_graph.request_alignments[1],
       native.host_graph.request_alignments[2], native.host_graph.request_alignments[3]},
      native.host_graph.blocks, native.host_graph.header_bytes, native.host_graph.header_alignment,
      native.host_graph.slots_bytes, native.host_graph.slots_alignment,
      native.host_graph.reserved_alignment, native.host_graph.requested_bytes,
      native.host_graph.allocation_extents, native.host_graph.event_construction_controls,
      native.host_graph.platform_events};
  *out = value;
  return true;
}

extern "C" unsigned mlx_operation_event_validate_traversal_context(
    mlx_submission_observer observer) {
  using namespace mlx::core;
  auto* scope = static_cast<submission::Scope*>(observer.ctx);
  if (!scope || !scope->valid_original_observer() ||
      submission::current_scope() != scope || !scope->active_on_owner_thread()) return 4;
  const auto controls = submission::current_native_controls();
  if (!controls.original || controls.failure != submission::NativeControlFailure::none ||
      controls.scope_identity != scope->identity()) return 4;
  if (scope->failure_owner().get()->borrow()) return 7;
  if (scope->failure_owner().get()->pending_publication()) return 10;
  if (scope->query().failed) return 9;
  // Dynamic broadcasts have a different input graph. Export/retained graphs
  // also fall outside this detached ordinary pointwise producer.
  if (detail::in_tracing() || detail::in_export_tracing() || detail::retain_graph()) return 4;
  return 0;
}
extern "C" bool mlx_operation_event_pointwise_graph_layout(
    mlx_pointwise_graph_layout* out, size_t operations, size_t maximum_rank) {
  if (!out) return false;
  mlx::core::submission::PointwiseGraphLayout native;
  if (!mlx::core::submission::pointwise_graph_layout(operations, maximum_rank, native)) return false;
  mlx_pointwise_graph_layout value{};
  value.operations = native.operations;
  value.maximum_rank = native.maximum_rank;
  value.blocks = native.blocks;
  value.header_bytes = native.header_bytes;
  value.header_alignment = native.header_alignment;
  value.slots_bytes = native.slots_bytes;
  value.slots_alignment = native.slots_alignment;
  value.reserved_alignment = native.reserved_alignment;
  value.requested_bytes = native.requested_bytes;
  value.allocation_extents = native.allocation_extents;
  value.named_control_bytes = native.named_control_bytes;
  std::copy(std::begin(native.request_bytes), std::end(native.request_bytes), value.request_bytes);
  std::copy(std::begin(native.request_alignments), std::end(native.request_alignments), value.request_alignments);
  std::copy(std::begin(native.request_counts), std::end(native.request_counts), value.request_counts);
  value.named_control_bytes += sizeof(native) + sizeof(value) + sizeof(out) +
      2 * sizeof(size_t) + sizeof(void*) + sizeof(mlx_submission_observer) + sizeof(unsigned);
  *out = value;
  return true;
}
extern "C" unsigned mlx_operation_event_prepare_pointwise_graph(
    void** out, mlx_submission_observer observer, size_t operations, size_t maximum_rank) {
  using namespace mlx::core::submission;
  if (!out || *out) return 4;
  if (auto status = mlx_operation_event_validate_traversal_context(observer)) return status;
  auto* scope = static_cast<Scope*>(observer.ctx);
  PointwiseGraphLayout layout;
  if (!pointwise_graph_layout(operations, maximum_rank, layout)) return 4;
  GraphConstruction* owner = nullptr;
  const auto failure = GraphConstruction::create(*scope->graph_quota(), scope->identity(), layout, owner);
  if (failure == GraphFailure::exhausted) return 2;
  if (failure == GraphFailure::construction_busy) return 10;
  if (static_cast<unsigned>(failure)) return 4;
  *out = owner;
  return 0;
}
extern "C" void mlx_operation_event_finish_pointwise_graph(void* owner) {
  if (owner) static_cast<mlx::core::submission::GraphConstruction*>(owner)->destroy();
}

extern "C" unsigned mlx_operation_event_validate_traversal_leaf(
    mlx_submission_observer observer, mlx_array input) {
  using namespace mlx::core;
  if (!input.ctx) return 4;
  if (auto status = mlx_operation_event_validate_traversal_context(observer)) return status;
  auto* scope = static_cast<submission::Scope*>(observer.ctx);
  const auto& value = mlx_array_get_(input);
  if (value.has_primitive() || !value.inputs().empty() || !value.siblings().empty() ||
      value.is_tracer() ||
      (value.status() != array::Status::available && !value.event().valid())) return 4;
  return static_cast<unsigned>(validate_scoped_array(value, *scope));
}

extern "C" bool mlx_operation_event_eval_traversal_layout(
    mlx_operation_eval_traversal_layout* out,
    const mlx_operation_eval_traversal_limits* limits) {
  if (!out || !limits) return false;
  mlx::core::CompletionEvalTraversalLayout native;
  if (!mlx::core::completion_eval_traversal_layout(native_traversal_limits(*limits), native))
    return false;
  mlx_operation_eval_traversal_layout value{};
  value.limits = *limits;
  std::copy(std::begin(native.request_bytes), std::end(native.request_bytes), value.request_bytes);
  std::copy(std::begin(native.request_alignments), std::end(native.request_alignments), value.request_alignments);
  value.record_allocations = native.record_allocations;
  value.record_requested_bytes = native.record_requested_bytes;
  value.named_control_bytes = native.named_control_bytes + sizeof(native) + sizeof(value) +
      sizeof(out) + sizeof(limits);
  *out = value;
  return true;
}

extern "C" bool mlx_operation_event_wait_record_layout(
    mlx_operation_wait_record_layout* out, size_t wait_count) {
  if (!out) return false;
  mlx::core::CompletionWaitRecordLayout per_wait;
  if (!mlx::core::completion_wait_record_layout(per_wait)) return false;
  const auto maximum = std::numeric_limits<size_t>::max();
  if ((wait_count && per_wait.record_allocations > maximum / wait_count) ||
      (wait_count && per_wait.record_requested_bytes > maximum / wait_count))
    return false;
  const mlx_operation_wait_record_layout value{wait_count,
      per_wait.object_bytes, per_wait.object_alignment, per_wait.capture_slots,
      per_wait.capture_bytes, per_wait.capture_alignment, per_wait.stream_receipts,
      per_wait.stream_bytes, per_wait.stream_alignment, per_wait.record_allocations,
      per_wait.record_requested_bytes, wait_count * per_wait.record_allocations,
      wait_count * per_wait.record_requested_bytes, per_wait.named_control_bytes};
  *out = value;
  return true;
}
extern "C" bool mlx_operation_event_root_storage_layout(
    mlx_operation_root_storage_layout* out, size_t root_count) {
  using namespace mlx::core;
  if (!out) return false;
  // Match the supported vector max_size without constructing a vector or a
  // default GraphAllocator (which would read the entered native domain).
  submission::GraphAllocator<array> allocator(nullptr);
  const size_t maximum = std::min<size_t>(
      std::allocator_traits<decltype(allocator)>::max_size(allocator),
      std::numeric_limits<ArrayVector::difference_type>::max());
  if (root_count > maximum || root_count > std::numeric_limits<size_t>::max() / sizeof(array))
    return false;
  mlx_operation_root_storage_layout value{};
  value.root_count = root_count;
  value.object_bytes = sizeof(OperationEvent);
  value.object_alignment = alignof(OperationEvent);
  value.roots_bytes = root_count * sizeof(array);
  value.roots_alignment = alignof(array);
  value.graph_blocks = root_count ? 2 : 1;
  if (!submission::GraphQuota::allocation_extent(
          value.object_bytes, value.object_alignment, value.object_graph_extent) ||
      (root_count && !submission::GraphQuota::allocation_extent(
          value.roots_bytes, value.roots_alignment, value.roots_graph_extent)) ||
      value.roots_graph_extent > std::numeric_limits<size_t>::max() - value.object_graph_extent)
    return false;
  value.graph_request_extent = value.object_graph_extent + value.roots_graph_extent;
  *out = value;
  return true;
}
extern "C" unsigned mlx_operation_event_new_exact(
    mlx_operation_event* out, mlx_submission_observer observer,
    mlx_stream selected_stream, size_t root_count) {
  using namespace mlx::core;
  if (!out || out->ctx || !selected_stream.ctx) return 4;
  mlx_operation_root_storage_layout layout{};
  if (!mlx_operation_event_root_storage_layout(&layout, root_count)) return 1;
  auto* scope = static_cast<submission::Scope*>(observer.ctx);
  if (!scope || !scope->valid_original_observer()) return 4;
  const auto ready = preflight_original_submission_stream(
      *scope, mlx_stream_get_(selected_stream));
  if (ready != ScopedEvaluation::complete) return static_cast<unsigned>(ready);
  // Reuse the unchanged dynamic constructor's exact owner checks/allocation.
  const auto created = mlx_operation_event_new(out, observer);
  if (created != 0) return created;
  auto* event = operation_get(*out); // *out already owns this partial wrapper
  event->root_state = OperationEvent::RootState::failed;
  event->expected_roots = root_count;
  try {
    event->roots.reserve(root_count);
    // Supported libc++ with GraphAllocator reserves exactly root_count. Never
    // report a smaller layout if another library supplies additional capacity.
    if (event->roots.capacity() != root_count) return 1;
    event->root_state = OperationEvent::RootState::ready;
    return 0;
  } catch (...) { return operation_failure(*event); }
}
extern "C" unsigned mlx_operation_event_append(mlx_operation_event value, mlx_array input) {
  auto* event = operation_get(value);
  if (!event || event->submitted || !input.ctx || !operation_current(*event) ||
      !mlx_c_detail::operation_accepts_root(*event)) return 4;
  try { event->roots.emplace_back(mlx_array_get_(input)); return 0; }
  catch (...) { return operation_failure(*event); }
}
extern "C" unsigned mlx_operation_event_submit(mlx_operation_event value) {
  using namespace mlx::core;
  auto* event = operation_get(value);
  if (!event || event->submitted || !operation_current(*event) ||
      !mlx_c_detail::operation_roots_complete(*event)) return 4;
  event->submitted = true; // no second attempt after any partial dispatch
  try {
    return static_cast<unsigned>(evaluation_progress(async_eval_with_completion_scoped(
        std::move(event->roots), *event->scope, event->completion)));
  } catch (...) { return operation_failure(*event); }
}
extern "C" unsigned mlx_operation_event_query(mlx_operation_event value) {
  auto* event = operation_get(value);
  if (!event || !event->submitted) return 5;
  return static_cast<unsigned>(mlx::core::query_scoped_completion(*event->scope, event->completion));
}
namespace {
unsigned submit_operation_stream(mlx_operation_event value, mlx_stream stream,
    const mlx_operation_eval_traversal_limits* limits) {
  using namespace mlx::core;
  auto* event = operation_get(value);
  if (!event || event->submitted || !stream.ctx || !operation_current(*event) ||
      !mlx_c_detail::operation_roots_complete(*event)) return 4;
  const auto selected = mlx_stream_get_(stream);
  const auto ready = preflight_original_submission_stream(*event->scope, selected);
  if (ready != ScopedEvaluation::complete) return static_cast<unsigned>(ready);
  mlx::core::CompletionEvalTraversalLimits prepared{};
  if (limits) {
    mlx::core::CompletionEvalTraversalLayout layout;
    prepared = native_traversal_limits(*limits);
    if (prepared.root_count != event->roots.size() ||
        !completion_eval_traversal_layout(prepared, layout)) return 1;
  }
  // Both exact prepared stream and current owner were checked before consuming
  // this operation or creating Graph/Record storage. The evaluator repeats
  // admission when its real receipt is created, before any dispatch.
  event->submitted = true;
  try {
    const auto outcome = limits
        ? async_eval_with_completion_scoped_on_stream_prepared(
              std::move(event->roots), *event->scope, event->completion, selected, prepared)
        : async_eval_with_completion_scoped_on_stream(
              std::move(event->roots), *event->scope, event->completion, selected);
    return static_cast<unsigned>(evaluation_progress(outcome));
  } catch (...) { return operation_failure(*event); }
}
} // namespace
extern "C" unsigned mlx_operation_event_submit_on_stream(
    mlx_operation_event value, mlx_stream stream) {
  return submit_operation_stream(value, stream, nullptr);
}
extern "C" unsigned mlx_operation_event_submit_on_stream_prepared(
    mlx_operation_event value, mlx_stream stream,
    const mlx_operation_eval_traversal_limits* limits) {
  return limits ? submit_operation_stream(value, stream, limits) : 4;
}
extern "C" unsigned mlx_operation_event_wait(mlx_operation_event value) {
  auto* event = operation_get(value);
  if (!event || !event->submitted) return 5;
  return static_cast<unsigned>(mlx::core::wait_scoped_completion(*event->scope, event->completion));
}
extern "C" unsigned mlx_operation_event_wait_stream(mlx_operation_event value, mlx_stream stream) {
  auto* event = operation_get(value);
  if (!event || !event->submitted || !stream.ctx || !operation_current(*event)) return 4;
  return static_cast<unsigned>(event->completion.wait_scoped(*event->scope, mlx_stream_get_(stream)));
}
extern "C" void mlx_operation_event_free(mlx_operation_event value) {
  if (auto* event = operation_get(value)) destroy_operation(event);
}
extern "C" void mlx_operation_event_defer(mlx_operation_event value) {
  if (auto* event = operation_get(value)) {
    event->destroy = [](mlx::core::submission::ScopedRetirement* node) noexcept {
      destroy_operation(static_cast<OperationEvent*>(node));
    };
    event->scope->defer_retirement(event);
  }
}
extern "C" unsigned mlx_operation_event_retire_deferred(mlx_submission_observer observer) {
  auto* scope = static_cast<mlx::core::submission::Scope*>(observer.ctx);
  if (!scope || !scope->valid_original_observer()) return 4;
  return static_cast<unsigned>(mlx::core::evaluation_progress(scope->retire_deferred()));
}
extern "C" size_t mlx_operation_event_control_bytes(void) {
  return sizeof(OperationEvent) + sizeof(mlx_operation_event) +
      sizeof(mlx_submission_observer) + sizeof(OperationEvent*) * 3 +
      sizeof(mlx::core::submission::Scope*) + sizeof(mlx::core::submission::GraphQuota*) +
      // Named exact-root query/construction controls, not a whole-stack bound.
      sizeof(mlx_operation_root_storage_layout) + // query value
      sizeof(mlx_operation_root_storage_layout) + // constructor layout
      sizeof(mlx::core::submission::GraphAllocator<mlx::core::array>) + // null allocator
      sizeof(size_t) + // maximum
      sizeof(mlx::core::submission::Scope*) + // constructor scope
      sizeof(mlx::core::ScopedEvaluation) + // ready
      sizeof(unsigned) + // created
      sizeof(OperationEvent*) + // constructed partial owner
      sizeof(mlx_operation_wait_record_layout) + // query output value
      sizeof(mlx::core::CompletionWaitRecordLayout) + // per-wait query
      sizeof(size_t) + // checked aggregate maximum
      sizeof(mlx::core::CompletionEvalTraversalLimits) + // submit conversion
      sizeof(mlx::core::CompletionEvalTraversalLayout) + // submit validation
      sizeof(mlx_operation_eval_traversal_limits*) + // borrowed plan
      sizeof(mlx::core::array*) + // exact leaf validation loan
      sizeof(mlx::core::ScopedEvaluation); // leaf health observation
}

extern "C" mlx_event mlx_event_new(void) {
  try {
    return mlx_event_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_event({nullptr});
  }
}

extern "C" int mlx_event_free(mlx_event event) {
  try {
    mlx_event_free_(event);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_event_synchronize(mlx_event event) {
  try {
    mlx_event_get_(event).wait();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_event_query(bool* complete, mlx_event event) {
  try {
    *complete = mlx_event_get_(event).is_complete();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_stream_wait_event(mlx_stream stream, mlx_event event) {
  try {
    mlx_event_get_(event).wait(mlx_stream_get_(stream));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_event_has_device(bool* has_device, mlx_event event) {
  try {
    *has_device = mlx_event_get_(event).device().has_value();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_event_get_device(mlx_device* device, mlx_event event) {
  try {
    auto& producer = mlx_event_get_(event).device();
    if (!producer) {
      throw std::runtime_error(
          "[mlx_event_get_device] Event has no producer device identity.");
    }
    mlx_device_set_(*device, *producer);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_event_get_backend(
    mlx_event_backend* backend,
    mlx_event event) {
  try {
    switch (mlx_event_get_(event).backend()) {
      case mlx::core::CompletionBackend::none:
        *backend = MLX_EVENT_BACKEND_NONE;
        break;
      case mlx::core::CompletionBackend::cpu:
        *backend = MLX_EVENT_BACKEND_CPU;
        break;
      case mlx::core::CompletionBackend::metal:
        *backend = MLX_EVENT_BACKEND_METAL;
        break;
      case mlx::core::CompletionBackend::cuda:
        *backend = MLX_EVENT_BACKEND_CUDA;
        break;
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_event_has_timing(bool* has_timing, mlx_event event) {
  try {
    *has_timing = mlx_event_get_(event).is_timed();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_event_elapsed(double* seconds, mlx_event event) {
  try {
    *seconds = mlx_event_get_(event).elapsed_seconds();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_event_try_elapsed(
    double* seconds,
    bool* ready,
    mlx_event event) {
  try {
    auto elapsed = mlx_event_get_(event).try_elapsed_seconds();
    *ready = elapsed.has_value();
    if (elapsed) {
      *seconds = *elapsed;
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" bool mlx_operation_event_gpu_eval_prologue_layout(
    mlx_gpu_eval_prologue_layout* out, size_t inputs, size_t siblings) {
  return mlx_operation_event_gpu_eval_prologue_layout_with_tracing(out, inputs, siblings, false);
}
extern "C" bool mlx_operation_event_gpu_eval_prologue_layout_with_tracing(
    mlx_gpu_eval_prologue_layout* out, size_t inputs, size_t siblings, bool tracer) {
  if (!out) return false;
  mlx::core::submission::GpuEvalPrologueLayout native;
  if (!mlx::core::submission::gpu_eval_prologue_layout(inputs, siblings, tracer, native)) return false;
  mlx_gpu_eval_prologue_layout value{};
  value.inputs = native.inputs; value.siblings = native.siblings;
  value.tracer_inputs = native.tracer_inputs;
  std::copy_n(native.request_bytes, 5, value.request_bytes);
  std::copy_n(native.request_alignments, 5, value.request_alignments);
  value.header_bytes = native.header_bytes; value.header_alignment = native.header_alignment;
  value.slots_bytes = native.slots_bytes; value.slots_alignment = native.slots_alignment;
  value.blocks = native.blocks; value.reserved_alignment = native.reserved_alignment;
  value.requested_bytes = native.requested_bytes; value.allocation_extents = native.allocation_extents;
  value.data_slot_bytes = native.data_slot_bytes; value.data_slot_alignment = native.data_slot_alignment;
  value.array_slot_bytes = native.array_slot_bytes; value.array_slot_alignment = native.array_slot_alignment;
  // Include the compatibility wrapper's named arguments in this shared bound.
  constexpr size_t transports = sizeof(native) + sizeof(value) + 2 * sizeof(out) +
      4 * sizeof(size_t) + sizeof(tracer);
  if (native.named_control_bytes > std::numeric_limits<size_t>::max() - transports) return false;
  value.named_control_bytes = native.named_control_bytes + transports;
  *out = value;
  return true;
}

extern "C" bool mlx_operation_event_cpu_eval_cleanup_layout(
    mlx_cpu_eval_cleanup_layout* out, size_t inputs, size_t siblings) {
  if (!out) return false;
  mlx::core::submission::CpuEvalCleanupLayout native;
  if (!mlx::core::submission::cpu_eval_cleanup_layout(inputs, siblings, native)) return false;
  mlx_cpu_eval_cleanup_layout value{};
  value.inputs = native.inputs; value.siblings = native.siblings;
  std::copy_n(native.request_bytes, 2, value.request_bytes);
  std::copy_n(native.request_alignments, 2, value.request_alignments);
  value.header_bytes = native.header_bytes; value.header_alignment = native.header_alignment;
  value.slots_bytes = native.slots_bytes; value.slots_alignment = native.slots_alignment;
  value.blocks = native.blocks; value.reserved_alignment = native.reserved_alignment;
  value.requested_bytes = native.requested_bytes; value.allocation_extents = native.allocation_extents;
  value.data_slot_bytes = native.data_slot_bytes; value.data_slot_alignment = native.data_slot_alignment;
  value.named_control_bytes = native.named_control_bytes + sizeof(native) + sizeof(value) +
      sizeof(out) + sizeof(inputs) + sizeof(siblings);
  *out = value;
  return true;
}

extern "C" bool mlx_operation_event_resident_graph_layout(
    mlx_resident_graph_layout* out, size_t primitives, size_t seeds, size_t maximum_rank) {
  return mlx_operation_event_resident_graph_layout_with_operands(out, primitives, seeds, maximum_rank, 4);
}
extern "C" bool mlx_operation_event_resident_graph_layout_with_operands(
    mlx_resident_graph_layout* out, size_t primitives, size_t seeds, size_t maximum_rank,
    size_t maximum_operands) {
  return mlx_operation_event_resident_graph_layout_with_shells(out, primitives, seeds,
      maximum_rank, maximum_operands, 0);
}
extern "C" bool mlx_operation_event_resident_graph_layout_with_shells(
    mlx_resident_graph_layout* out, size_t primitives, size_t seeds, size_t maximum_rank,
    size_t maximum_operands, size_t additional_shells) {
  if (!out) return false;
  mlx::core::submission::ResidentGraphLayout native;
  if (!mlx::core::submission::resident_graph_layout_with_shells(primitives, seeds, maximum_rank, maximum_operands, additional_shells, native)) return false;
  mlx_resident_graph_layout value{};
  value.primitives = native.primitives; value.seeds = native.seeds; value.maximum_rank = native.maximum_rank; value.maximum_operands = native.maximum_operands;
  std::copy_n(native.request_bytes, 10, value.request_bytes);
  std::copy_n(native.request_alignments, 10, value.request_alignments);
  std::copy_n(native.request_counts, 10, value.request_counts);
  value.blocks = native.blocks; value.header_bytes = native.header_bytes;
  value.header_alignment = native.header_alignment; value.slots_bytes = native.slots_bytes;
  value.slots_alignment = native.slots_alignment; value.reserved_alignment = native.reserved_alignment;
  value.requested_bytes = native.requested_bytes; value.allocation_extents = native.allocation_extents;
  value.named_control_bytes = native.named_control_bytes + sizeof(native) + sizeof(value) +
      sizeof(out) + sizeof(primitives) + sizeof(seeds) + sizeof(maximum_rank) + sizeof(maximum_operands) + sizeof(additional_shells) +
      sizeof(void**) + sizeof(mlx_submission_observer) + sizeof(unsigned);
  *out = value; return true;
}
extern "C" unsigned mlx_operation_event_prepare_resident_graph(
    void** out, mlx_submission_observer observer, size_t primitives, size_t seeds, size_t maximum_rank) {
  return mlx_operation_event_prepare_resident_graph_with_operands(out, observer, primitives, seeds, maximum_rank, 4);
}
extern "C" unsigned mlx_operation_event_prepare_resident_graph_with_operands(
    void** out, mlx_submission_observer observer, size_t primitives, size_t seeds, size_t maximum_rank,
    size_t maximum_operands) {
  return mlx_operation_event_prepare_resident_graph_with_shells(out, observer, primitives,
      seeds, maximum_rank, maximum_operands, 0);
}
extern "C" unsigned mlx_operation_event_prepare_resident_graph_with_shells(
    void** out, mlx_submission_observer observer, size_t primitives, size_t seeds, size_t maximum_rank,
    size_t maximum_operands, size_t additional_shells) {
  using namespace mlx::core::submission;
  if (!out || *out) return 4;
  if (auto status = mlx_operation_event_validate_traversal_context(observer)) return status;
  auto* scope = static_cast<Scope*>(observer.ctx);
  ResidentGraphLayout layout;
  if (!resident_graph_layout_with_shells(primitives, seeds, maximum_rank, maximum_operands, additional_shells, layout)) return 4;
  GraphConstruction* owner = nullptr;
  const auto failure = GraphConstruction::create_resident(*scope->graph_quota(), scope->identity(), layout, owner);
  if (failure == GraphFailure::exhausted) return 2;
  if (failure == GraphFailure::construction_busy) return 10;
  if (static_cast<unsigned>(failure)) return 4;
  *out = owner; return 0;
}
extern "C" void mlx_operation_event_finish_resident_graph(void* owner) {
  if (owner) static_cast<mlx::core::submission::GraphConstruction*>(owner)->destroy();
}

namespace {
unsigned nested_graph_failure(mlx::core::submission::GraphFailure failure) noexcept {
  using F = mlx::core::submission::GraphFailure;
  if (failure == F::exhausted) return 2;
  if (failure == F::construction_busy) return 10;
  return static_cast<unsigned>(failure) ? 4 : 0;
}
}
extern "C" unsigned mlx_operation_event_configure_nested_graph(
    void* owner, mlx_submission_observer observer,
    const mlx_operation_eval_traversal_limits* limits, size_t attempts) {
  using namespace mlx::core;
  if (!owner || !limits || !attempts) return 4;
  if (auto status = mlx_operation_event_validate_traversal_context(observer)) return status;
  const auto native = native_traversal_limits(*limits);
  CompletionEvalTraversalLayout checked;
  if (!completion_eval_traversal_layout(native, checked)) return 4;
  const submission::NestedGraphEvaluationLimits plan{limits->root_count,
      limits->array_nodes, limits->tape_entries, limits->input_edges,
      limits->output_slots, limits->stream_count, limits->capture_slots};
  return nested_graph_failure(
      static_cast<submission::GraphConstruction*>(owner)->configure_nested(plan, attempts));
}
extern "C" unsigned mlx_operation_event_validate_nested_graph(
    mlx_submission_observer observer, size_t roots) {
  using namespace mlx::core::submission;
  if (auto status = mlx_operation_event_validate_traversal_context(observer)) return status;
  auto* scope = static_cast<Scope*>(observer.ctx);
  return nested_graph_failure(GraphConstruction::validate_nested(
      *scope->graph_quota(), scope->identity(), roots));
}
namespace {
// The Event becoming readable is distinct from the last CPU signal task
// releasing its Record-owned controls. Keep the resident bank suspended until
// both have completed. Refusals preserve the Scope's accepted Records; they do
// not certify completion or release their independently retained allocations.
unsigned settle_nested_event(
    mlx_operation_event& value, mlx::core::submission::Scope& scope,
    bool release_event) noexcept {
  using namespace mlx::core;
  using namespace mlx::core::submission;
  try {
    if (auto status = mlx_operation_event_wait(value)) return status;
    for (;;) {
      const auto progress = scope.progress_scoped();
      if (progress != ScopedProgress::observed)
        return static_cast<unsigned>(evaluation_progress(progress));
      const auto records = scope.query_records();
      if (records.failed || records.blocked) return 7;
      if (!records.pending) break;
      std::this_thread::yield();
    }
    if (release_event) {
      mlx_operation_event_free(value);
      value.ctx = nullptr;
    }
    if (auto retirement = scope.retire_scoped_records();
        retirement != ScopedProgress::observed)
      return static_cast<unsigned>(evaluation_progress(retirement));
    return 0;
  } catch (...) {
    scope.failure_owner().get()->capture_current_exception();
    return 7;
  }
}
size_t nested_settlement_control_bytes() noexcept {
  using namespace mlx::core::submission;
  return sizeof(mlx_operation_event*) + sizeof(Scope*) + sizeof(bool) +
      sizeof(unsigned) + 2 * sizeof(ScopedProgress) + sizeof(RecordStatus);
}
// Both public entries use this same exact nested attempt, Event constructor,
// prepared Eval and terminal-record worker. Mixed-stream Eval can queue CPU
// callbacks which borrow its graph banks: keep the resident bank suspended until
// those callbacks finish, even when the caller retains the completed Event.
unsigned run_nested_graph_event(
    mlx_operation_event* out, mlx_submission_observer observer,
    mlx_stream stream, const mlx_array* roots, size_t count, bool complete) {
  using namespace mlx::core;
  using namespace mlx::core::submission;
  if (!out || out->ctx || !stream.ctx || !roots || !count) return 4;
  for (size_t i = 0; i < count; ++i) if (!roots[i].ctx) return 4;
  if (auto status = mlx_operation_event_validate_nested_graph(observer, count)) return status;
  auto* scope = static_cast<Scope*>(observer.ctx);
  const auto selected = mlx_stream_get_(stream);
  const auto ready = preflight_original_submission_stream(*scope, selected);
  if (ready != ScopedEvaluation::complete) return static_cast<unsigned>(ready);
  GraphNestedEvaluation suspended;
  if (auto status = nested_graph_failure(GraphConstruction::suspend_nested(
          *scope->graph_quota(), scope->identity(), count, suspended))) return status;
  const auto& bound = suspended.limits();
  // CPU callbacks borrow the suspended graph's banks just as mixed-stream
  // callbacks do. Finish their real records before restoring that bank while
  // retaining the actual completed Event for the ordinary consumer lifecycle.
  const bool settle_records =
      complete || selected.device.type == Device::cpu || bound.streams != 1;
  struct EventOwner {
    mlx_operation_event raw{nullptr};
    ~EventOwner() { reset(); }
    void reset() noexcept { if (raw.ctx) { mlx_operation_event_free(raw); raw.ctx = nullptr; } }
  } event;
  try {
    auto status = mlx_operation_event_new_exact(&event.raw, observer, stream, count);
    if (status) return status;
    for (size_t i = 0; i < count; ++i)
      if ((status = mlx_operation_event_append(event.raw, roots[i]))) return status;
    const mlx_operation_eval_traversal_limits limits{count, bound.arrays,
        bound.tape_entries, bound.input_edges, bound.output_slots, bound.streams, bound.captures};
    status = mlx_operation_event_submit_on_stream_prepared(event.raw, stream, &limits);
    if (status) return status;
    if (settle_records) {
      // Complete-only retains the former Event-before-record destruction order.
      // A published completed Event retains its roots and quota as usual.
      if ((status = settle_nested_event(event.raw, *scope, complete))) return status;
    }
    if (!complete) {
      *out = event.raw;
      event.raw.ctx = nullptr;
    }
    return 0;
  } catch (...) {
    scope->failure_owner().get()->capture_current_exception();
    return 7;
  }
}
} // namespace
extern "C" unsigned mlx_operation_event_complete_nested_graph(
    mlx_submission_observer observer, mlx_stream stream,
    const mlx_array* roots, size_t count) {
  mlx_operation_event unused{nullptr};
  return run_nested_graph_event(&unused, observer, stream, roots, count, true);
}
extern "C" unsigned mlx_operation_event_submit_nested_graph_roots(
    mlx_operation_event* out, mlx_submission_observer observer,
    mlx_stream stream, const mlx_array* roots, size_t count) {
  return run_nested_graph_event(out, observer, stream, roots, count, false);
}
extern "C" unsigned mlx_operation_event_submit_nested_graph(
    mlx_operation_event* out, mlx_submission_observer observer,
    mlx_stream stream, mlx_array root) {
  return mlx_operation_event_submit_nested_graph_roots(out, observer, stream, &root, 1);
}
extern "C" size_t mlx_operation_event_nested_graph_control_bytes(void) {
  using namespace mlx::core;
  using namespace mlx::core::submission;
  return sizeof(GraphNestedEvaluation) + sizeof(NestedGraphEvaluationLimits) +
      sizeof(CompletionEvalTraversalLimits) + sizeof(CompletionEvalTraversalLayout) +
      sizeof(mlx_operation_eval_traversal_limits) + sizeof(mlx_operation_event) +
      sizeof(mlx_submission_observer) + sizeof(mlx_stream) + sizeof(Stream) +
      sizeof(Scope*) + sizeof(GraphConstruction*) + sizeof(const mlx_array*) +
      4 * sizeof(size_t) + 3 * sizeof(unsigned) + sizeof(ScopedEvaluation) +
      2 * sizeof(bool) + nested_settlement_control_bytes() +
      sizeof(mlx_operation_event) + sizeof(mlx_operation_event*) +
      scoped_observer_control_bytes();
}

namespace {
bool write_cpu_partition_layout(mlx_cpu_argpartition_layout* out, bool tracer,
    const mlx::core::submission::CpuArgPartitionLayout& native) {
  if (!out) return false;
  mlx_cpu_argpartition_layout value{};
  std::copy_n(native.request_bytes, 8, value.request_bytes);
  std::copy_n(native.request_alignments, 8, value.request_alignments);
  std::copy_n(native.request_counts, 8, value.request_counts);
  value.header_bytes = native.header_bytes; value.header_alignment = native.header_alignment;
  value.slots_bytes = native.slots_bytes; value.slots_alignment = native.slots_alignment;
  value.blocks = native.blocks; value.reserved_alignment = native.reserved_alignment;
  value.requested_bytes = native.requested_bytes; value.allocation_extents = native.allocation_extents;
  constexpr size_t transports = sizeof(native) + sizeof(value) + sizeof(out) + sizeof(tracer);
  if (native.named_control_bytes > std::numeric_limits<size_t>::max() - transports) return false;
  value.named_control_bytes = native.named_control_bytes + transports;
  *out = value;
  return true;
}

} // namespace
extern "C" bool mlx_operation_event_cpu_argpartition_layout(mlx_cpu_argpartition_layout* out, bool tracer) {
  mlx::core::submission::CpuArgPartitionLayout native;
  return mlx::core::submission::cpu_argpartition_layout(tracer,native) && write_cpu_partition_layout(out,tracer,native);
}
extern "C" bool mlx_operation_event_cpu_argpartition_source_layout(mlx_cpu_argpartition_layout* out,
    size_t rank,size_t elements,bool tracer) {
  mlx::core::submission::CpuArgPartitionLayout native;
  if(!mlx::core::submission::cpu_argpartition_source_layout(rank,elements,tracer,native))return false;
  const size_t controls=sizeof(rank)+sizeof(elements)+sizeof(native)+sizeof(out)+sizeof(tracer);
  if(controls>SIZE_MAX-native.named_control_bytes)return false;
  native.named_control_bytes+=controls;
  return write_cpu_partition_layout(out,tracer,native);
}

extern "C" bool mlx_operation_event_router_receipt_layout(mlx_router_receipt_layout* out) {
  if (!out) return false;
  mlx::core::EventControlLayout event{};
  mlx::core::EventDispatchControlLayout dispatch{};
  mlx::core::FenceControlLayout fence{};
  mlx::core::submission::OriginalBufferRoutingLayout routing{};
  size_t allocation_controls = 0;
  if (!mlx::core::event_control_layout(event) ||
      !mlx::core::event_dispatch_control_layout(dispatch) ||
      !mlx::core::fence_control_layout(fence) ||
      !mlx::core::submission::original_buffer_routing_layout(routing) ||
      !mlx::core::allocator::original_buffer_allocation_control_bytes(allocation_controls)) return false;
  mlx_router_receipt_layout value{};
  const auto add = [](size_t& total, size_t bytes, size_t count) {
    if (count && bytes > (std::numeric_limits<size_t>::max() - total) / count) return false;
    total += count * bytes; return true;
  };
  // Two stream Events and at most two slow-Fence Events. A fast Fence replaces
  // its Event with one U32 birth. CPU input wait/output update each enqueue one
  // task; final CPU Event signal enqueues one more. GPU wait/update each retain
  // one handler; its final stream Event signal retains one more.
  if (!add(value.graph_requested_bytes, event.shared_bytes, 4) ||
      !add(value.graph_requested_bytes, fence.shared_bytes, 2) ||
      !add(value.graph_requested_bytes, routing.birth_bytes, 2) ||
      !add(value.graph_requested_bytes, std::max(fence.cpu_wait_bytes, dispatch.cpu_wait_bytes), 1) ||
      !add(value.graph_requested_bytes, std::max(fence.cpu_update_bytes, dispatch.cpu_signal_bytes), 1) ||
      !add(value.graph_requested_bytes, dispatch.cpu_signal_bytes, 1) ||
      !add(value.graph_requested_bytes, std::max(fence.gpu_handler_bytes, dispatch.gpu_handler_bytes), 2) ||
      !add(value.graph_requested_bytes, dispatch.gpu_handler_bytes, 1) ||
      !add(value.graph_allocation_extents, event.graph_extent, 4) ||
      !add(value.graph_allocation_extents, fence.shared_extent, 2) ||
      !add(value.graph_allocation_extents, routing.birth_graph_extent, 2) ||
      !add(value.graph_allocation_extents, std::max(fence.cpu_wait_extent, dispatch.cpu_wait_extent), 1) ||
      !add(value.graph_allocation_extents, std::max(fence.cpu_update_extent, dispatch.cpu_signal_extent), 1) ||
      !add(value.graph_allocation_extents, dispatch.cpu_signal_extent, 1) ||
      !add(value.graph_allocation_extents, std::max(fence.gpu_handler_extent, dispatch.gpu_handler_extent), 2) ||
      !add(value.graph_allocation_extents, dispatch.gpu_handler_extent, 1) ||
      !add(value.platform_events, event.platform_events, 4) ||
      !add(value.fast_backing_bytes, fence.fast_backing_bytes, 2) ||
      !add(value.fast_backing_births, fence.fast_backing_births, 2) ||
      !add(value.named_control_bytes, event.construction_controls, 4) ||
      !add(value.named_control_bytes, fence.controls, 2) ||
      !add(value.named_control_bytes, dispatch.controls, 3) ||
      !add(value.named_control_bytes, allocation_controls, 2) ||
      !add(value.named_control_bytes, sizeof(value) + sizeof(event) + sizeof(dispatch) +
          sizeof(fence) + sizeof(routing) + sizeof(out) + sizeof(add) + 4 * sizeof(size_t), 1)) return false;
  *out = value;
  return true;
}

extern "C" bool mlx_operation_event_resident_gpu_worker_layout(
    mlx_resident_gpu_worker_layout* out, size_t entries, size_t input_edges,
    size_t output_slots, size_t array_nodes, size_t backing_births,
    size_t maximum_rank, size_t maximum_operands) {
  if (!out) return false;
  mlx::core::submission::ResidentGpuWorkerLayout native;
  if (!mlx::core::submission::resident_gpu_worker_layout(entries, input_edges,
      output_slots, array_nodes, backing_births, maximum_rank, maximum_operands, native)) return false;
  const size_t transports = sizeof(native) + sizeof(mlx_resident_gpu_worker_layout) +
      sizeof(out) + 7 * sizeof(size_t);
  if (native.named_control_bytes > SIZE_MAX - transports) return false;
  *out = {native.allocation_extents, native.named_control_bytes + transports,
      native.kernel_attempts, native.encoding_intervals, native.buffer_universe};
  return true;
}

extern "C" bool mlx_operation_event_resident_gpu_worker_layout_with_sorts(
    mlx_resident_gpu_worker_layout* out, size_t entries, size_t input_edges,
    size_t output_slots, size_t array_nodes, size_t backing_births,
    size_t maximum_rank, size_t maximum_operands, size_t additional_sort_kernels) {
  if (!out) return false;
  mlx::core::submission::ResidentGpuWorkerLayout native;
  if (!mlx::core::submission::resident_gpu_worker_layout_with_sorts(entries, input_edges,
      output_slots, array_nodes, backing_births, maximum_rank, maximum_operands, additional_sort_kernels, native)) return false;
  const size_t transports = sizeof(native) + sizeof(mlx_resident_gpu_worker_layout) +
      sizeof(out) + 8 * sizeof(size_t);
  if (native.named_control_bytes > SIZE_MAX - transports) return false;
  *out = {native.allocation_extents, native.named_control_bytes + transports,
      native.kernel_attempts, native.encoding_intervals, native.buffer_universe};
  return true;
}

extern "C" bool mlx_operation_event_resident_gpu_worker_layout_with_router(
    mlx_resident_gpu_worker_layout* out, size_t entries, size_t input_edges,
    size_t output_slots, size_t array_nodes, size_t backing_births,
    size_t maximum_rank, size_t maximum_operands, size_t additional_sort_kernels, size_t cpu_partitions) {
  if (!out) return false;
  mlx::core::submission::ResidentGpuWorkerLayout native;
  if (!mlx::core::submission::resident_gpu_worker_layout_with_router(entries, input_edges,
      output_slots, array_nodes, backing_births, maximum_rank, maximum_operands, additional_sort_kernels, cpu_partitions, native)) return false;
  const size_t transports = sizeof(native) + sizeof(mlx_resident_gpu_worker_layout) +
      sizeof(out) + 9 * sizeof(size_t);
  if (native.named_control_bytes > SIZE_MAX - transports) return false;
  *out = {native.allocation_extents, native.named_control_bytes + transports,
      native.kernel_attempts, native.encoding_intervals, native.buffer_universe};
  return true;
}

extern "C" bool mlx_operation_event_resident_gpu_worker_layout_with_frontiers(
    mlx_resident_gpu_worker_layout* out, size_t entries, size_t input_edges,
    size_t output_slots, size_t array_nodes, size_t backing_births,
    size_t maximum_rank, size_t maximum_operands, size_t additional_sort_kernels,
    size_t cpu_partitions, size_t evaluations, size_t consumer_waits) {
  if (!out) return false;
  mlx::core::submission::ResidentGpuWorkerLayout native;
  if (!mlx::core::submission::resident_gpu_worker_layout_with_frontiers(entries, input_edges,
      output_slots, array_nodes, backing_births, maximum_rank, maximum_operands,
      additional_sort_kernels, cpu_partitions, evaluations, consumer_waits, native)) return false;
  const size_t transports = sizeof(native) + sizeof(mlx_resident_gpu_worker_layout) +
      sizeof(out) + 11 * sizeof(size_t);
  if (native.named_control_bytes > SIZE_MAX - transports) return false;
  *out = {native.allocation_extents, native.named_control_bytes + transports,
      native.kernel_attempts, native.encoding_intervals, native.buffer_universe};
  return true;
}

extern "C" bool mlx_operation_event_resident_grouped_sort_additional_kernels(size_t selections, size_t* out) {
  if (!out) return false;
  size_t native;
  if (!mlx::core::submission::resident_grouped_sort_additional_kernels(selections, native)) return false;
  *out = native;
  return true;
}

extern "C" unsigned mlx_operation_event_new_nested_scheduled(
    mlx_operation_event* out, mlx_submission_observer observer,
    mlx_stream stream, size_t roots) {
  using namespace mlx::core;
  using namespace mlx::core::submission;
  if (!out || out->ctx || !stream.ctx || roots == SIZE_MAX) return 4;
  if (auto status = mlx_operation_event_validate_nested_graph(observer, std::max<size_t>(1, roots))) return status;
  auto* scope = static_cast<Scope*>(observer.ctx);
  const auto selected = mlx_stream_get_(stream);
  if (selected.device.type != Device::gpu && selected.device.type != Device::cpu) return 4;
  const auto ready = preflight_original_submission_stream(*scope, selected);
  if (ready != ScopedEvaluation::complete) return static_cast<unsigned>(ready);
  if (auto status = mlx_operation_event_new(out, observer)) return status;
  auto* event = operation_get(*out); // partial owner remains in *out on refusal
  event->root_state = OperationEvent::RootState::failed;
  event->expected_roots = roots;
  event->nested.emplace();
  if (auto status = nested_graph_failure(GraphConstruction::suspend_nested(
          *scope->graph_quota(), scope->identity(), std::max<size_t>(1, roots), *event->nested))) return status;
  const auto& bound = event->nested->limits();
  CompletionEvalRecordLayout record;
  // Suspended limits describe the enclosing bank's capacity. This producer
  // authenticates one selected stream above and submits only that stream's
  // new Synchronizer below; already-scheduled roots add no lazy traversal.
  if (bound.streams < 1 || bound.arrays < roots + 1 || bound.input_edges < roots ||
      !bound.tape_entries || !bound.output_slots ||
      !completion_eval_record_layout(1, 1, 1, record) || bound.captures < record.capture_slots)
    return 4;
  try {
    event->roots.reserve(roots);
    if (event->roots.capacity() != roots) return 1;
    event->root_state = OperationEvent::RootState::ready;
    return 0;
  } catch (...) { return operation_failure(*event); }
}
extern "C" unsigned mlx_operation_event_append_nested_scheduled(
    mlx_operation_event value, mlx_array root) {
  const auto* event = operation_get(value);
  if (!event || !event->nested || !root.ctx ||
      mlx_array_get_(root).status() == mlx::core::array::Status::unscheduled) return 4;
  return mlx_operation_event_append(value, root);
}
extern "C" void mlx_operation_event_abort_nested(mlx_operation_event value) {
  auto* event = operation_get(value);
  if (!event || !event->nested) return;
  // Called under the same native runtime loan before the bank owner can leave.
  // A retained Record owns any accepted work after this unpublished Event dies.
  event->roots.clear();
  event->completion = mlx::core::Completion{};
  event->nested.reset();
}
extern "C" unsigned mlx_operation_event_submit_nested_scheduled(
    mlx_operation_event value, mlx_stream stream) {
  using namespace mlx::core;
  auto* event = operation_get(value);
  if (!event || !event->nested || !stream.ctx || !operation_current(*event) ||
      !mlx_c_detail::operation_roots_complete(*event)) return 4;
  CompletionEvalRecordLayout record;
  if (!completion_eval_record_layout(1, 1, 1, record)) return 1;
  const auto roots = event->expected_roots;
  const mlx_operation_eval_traversal_limits limits{roots, roots + 1, 1, roots, 1, 1, record.capture_slots};
  const auto status = mlx_operation_event_submit_on_stream_prepared(value, stream, &limits);
  if (status) return status;
  if (mlx_stream_get_(stream).device.type == Device::cpu) {
    if (auto settled = settle_nested_event(value, *event->scope, false)) return settled;
  }
  // GPU prologues are fully constructed; CPU signal tasks and their actual
  // Records have settled. Neither path refunds this nested attempt.
  event->nested.reset();
  return 0;
}

extern "C" bool mlx_operation_event_cpu_copy_eval_layout(
    mlx_cpu_copy_eval_layout* out, size_t rank, size_t inputs, bool copy, bool tracer) {
  if (!out) return false;
  mlx::core::cpu::CopyEvalStorage native;
  if (!mlx::core::cpu::copy_eval_layout(rank, inputs, copy, tracer, native)) return false;
  mlx::core::EventDispatchControlLayout dispatch{};
  if (!copy && !mlx::core::event_dispatch_control_layout(dispatch)) return false;
  size_t controls = sizeof(native) + sizeof(*out) + sizeof(out) + sizeof(dispatch) +
      sizeof(rank) + sizeof(inputs) + sizeof(copy) + sizeof(tracer) + sizeof(size_t);
  if (dispatch.controls > SIZE_MAX - controls) return false;
  controls += dispatch.controls;
  if (native.named_control_bytes > SIZE_MAX - controls) return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,
      native.worker_graph_extents, native.backing_births, native.named_control_bytes + controls,
      copy ? 0 : dispatch.cpu_signal_extent};
  *out = value;
  return true;
}

extern "C" bool mlx_operation_event_cpu_host_transfer_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype dtype,size_t rank,bool store,bool tracer) {
  using namespace mlx::core;
  if(!out||dtype<MLX_BOOL||dtype>MLX_COMPLEX64)return false;
  cpu::CopyEvalStorage native;
  if(!cpu::host_transfer_eval_layout(mlx_dtype_to_cpp(dtype),rank,store,tracer,native))return false;
  const size_t controls=sizeof(out)+sizeof(*out)+sizeof(dtype)+sizeof(rank)+sizeof(store)+sizeof(tracer)+
      sizeof(native)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

#include "mlx/backend/cpu/unary_storage.h"
#include "mlx/c/private/array.h"
extern "C" bool mlx_operation_event_cpu_unary_eval_layout(mlx_cpu_unary_eval_layout* out,
    uint32_t operation, mlx_dtype dtype, size_t rank, bool tracer) {
  using namespace mlx::core;
  if (!out || operation > uint32_t(cpu::UnaryEvalKind::round) || dtype < MLX_BOOL || dtype > MLX_COMPLEX64) return false;
  cpu::UnaryEvalStorage native;
  if (!cpu::unary_eval_layout(static_cast<cpu::UnaryEvalKind>(operation), mlx_dtype_to_cpp(dtype),
      rank, tracer, native)) return false;
  const size_t controls = sizeof(native) + sizeof(*out) + sizeof(out) + sizeof(operation) +
      sizeof(dtype) + sizeof(rank) + sizeof(tracer) + sizeof(size_t) + sizeof(Dtype);
  if (native.named_control_bytes > SIZE_MAX - controls) return false;
  const mlx_cpu_unary_eval_layout value{native.allocation_extents,
      native.worker_graph_extents, native.backing_births, native.named_control_bytes + controls};
  *out = value; return true;
}

#include "mlx/backend/cpu/binary_storage.h"
extern "C" bool mlx_operation_event_cpu_binary_eval_layout(mlx_cpu_binary_eval_layout* out,
    uint32_t operation, mlx_dtype dtype, size_t rank, size_t elements, bool tracer) {
  using namespace mlx::core;
  if (!out || operation > uint32_t(cpu::BinaryEvalKind::log_add_exp) || dtype < MLX_BOOL || dtype > MLX_COMPLEX64) return false;
  cpu::BinaryEvalStorage native;
  if (!cpu::binary_eval_layout(static_cast<cpu::BinaryEvalKind>(operation), mlx_dtype_to_cpp(dtype),
      rank, elements, tracer, native)) return false;
  const size_t controls = sizeof(native) + sizeof(*out) + sizeof(out) + sizeof(operation) +
      sizeof(dtype) + sizeof(rank) + sizeof(elements) + sizeof(tracer) + sizeof(size_t) + sizeof(Dtype);
  if (native.named_control_bytes > SIZE_MAX - controls) return false;
  const mlx_cpu_binary_eval_layout value{native.allocation_extents,
      native.worker_graph_extents, native.backing_births, native.named_control_bytes + controls};
  *out = value; return true;
}

extern "C" bool mlx_operation_event_cpu_cast_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype source, mlx_dtype destination, size_t rank, size_t elements, bool tracer) {
  using namespace mlx::core;
  if (!out || source < MLX_BOOL || source > MLX_COMPLEX64 ||
      destination < MLX_BOOL || destination > MLX_COMPLEX64) return false;
  cpu::CopyEvalStorage native;
  if (!cpu::cast_eval_layout(rank, mlx_dtype_to_cpp(source), mlx_dtype_to_cpp(destination),
      elements, tracer, native)) return false;
  const size_t controls = sizeof(native) + sizeof(*out) + sizeof(out) + sizeof(source) +
      sizeof(destination) + sizeof(rank) + sizeof(elements) + sizeof(tracer) + sizeof(size_t);
  if (native.named_control_bytes > SIZE_MAX - controls) return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents, native.worker_graph_extents,
      native.backing_births, native.named_control_bytes + controls, 0};
  *out = value; return true;
}
extern "C" bool mlx_operation_event_cpu_byte_view_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype source, mlx_dtype destination, size_t rank, size_t bytes, bool copy, bool tracer) {
  using namespace mlx::core;
  if (!out || source < MLX_BOOL || source > MLX_COMPLEX64 ||
      destination < MLX_BOOL || destination > MLX_COMPLEX64) return false;
  cpu::CopyEvalStorage native;
  if (!cpu::byte_view_eval_layout(mlx_dtype_to_cpp(source), mlx_dtype_to_cpp(destination),rank,
      bytes, copy, tracer, native)) return false;
  const size_t controls = sizeof(native) + sizeof(*out) + sizeof(out) + sizeof(source) +
      sizeof(destination) + sizeof(rank) + sizeof(bytes) + sizeof(copy) + sizeof(tracer) + sizeof(size_t);
  if (native.named_control_bytes > SIZE_MAX - controls) return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents, native.worker_graph_extents,
      native.backing_births, native.named_control_bytes + controls, 0};
  *out = value; return true;
}

#include "mlx/backend/cpu/slice_storage.h"
extern "C" bool mlx_operation_event_cpu_slice_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,bool empty,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::slice_eval_layout(rank,empty,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(empty)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_scalar_update_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t elements,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::scalar_update_eval_layout(elements,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(elements)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_static_update_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t elements,size_t update_elements,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::static_update_eval_layout(rank,elements,update_elements,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(elements)+sizeof(update_elements)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

#include "mlx/backend/cpu/argsort_f32.h"
extern "C" bool mlx_operation_event_cpu_argsort_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype source,size_t rank,size_t columns,size_t rows,bool tracer) {
  if(!out||(source!=MLX_FLOAT32&&source!=MLX_INT32))return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::argsort_eval_layout(mlx_dtype_to_cpp(source),rank,columns,rows,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(source)+sizeof(rank)+
      sizeof(columns)+sizeof(rows)+sizeof(tracer)+sizeof(size_t)+sizeof(mlx::core::Dtype);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

#include "mlx/backend/cpu/softmax_storage.h"
extern "C" bool mlx_operation_event_cpu_softmax_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t columns,size_t rows,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::softmax_eval_layout(rank,columns,rows,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(columns)+sizeof(rows)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

// Exact immutable dtype/precision variant of the same CPU Softmax producer.
extern "C" bool mlx_operation_event_cpu_typed_softmax_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype dtype,bool precise,size_t rank,size_t columns,size_t rows,bool tracer) {
  if(!out||dtype<MLX_BOOL||dtype>MLX_COMPLEX64)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::softmax_typed_eval_layout(mlx_dtype_to_cpp(dtype),precise,
      rank,columns,rows,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(dtype)+sizeof(precise)+
      sizeof(rank)+sizeof(columns)+sizeof(rows)+sizeof(tracer)+sizeof(size_t)+sizeof(mlx::core::Dtype);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

#include "mlx/backend/cpu/greedy_storage.h"
extern "C" bool mlx_operation_event_cpu_greedy_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t columns,size_t rows,bool reduction,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!(reduction?mlx::core::cpu::arg_reduce_eval_layout(rank,columns,rows,tracer,native)
      :mlx::core::cpu::squeeze_eval_layout(rank,tracer,native)))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(columns)+sizeof(rows)+
      sizeof(reduction)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_typed_arg_reduce_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype dtype,size_t rank,size_t columns,size_t rows,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::typed_arg_reduce_eval_layout(mlx_dtype_to_cpp(dtype),
      rank,columns,rows,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(dtype)+
      sizeof(rank)+sizeof(columns)+sizeof(rows)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

#include "mlx/backend/cpu/quantize_storage.h"
extern "C" bool mlx_operation_event_cpu_affine_quantize_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype dtype,size_t rank,size_t rows,size_t columns,int group_size,int bits,bool copy,bool tracer) {
  if(!out||dtype<MLX_BOOL||dtype>MLX_COMPLEX64)return false;
  mlx::core::cpu::QuantizeEvalStorage native;
  if(!mlx::core::cpu::affine_quantize_eval_layout(mlx_dtype_to_cpp(dtype),rank,rows,columns,
      group_size,bits,copy,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(dtype)+sizeof(rank)+
      sizeof(rows)+sizeof(columns)+sizeof(group_size)+sizeof(bits)+sizeof(copy)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

#include "mlx/backend/cpu/reshape_storage.h"
extern "C" bool mlx_operation_event_cpu_reshape_alias_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t output_rank,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::reshape_alias_eval_layout(rank,output_rank,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(output_rank)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_reshape_copy_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t output_rank,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::reshape_copy_eval_layout(rank,output_rank,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(output_rank)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

#include "mlx/backend/cpu/random_storage.h"
extern "C" bool mlx_operation_event_cpu_reshape_eval_layout(mlx_cpu_copy_eval_layout* out,
    const int* source_shape,const int64_t* source_strides,size_t rank,
    const int* output_shape,size_t output_rank,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::reshape_eval_layout(source_shape,source_strides,rank,
      output_shape,output_rank,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(source_shape)+
      sizeof(source_strides)+sizeof(rank)+sizeof(output_shape)+sizeof(output_rank)+sizeof(tracer);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_random_bits_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t elements,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::random_bits_eval_layout(rank,elements,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(elements)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

#include "mlx/backend/cpu/matmul_storage.h"
extern "C" bool mlx_operation_event_cpu_bf16_matmul_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t m,size_t n,size_t k,size_t batches,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::bf16_matmul_eval_layout(rank,m,n,k,batches,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(m)+sizeof(n)+
      sizeof(k)+sizeof(batches)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_bf16_matmul_copy_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t m,size_t n,size_t k,size_t batches,size_t copies,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::bf16_matmul_copy_eval_layout(rank,m,n,k,batches,copies,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(m)+sizeof(n)+
      sizeof(k)+sizeof(batches)+sizeof(copies)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_f16_matmul_copy_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t m,size_t n,size_t k,size_t batches,size_t copies,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::float16_matmul_copy_eval_layout(rank,m,n,k,batches,copies,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(m)+sizeof(n)+
      sizeof(k)+sizeof(batches)+sizeof(copies)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_tiled_matmul_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t m,size_t n,size_t k,size_t batches,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::tiled_matmul_eval_layout(rank,m,n,k,batches,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(m)+sizeof(n)+
      sizeof(k)+sizeof(batches)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_tiled_matmul_copy_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t m,size_t n,size_t k,size_t batches,size_t copies,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::tiled_matmul_copy_eval_layout(rank,m,n,k,batches,copies,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(m)+sizeof(n)+
      sizeof(k)+sizeof(batches)+sizeof(copies)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

#include "mlx/backend/cpu/alias_storage.h"
extern "C" bool mlx_operation_event_cpu_alias_eval_layout(mlx_cpu_copy_eval_layout* out,
    unsigned operation,size_t rank,size_t output_rank,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::alias_eval_layout(static_cast<mlx::core::cpu::AliasOperation>(operation),
      rank,output_rank,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(operation)+
      sizeof(rank)+sizeof(output_rank)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

#include "mlx/backend/cpu/gather_storage.h"
extern "C" bool mlx_operation_event_cpu_gather_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype source,mlx_dtype index,size_t source_rank,size_t index_rank,
    size_t source_elements,size_t index_elements,size_t slice_elements,bool tracer) {
  using namespace mlx::core;
  if(!out||source<MLX_BOOL||source>MLX_COMPLEX64||index<MLX_BOOL||index>MLX_COMPLEX64)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::gather_eval_layout(mlx_dtype_to_cpp(source),mlx_dtype_to_cpp(index),
      source_rank,index_rank,source_elements,index_elements,slice_elements,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(source)+sizeof(index)+
      sizeof(source_rank)+sizeof(index_rank)+sizeof(source_elements)+sizeof(index_elements)+
      sizeof(slice_elements)+sizeof(tracer)+sizeof(size_t)+sizeof(Dtype)*2;
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

#include "mlx/backend/cpu/reduction_storage.h"
extern "C" bool mlx_operation_event_cpu_reduction_eval_layout(mlx_cpu_copy_eval_layout* out,
    unsigned operation,size_t rank,size_t width,size_t rows,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::reduction_eval_layout(static_cast<mlx::core::cpu::ReductionEvalKind>(operation),
      rank,width,rows,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(operation)+
      sizeof(rank)+sizeof(width)+sizeof(rows)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

#include "mlx/backend/cpu/selection_storage.h"
extern "C" bool mlx_operation_event_cpu_selection_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype dtype,size_t rank,size_t elements,bool full,bool tracer) {
  using namespace mlx::core;
  if(!out||dtype<MLX_BOOL||dtype>MLX_COMPLEX64)return false;
  cpu::CopyEvalStorage native;
  if(!(full?cpu::scalar_full_eval_layout(mlx_dtype_to_cpp(dtype),rank,elements,tracer,native):
      cpu::select_eval_layout(mlx_dtype_to_cpp(dtype),rank,elements,tracer,native)))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(dtype)+sizeof(rank)+
      sizeof(elements)+sizeof(full)+sizeof(tracer)+sizeof(size_t)+sizeof(Dtype);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_select_broadcast_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t elements,bool tracer) {
  using namespace mlx::core;
  if(!out)return false;
  cpu::CopyEvalStorage native;
  if(!cpu::select_broadcast_eval_layout(rank,elements,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+
      sizeof(elements)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_typed_select_broadcast_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype dtype,size_t rank,size_t elements,bool tracer) {
  using namespace mlx::core;
  if(!out || !(dtype==MLX_FLOAT32||dtype==MLX_FLOAT16||dtype==MLX_BFLOAT16||dtype==MLX_INT32))return false;
  cpu::CopyEvalStorage native;
  if(!cpu::typed_select_broadcast_eval_layout(mlx_dtype_to_cpp(dtype),rank,elements,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(dtype)+sizeof(Dtype)+sizeof(rank)+
      sizeof(elements)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_row_full_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t width,size_t rows,bool tracer) {
  using namespace mlx::core;
  if(!out)return false;
  cpu::CopyEvalStorage native;
  if(!cpu::row_full_eval_layout(float32,rank,width,rows,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(width)+
      sizeof(rows)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

#include "mlx/fast.h"
#include <tuple>
extern "C" bool mlx_operation_event_cpu_rms_fallback_control_bytes(
    size_t* out, mlx_dtype dtype, size_t rank, size_t width, size_t rows) {
  if (!out || (dtype != MLX_FLOAT32 && dtype != MLX_FLOAT16 && dtype != MLX_BFLOAT16)) return false;
  size_t native = 0;
  if (!mlx::core::fast::cpu_rms_fallback_control_bytes(
      mlx_dtype_to_cpp(dtype), rank, width, rows, native)) return false;
  const size_t frames = sizeof(native) +
      sizeof(std::tuple<size_t*, mlx_dtype, size_t, size_t, size_t>);
  if (native > SIZE_MAX - frames) return false;
  *out = native + frames;
  return true;
}

#include "mlx/backend/cpu/concatenate_storage.h"
extern "C" bool mlx_operation_event_cpu_concatenate_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype dtype,size_t rank,size_t left,size_t right,bool tracer) {
  // Validate the conversion-table index; the native source owns supported types.
  if(!out||dtype<MLX_BOOL||dtype>MLX_COMPLEX64)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::concatenate_eval_layout(mlx_dtype_to_cpp(dtype),rank,left,right,tracer,native))return false;
  const size_t frames=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(dtype)+sizeof(rank)+sizeof(left)+sizeof(right)+sizeof(tracer);
  if(native.named_control_bytes>SIZE_MAX-frames)return false;
  *out={native.allocation_extents,native.worker_graph_extents,native.backing_births,native.named_control_bytes+frames,0};
  return true;
}

extern "C" bool mlx_operation_event_cpu_concatenate_many_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype dtype,size_t rank,size_t inputs,size_t elements,bool tracer) {
  if(!out||dtype<MLX_BOOL||dtype>MLX_COMPLEX64)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::concatenate_many_eval_layout(mlx_dtype_to_cpp(dtype),rank,inputs,elements,tracer,native))return false;
  const size_t frames=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(dtype)+sizeof(rank)+sizeof(inputs)+sizeof(elements)+sizeof(tracer);
  if(native.named_control_bytes>SIZE_MAX-frames)return false;
  *out={native.allocation_extents,native.worker_graph_extents,native.backing_births,native.named_control_bytes+frames,0};
  return true;
}

#include "mlx/backend/cpu/arange_storage.h"
extern "C" bool mlx_operation_event_cpu_arange_float_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t elements,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::arange_float_eval_layout(elements,tracer,native))return false;
  const size_t frames=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(elements)+sizeof(tracer);
  if(native.named_control_bytes>SIZE_MAX-frames)return false;
  *out={native.allocation_extents,native.worker_graph_extents,native.backing_births,native.named_control_bytes+frames,0};
  return true;
}
extern "C" bool mlx_operation_event_cpu_arange_int_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype dtype,size_t elements,bool tracer) {
  if(!out || (dtype != MLX_INT32 && dtype != MLX_UINT32))return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::arange_int_eval_layout(mlx_dtype_to_cpp(dtype),elements,tracer,native))return false;
  const size_t frames=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(dtype)+sizeof(elements)+sizeof(tracer);
  if(native.named_control_bytes>SIZE_MAX-frames)return false;
  *out={native.allocation_extents,native.worker_graph_extents,native.backing_births,native.named_control_bytes+frames,0};
  return true;
}
#include "mlx/backend/cpu/gather_mm_storage.h"
extern "C" bool mlx_operation_event_cpu_tiled_gather_mm_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t lhs_rank,size_t rhs_rank,size_t index_rank,size_t m,size_t n,size_t k,
    size_t batches,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::tiled_gather_mm_eval_layout(lhs_rank,rhs_rank,index_rank,m,n,k,batches,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(lhs_rank)+sizeof(rhs_rank)+
      sizeof(index_rank)+sizeof(m)+sizeof(n)+sizeof(k)+sizeof(batches)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}
extern "C" bool mlx_operation_event_cpu_rope_fallback_control_bytes(size_t* out,
    size_t rank,size_t dimensions,size_t elements) {
  if(!out)return false;
  size_t native=0;
  if(!mlx::core::fast::cpu_rope_fallback_control_bytes(rank,dimensions,elements,native))return false;
  const size_t frames=sizeof(native)+sizeof(out)+sizeof(rank)+sizeof(dimensions)+sizeof(elements);
  if(native>SIZE_MAX-frames)return false;*out=native+frames;return true;
}

extern "C" bool mlx_operation_event_cpu_sdpa_fallback_control_bytes(size_t* out,
    size_t rank,size_t queries,size_t keys,size_t values,size_t scores) {
  if(!out)return false;
  size_t native=0;
  if(!mlx::core::fast::cpu_sdpa_fallback_control_bytes(rank,queries,keys,values,scores,native))return false;
  const size_t frames=sizeof(native)+sizeof(out)+sizeof(rank)+sizeof(queries)+sizeof(keys)+sizeof(values)+sizeof(scores);
  if(native>SIZE_MAX-frames)return false;*out=native+frames;return true;
}

extern "C" bool mlx_operation_event_cpu_sdpa_array_mask_control_bytes(size_t* out,
    size_t rank,size_t queries,size_t keys,size_t values,size_t scores) {
  if(!out)return false;
  size_t native=0;
  if(!mlx::core::fast::cpu_sdpa_array_mask_control_bytes(rank,queries,keys,values,scores,native))return false;
  const size_t frames=sizeof(native)+sizeof(out)+sizeof(rank)+sizeof(queries)+sizeof(keys)+sizeof(values)+sizeof(scores);
  if(native>SIZE_MAX-frames)return false;*out=native+frames;return true;
}

#include "mlx/backend/cpu/scatter_storage.h"
extern "C" bool mlx_operation_event_cpu_scatter_axis_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype index,size_t rank,size_t output_elements,size_t update_elements,bool tracer) {
  using namespace mlx::core;
  if(!out||(index!=MLX_INT32&&index!=MLX_UINT32))return false;
  cpu::CopyEvalStorage native;
  if(!cpu::scatter_axis_eval_layout(mlx_dtype_to_cpp(index),rank,output_elements,update_elements,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(index)+sizeof(rank)+
      sizeof(output_elements)+sizeof(update_elements)+sizeof(tracer)+sizeof(size_t)+sizeof(Dtype);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_scatter_add_rows_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype index,size_t output_elements,size_t update_elements,bool tracer) {
  using namespace mlx::core;
  if(!out||(index!=MLX_INT32&&index!=MLX_UINT32))return false;
  cpu::CopyEvalStorage native;
  if(!cpu::scatter_add_rows_eval_layout(mlx_dtype_to_cpp(index),output_elements,update_elements,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(index)+
      sizeof(output_elements)+sizeof(update_elements)+sizeof(tracer)+sizeof(size_t)+sizeof(Dtype);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

#include "mlx/backend/cpu/reduction_storage.h"


extern "C" bool mlx_operation_event_cpu_partition_row_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t elements,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::partition_row_eval_layout(rank,elements,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(elements)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_scan_sum_row_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t elements,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::scan_sum_row_eval_layout(rank,elements,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(elements)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_maximum_row_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t elements,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::reduction_eval_layout(mlx::core::cpu::ReductionEvalKind::Float32FlatMaximum,rank,elements,1,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(elements)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_gather_axis_row_eval_layout(mlx_cpu_copy_eval_layout* out,
    size_t rank,size_t elements,bool tracer) {
  if(!out)return false;
  mlx::core::cpu::CopyEvalStorage native;
  if(!mlx::core::cpu::gather_axis_row_eval_layout(rank,elements,tracer,native))return false;
  const size_t controls=sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(rank)+sizeof(elements)+sizeof(tracer)+sizeof(size_t);
  if(native.named_control_bytes>SIZE_MAX-controls)return false;
  const mlx_cpu_copy_eval_layout value{native.allocation_extents,native.worker_graph_extents,
      native.backing_births,native.named_control_bytes+controls,0};
  *out=value;return true;
}

extern "C" bool mlx_operation_event_cpu_scatter_eval_layout(mlx_cpu_copy_eval_layout* out,
    mlx_dtype source,mlx_dtype index,size_t rank,size_t output_elements,size_t update_elements,bool tracer) {
  using namespace mlx::core;
  if (!out || index != MLX_INT32 || (source!=MLX_FLOAT32&&source!=MLX_INT32)) return false;
  cpu::CopyEvalStorage native;
  if (!cpu::scatter_eval_layout(mlx_dtype_to_cpp(source),mlx_dtype_to_cpp(index),rank,
      output_elements,update_elements,tracer,native)) return false;
  const size_t controls = sizeof(native)+sizeof(*out)+sizeof(out)+sizeof(source)+sizeof(index)+sizeof(rank)+
      sizeof(output_elements)+sizeof(update_elements)+sizeof(tracer)+sizeof(size_t)+sizeof(Dtype)*2;
  if (native.named_control_bytes > SIZE_MAX-controls) return false;
  *out = {native.allocation_extents,native.worker_graph_extents,native.backing_births,native.named_control_bytes+controls,0};
  return true;
}

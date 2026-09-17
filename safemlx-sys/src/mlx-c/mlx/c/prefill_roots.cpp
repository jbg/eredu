#include <thread>
#include "mlx/c/prefill_roots.h"
#include "mlx/c/error.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include "mlx/completion.h"
#include "mlx/scoped_evaluation.h"
#include "mlx/submission.h"
#include "mlx/graph_quota.h"
#include "mlx/transforms.h"
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>

using namespace mlx::core;
namespace {
using submission::GraphQuota;
using Allocator = submission::GraphAllocator<array>;
enum class State : unsigned { filling, submitted, failed, validated };
struct Roots final {
  explicit Roots(size_t limit, GraphQuota* graph) noexcept
      : retained(Allocator(graph)), submitted(Allocator(graph)), capacity(limit) {}
  // Both actual vectors have the same original domain. Submitted moves into
  // evaluation; retained survives that move and the final completion wait.
  ArrayVector retained;
  ArrayVector submitted;
  Completion completion;
  size_t capacity;
  State state{State::filling};
  FailureCarrierRef failure;
  uint64_t scope_identity{0};
};
static_assert(std::is_nothrow_copy_constructible_v<array>);
static_assert(std::is_nothrow_move_assignable_v<Completion>);

struct PreparedBlocks final {
  GraphQuota* quota;
  ~PreparedBlocks() { if (quota) quota->finish_blocks(); }
};
// These are actual named constructor locals, not a guessed control envelope.
// Native eval/event/task/error controls remain separate, explicitly unbounded
// here. Completion's preallocated destination is already inside Roots.
struct Construction final {
  mlx_prefill_roots_layout selected{};
  std::unique_ptr<Roots> roots;
  GraphQuota::Request requests[2]{};
  PreparedBlocks prepared{nullptr};
  GraphQuota* quota{nullptr};
  size_t payload{0};
};
bool layout(size_t capacity, mlx_prefill_roots_layout& out) noexcept {
  // std::vector also limits max_size by ptrdiff_t, not just size_t. Query its
  // actual empty explicit-null allocator representation without TLS access.
  const ArrayVector empty{Allocator(nullptr)};
  if (capacity > empty.max_size() ||
      capacity > std::numeric_limits<size_t>::max() / sizeof(array)) return false;
  const size_t payload = capacity * sizeof(array);
  size_t block = 0, descriptor_minimum = 0;
  if (!array::graph_descriptor_minimum_extent(descriptor_minimum)) return false;
  if (capacity && !GraphQuota::allocation_extent(payload, alignof(array), block)) return false;
  if (block > std::numeric_limits<size_t>::max() / 2) return false;
  out = {capacity, sizeof(Roots), alignof(Roots), 2 * block,
      sizeof(Construction) + sizeof(empty) + sizeof(mlx_prefill_roots_layout) +
      sizeof(size_t) * 4 + sizeof(GraphQuota*) + sizeof(mlx_prefill_roots*) +
      // Bound observation/submit locals, plus bind and final-free carrier moves.
      sizeof(submission::Scope*) + sizeof(mlx_submission_scope) +
      sizeof(submission::ScopedProgress) + sizeof(EventObservation) +
      sizeof(FailureCarrierRef) * 2 + sizeof(unsigned) * 2 +
      // Actual selected-completion arguments and named live locals: current,
      // owner, raw, scope, selected, ready, submitted and waited.
      sizeof(mlx_prefill_roots) + sizeof(mlx_stream) + sizeof(Roots*) +
      sizeof(submission::Scope*) * 2 + sizeof(mlx_submission_scope) + sizeof(Stream) +
      sizeof(ScopedEvaluation) + sizeof(submission::ScopedProgress) + sizeof(unsigned) +
      sizeof(const mlx_operation_eval_traversal_limits*) +
      sizeof(CompletionEvalTraversalLimits) + sizeof(CompletionEvalTraversalLayout),
      descriptor_minimum};
  return true;
}
Roots* get(mlx_prefill_roots roots) noexcept {
  return static_cast<Roots*>(roots.ctx);
}
}
extern "C" unsigned mlx_prefill_roots_layout_for(
    mlx_prefill_roots_layout* out, size_t capacity) {
  return out && layout(capacity, *out) ? 0 : 1;
}
extern "C" unsigned mlx_prefill_roots_new(
    mlx_prefill_roots* out, size_t capacity, mlx_submission_graph_quota graph) {
  Construction local;
  if (!out || out->ctx || !layout(capacity, local.selected)) return 1;
  local.quota = static_cast<GraphQuota*>(graph.ctx);
  // Null selects only the explicit ordinary path. No fallback after refusal.
  local.roots.reset(new (std::nothrow) Roots(capacity, local.quota));
  if (!local.roots) return 3;
  local.payload = capacity * sizeof(array);
  local.requests[0] = {local.payload, alignof(array)};
  local.requests[1] = {local.payload, alignof(array)};
  if (local.quota && capacity && !local.quota->prepare_blocks(local.requests, 2)) return 2;
  local.prepared.quota = capacity ? local.quota : nullptr;
  try {
    local.roots->retained.reserve(capacity);
    local.roots->submitted.reserve(capacity);
  } catch (const submission::GraphQuotaError&) {
    return 2;
  } catch (const std::bad_alloc&) {
    return 3;
  } catch (const std::length_error&) {
    return 1;
  }
  // The complete owner exists before any root or native producer can enter.
  out->ctx = local.roots.release();
  return 0;
}
extern "C" unsigned mlx_prefill_roots_append(mlx_prefill_roots roots, mlx_array value) {
  auto* owner = get(roots);
  if (!owner || !value.ctx) return 1;
  if (owner->state != State::filling) return 5;
  if (owner->retained.size() >= owner->capacity ||
      owner->submitted.size() >= owner->capacity) return 2;
  // Both capacity checks precede either copy. These copies cannot allocate,
  // destroy existing elements, call user hooks or export an owning C handle.
  const auto& source = mlx_array_get_(value);
  if (!source.id()) return 1;
  owner->retained.emplace_back(source);
  owner->submitted.emplace_back(source);
  return 0;
}
extern "C" unsigned mlx_prefill_roots_append_validation(mlx_prefill_roots roots, mlx_array value) {
  auto* owner = get(roots);
  if (!owner || !value.ctx) return 1;
  if (owner->state != State::filling) return 5;
  const auto* quota = owner->retained.get_allocator().resource();
  if (quota) {
    const auto& source = mlx_array_get_(value);
    if (!source.has_graph_descriptor(quota)) return 4;
    // The original bound counts distinct live descriptor blocks. This also
    // preserves validation semantics if a future private producer reuses one:
    // its messages are still checked individually after one root evaluation.
    for (const auto& retained : owner->retained) {
      if (retained.id() == source.id()) return 0;
    }
  }
  return mlx_prefill_roots_append(roots, value);
}
extern "C" unsigned mlx_prefill_roots_submit(mlx_prefill_roots roots) {
  auto* owner = get(roots);
  if (!owner) return 1;
  if (owner->state != State::filling) return 5;
  if (owner->submitted.get_allocator().resource() != submission::current_graph_quota()) return 4;
  // The attempt is consumed before evaluation can create partial native work.
  // The allocator association prevents GraphVector's move from copying roots
  // into another arena. Publication only moves into an existing Completion.
  owner->state = State::submitted;
  try {
    owner->completion = async_eval_with_completion(std::move(owner->submitted));
    return 0;
  } catch (const std::exception& error) {
    owner->state = State::failed;
    mlx_error("%s", error.what());
    return 7;
  }
}
extern "C" unsigned mlx_prefill_roots_wait(mlx_prefill_roots roots) {
  auto* owner = get(roots);
  if (!owner || owner->state == State::filling) return 1;
  if (owner->state == State::failed) return 5;
  try { owner->completion.wait(); return 0; }
  catch (const std::exception& error) { mlx_error("%s", error.what()); return 7; }
}
extern "C" unsigned mlx_prefill_roots_query(mlx_prefill_roots roots) {
  auto* owner = get(roots);
  if (!owner || owner->state == State::filling) return 1;
  if (owner->state == State::failed) return 5;
  try { return owner->completion.is_complete() ? 0 : 6; }
  catch (const std::exception& error) { mlx_error("%s", error.what()); return 7; }
}
extern "C" unsigned mlx_prefill_roots_validate(mlx_prefill_roots roots) {
  auto* owner = get(roots);
  if (!owner || owner->state == State::filling) return 1;
  if (owner->state == State::failed) return 5;
  try {
    if (!owner->completion.is_complete()) return 6;
    for (auto& root : owner->retained) {
      // A completed submission cannot leave an unscheduled root. Do not start
      // a replacement eval or allocate its initializer-list vector here.
      if (root.status() == array::Status::unscheduled) return 1;
      root.wait();
    }
    owner->state = State::validated;
    return 0;
  } catch (const std::exception& error) { mlx_error("%s", error.what()); return 7; }
}
extern "C" size_t mlx_prefill_roots_size(mlx_prefill_roots roots) {
  const auto* owner = get(roots);
  return owner ? owner->retained.size() : 0;
}
extern "C" void mlx_prefill_roots_free(mlx_prefill_roots roots) {
  auto* owner = get(roots);
  if (!owner) return;
  // The fixed root shell/vectors/Completion retire before the final carrier
  // may enqueue their shared original custody. Do not rely on member order.
  auto failure = std::move(owner->failure);
  delete owner;
}

namespace {
submission::Scope* bound_scope(Roots* roots, mlx_submission_scope raw) noexcept {
  auto* scope = static_cast<submission::Scope*>(raw.ctx);
  if (!roots || !roots->failure || !scope || !scope->on_owner_thread() ||
      !scope->scoped_observation_only() || !roots->scope_identity ||
      roots->scope_identity != scope->identity() ||
      roots->failure.get() != scope->failure_owner().get()) return nullptr;
  return scope;
}
unsigned progress_status(submission::ScopedProgress status) noexcept {
  return static_cast<unsigned>(evaluation_progress(status));
}
}
extern "C" unsigned mlx_prefill_roots_new_owned(
    mlx_prefill_roots* out, size_t capacity, mlx_submission_graph_quota graph,
    mlx_prefill_failure raw_failure) {
  auto* failure = static_cast<FailureCarrier*>(raw_failure.ctx);
  if (!failure || failure->scope_identity()) return 4;
  const auto status = mlx_prefill_roots_new(out, capacity, graph);
  if (status) return status;
  get(*out)->failure = failure->clone();
  return 0;
}
extern "C" unsigned mlx_prefill_roots_bind(
    mlx_prefill_roots roots, mlx_submission_scope raw) {
  auto* owner = get(roots);
  auto* scope = static_cast<submission::Scope*>(raw.ctx);
  if (!owner || !owner->failure || owner->scope_identity || !scope ||
      submission::current_scope() != scope ||
      owner->retained.get_allocator().resource() != scope->graph_quota()) return 4;
  if (!scope->bind_failure(owner->failure)) return 4;
  owner->scope_identity = scope->identity();
  return 0;
}
extern "C" unsigned mlx_prefill_roots_submit_scoped(
    mlx_prefill_roots roots, mlx_submission_scope raw) {
  auto* owner = get(roots);
  auto* scope = bound_scope(owner, raw);
  if (!scope || submission::current_scope() != scope) return 4;
  if (owner->state != State::filling) return 5;
  if (owner->submitted.get_allocator().resource() != submission::current_graph_quota()) return 4;
  owner->state = State::submitted;
  try {
    const auto result = async_eval_with_completion_scoped(
        std::move(owner->submitted), *scope, owner->completion);
    if (result != submission::ScopedProgress::observed) owner->state = State::failed;
    return progress_status(result);
  } catch (...) {
    owner->state = State::failed;
    owner->failure.get()->capture_current_exception();
    return 7;
  }
}
extern "C" unsigned mlx_prefill_roots_query_scoped(
    mlx_prefill_roots roots, mlx_submission_scope raw) {
  auto* owner = get(roots);auto* scope = bound_scope(owner, raw);
  if (!scope || owner->state == State::filling) return 4;
  if (owner->state == State::failed) return 5;
  return static_cast<unsigned>(query_scoped_completion(*scope, owner->completion));
}
extern "C" unsigned mlx_prefill_roots_wait_scoped(
    mlx_prefill_roots roots, mlx_submission_scope scope) {
  for (;;) {
    const auto status = mlx_prefill_roots_query_scoped(roots, scope);
    // Busy, NeedsFundedProgress and Unobservable never spin/fall back globally.
    if (status != 6) return status;
    std::this_thread::yield();
  }
}
extern "C" unsigned mlx_prefill_roots_validate_scoped(
    mlx_prefill_roots roots, mlx_submission_scope scope) {
  const auto status = mlx_prefill_roots_query_scoped(roots, scope);
  if (status) return status;
  auto* owner = get(roots);
  auto* bound = bound_scope(owner, scope);
  for (auto& root : owner->retained) {
    const auto result = validate_scoped_array(root, *bound);
    if (result != ScopedEvaluation::complete) return static_cast<unsigned>(result);
  }
  owner->state = State::validated;
  return 0;
}

extern "C" unsigned mlx_prefill_roots_complete_current(mlx_prefill_roots roots) {
  auto* current = submission::current_scope();
  if (!current) return 4;
  const mlx_submission_scope scope{current};
  auto status = mlx_prefill_roots_submit_scoped(roots, scope);
  if (status) return status;
  status = mlx_prefill_roots_wait_scoped(roots, scope);
  return status ? status : mlx_prefill_roots_validate_scoped(roots, scope);
}

namespace {
unsigned complete_current_on_stream(
    mlx_prefill_roots roots, mlx_stream stream,
    const mlx_operation_eval_traversal_limits* limits) {
  auto* current = submission::current_scope();
  auto* owner = get(roots);
  const mlx_submission_scope raw{current};
  auto* scope = bound_scope(owner, raw);
  if (!scope || !stream.ctx) return 4;
  if (owner->state != State::filling) return 5;
  if (owner->submitted.get_allocator().resource() != submission::current_graph_quota()) return 4;
  const auto selected = mlx_stream_get_(stream);
  const auto ready = preflight_original_submission_stream(*scope, selected);
  if (ready != ScopedEvaluation::complete) return static_cast<unsigned>(ready);
  CompletionEvalTraversalLimits prepared{};
  if (limits) {
    prepared = {limits->root_count, limits->array_nodes, limits->tape_entries,
        limits->input_edges, limits->output_slots, limits->stream_count,
        limits->capture_slots};
    CompletionEvalTraversalLayout layout;
    if (prepared.root_count < owner->submitted.size()) return 1;
    // The producer supplies a maximum because repeated validation descriptors
    // are deduplicated by append. Tightening roots cannot enlarge any request.
    prepared.root_count = owner->submitted.size();
    if (!completion_eval_traversal_layout(prepared, layout)) return 1;
  }
  // Exact current owner, prepared stream and finite layout precede state
  // mutation. The same prepared root backing moves into the shared evaluator.
  owner->state = State::submitted;
  try {
    const auto submitted = limits
        ? async_eval_with_completion_scoped_on_stream_prepared(
              std::move(owner->submitted), *scope, owner->completion, selected, prepared)
        : async_eval_with_completion_scoped_on_stream(
              std::move(owner->submitted), *scope, owner->completion, selected);
    if (submitted != submission::ScopedProgress::observed) {
      owner->state = State::failed;
      return progress_status(submitted);
    }
  } catch (...) {
    owner->state = State::failed;
    owner->failure.get()->capture_current_exception();
    return 7;
  }
  const auto waited = mlx_prefill_roots_wait_scoped(roots, raw);
  return waited ? waited : mlx_prefill_roots_validate_scoped(roots, raw);
}

} // namespace
extern "C" unsigned mlx_prefill_roots_complete_current_on_stream(
    mlx_prefill_roots roots, mlx_stream stream) {
  return complete_current_on_stream(roots, stream, nullptr);
}
extern "C" unsigned mlx_prefill_roots_complete_current_on_stream_prepared(
    mlx_prefill_roots roots, mlx_stream stream,
    const mlx_operation_eval_traversal_limits* limits) {
  return limits ? complete_current_on_stream(roots, stream, limits) : 1;
}

extern "C" unsigned mlx_array_eval_scoped_failure(mlx_prefill_failure* error) {
  if (!error || error->ctx) return 1;
  auto* scope = original_evaluation_scope();
  if (!scope || !scope->on_owner_thread() || !scope->failure_owner()) return 4;
  if (scope->failure_owner().get()->scope_identity() != scope->identity()) return 4;
  error->ctx = FailureCarrier::into_raw(scope->failure_owner().get()->clone());
  return 0;
}
extern "C" unsigned mlx_array_eval_scoped(mlx_array value, mlx_prefill_failure* error) {
  const auto status = mlx_array_eval_scoped_failure(error);
  if (status) return status;
  if (!value.ctx) return 1;
  auto* scope = original_evaluation_scope();
  return static_cast<unsigned>(evaluate_scoped_array(mlx_array_get_(value), *scope));
}
extern "C" size_t mlx_array_eval_scoped_control_bytes(void) {
  // Actual synchronous companion and shared observation locals. The root
  // vector backing is supplied by the same enforced Graph arena, not this sum.
  return sizeof(Completion) + sizeof(ArrayVector) + sizeof(submission::NativeControlBinding) * 2 +
      sizeof(submission::Scope*) * 3 + sizeof(FailureCarrier*) + sizeof(FailureCarrierRef) +
      sizeof(ScopedEvaluationError) +
      sizeof(ScopedEvaluation) * 4 + sizeof(submission::ScopedProgress) +
      sizeof(EventObservation) + sizeof(mlx_prefill_failure*) + sizeof(mlx_array) +
      sizeof(unsigned) + sizeof(bool);
}

#include "mlx/backend/common/pipeline_cache.h"
#include "mlx/c/submission.h"
#include "mlx/c/prefill_failure.h"

#include <stdexcept>

#include "mlx/c/error.h"
#include "mlx/c/private/stream.h"
#include "mlx/c/private/array.h"
#include <array>
#include "mlx/submission.h"
#include "mlx/event.h"
#include "mlx/scoped_evaluation.h"

namespace {
using mlx::core::submission::Activity;
using mlx::core::submission::Scope;

Scope* get_scope(mlx_submission_scope scope) noexcept {
  auto* value = static_cast<Scope*>(scope.ctx);
  return value && value->on_owner_thread() ? value : nullptr;
}

void write_status(mlx_submission_status* out, Scope* scope) noexcept {
  auto status = scope->query();
  switch (status.activity) {
    case Activity::none:
      out->activity = MLX_SUBMISSION_ACTIVITY_NONE;
      break;
    case Activity::pending:
      out->activity = MLX_SUBMISSION_ACTIVITY_PENDING;
      break;
    case Activity::terminal:
      out->activity = MLX_SUBMISSION_ACTIVITY_TERMINAL;
      break;
  }
  out->failed = status.failed;
  out->blocked = status.blocked;
}
} // namespace

extern "C" unsigned mlx_submission_observer_current(mlx_submission_observer* out) {
  if (!out || out->ctx) return 1;
  auto* current = mlx::core::submission::current_scope();
  if (!current || !current->original_controls_required()) return 11;
  auto* retained = Scope::retain_current_original();
  if (!retained) return 4;
  out->ctx = retained;
  return 0;
}
extern "C" unsigned mlx_submission_observer_retain(mlx_submission_observer value) {
  auto* scope = static_cast<Scope*>(value.ctx);
  return scope && scope->retain_observer() ? 0 : 4;
}
extern "C" void mlx_submission_observer_release(mlx_submission_observer value) {
  if (auto* scope = static_cast<Scope*>(value.ctx)) scope->release();
}
extern "C" unsigned mlx_submission_observer_query(
    mlx_submission_status* out, mlx_submission_observer value) {
  auto* scope = static_cast<Scope*>(value.ctx);
  if (!out || !scope || !scope->valid_original_observer()) return 4;
  write_status(out, scope);
  return 0;
}
extern "C" unsigned mlx_submission_observer_progress(
    mlx_submission_status* out, mlx_submission_observer value) {
  auto* scope = static_cast<Scope*>(value.ctx);
  if (!out || !scope || !scope->valid_original_observer()) return 4;
  const auto result = scope->progress_scoped();
  write_status(out, scope);
  return static_cast<unsigned>(mlx::core::evaluation_progress(result));
}
extern "C" unsigned mlx_submission_observer_retire(mlx_submission_observer value) {
  auto* scope = static_cast<Scope*>(value.ctx);
  return scope ? static_cast<unsigned>(mlx::core::evaluation_progress(
      scope->retire_scoped_records())) : 4;
}
extern "C" unsigned mlx_submission_observer_failure(
    mlx_prefill_failure* out, mlx_submission_observer value) {
  auto* scope = static_cast<Scope*>(value.ctx);
  if (!out || out->ctx || !scope || !scope->valid_original_observer()) return 4;
  auto* failure = scope->failure_owner().get();
  out->ctx = mlx::core::FailureCarrier::into_raw(failure->clone());
  return 0;
}
extern "C" unsigned mlx_submission_observer_validate_array(
    mlx_submission_observer value, mlx_array array) {
  auto* scope = static_cast<Scope*>(value.ctx);
  if (!array.ctx || !scope || !scope->valid_original_observer()) return 4;
  // Health is checked before the existing completed-only kernel can detach an
  // event. This retained observer remains valid without current-role authority.
  if (scope->failure_owner().get()->borrow()) return 7;
  if (scope->failure_owner().get()->pending_publication()) return 10;
  if (scope->query().failed) return 9;
  return static_cast<unsigned>(mlx::core::validate_scoped_array(
      mlx_array_get_(array), *scope));
}
extern "C" size_t mlx_submission_observer_control_bytes(void) {
  using namespace mlx::core;
  return submission::scoped_observer_control_bytes() +
      sizeof(mlx_submission_observer) + sizeof(mlx_submission_status) +
      sizeof(submission::NativeControlBinding) + sizeof(submission::Status) +
      sizeof(submission::ScopedProgress) + sizeof(submission::ScopedRetirement) +
      sizeof(Scope*) * 3 + sizeof(FailureCarrier*) + sizeof(unsigned) +
      sizeof(mlx_array) + sizeof(ScopedEvaluation);
}

extern "C" int mlx_submission_scope_new(mlx_submission_scope* out) {
  if (!out || out->ctx) {
    return 1;
  }
  try {
    out->ctx = new Scope;
    return 0;
  } catch (...) {
    return 1;
  }
}

extern "C" int mlx_submission_scope_new_retaining(
    mlx_submission_scope* out, void* owner, void (*retire)(void*)) {
  if (!out || out->ctx || !owner || !retire) {
    return 1;
  }
  try {
    out->ctx = new Scope(owner, retire);
    return 0;
  } catch (...) {
    // Scope arms custody only after its final fallible initialization.
    return 1;
  }
}

namespace {
struct ScopedPhysicalObserver {
  std::atomic<size_t> references{1};
  void* context;
  mlx_physical_backing_observer callback;
  void (*release)(void*);
  bool retain_in_cache;
  bool reuse_cached_backing;
};
bool observe_scoped_backing(void* opaque, uint64_t identity, size_t capacity,
    size_t controls, mlx::core::allocator::MemoryPlacement placement,
    void** owner, void (**release)(void*), mlx::core::allocator::PhysicalBackingPublish* publish) {
  const auto* observer = static_cast<const ScopedPhysicalObserver*>(opaque);
  return observer->callback(observer->context, identity, capacity, controls,
      {placement.kind, placement.device, placement.device_count}, owner, release, publish);
}
void release_scoped_backing(void* opaque) {
  auto* value = static_cast<ScopedPhysicalObserver*>(opaque);
  if (value->references.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
  auto* context = value->context;
  const auto release = value->release;
  delete value;
  release(context);
}
}
extern "C" size_t mlx_submission_scope_physical_observer_control_bytes(void) {
  return sizeof(ScopedPhysicalObserver);
}
static int make_scoped_physical_observer(mlx_scoped_physical_observer* out,
    void* context, mlx_physical_backing_observer callback, void (*release)(void*),
    bool retain_in_cache, bool reuse_cached_backing) {
  if (!out || out->ctx || !context || !callback || !release) return 1;
  try {
    out->ctx = new ScopedPhysicalObserver{{1}, context, callback, release, retain_in_cache, reuse_cached_backing};
    return 0;
  } catch (...) { return 1; }
}
extern "C" int mlx_scoped_physical_observer_new(mlx_scoped_physical_observer* out,
    void* context, mlx_physical_backing_observer callback, void (*release)(void*)) {
  return make_scoped_physical_observer(out, context, callback, release, true, true);
}
extern "C" int mlx_scoped_physical_observer_new_uncached(mlx_scoped_physical_observer* out,
    void* context, mlx_physical_backing_observer callback, void (*release)(void*)) {
  return make_scoped_physical_observer(out, context, callback, release, false, true);
}
extern "C" int mlx_scoped_physical_observer_new_fresh(mlx_scoped_physical_observer* out,
    void* context, mlx_physical_backing_observer callback, void (*release)(void*)) {
  return make_scoped_physical_observer(out, context, callback, release, false, false);
}
extern "C" void mlx_scoped_physical_observer_retain(mlx_scoped_physical_observer value) {
  if (auto* observer = static_cast<ScopedPhysicalObserver*>(value.ctx))
    observer->references.fetch_add(1, std::memory_order_relaxed);
}
extern "C" void mlx_scoped_physical_observer_free(mlx_scoped_physical_observer value) {
  if (value.ctx) release_scoped_backing(value.ctx);
}
extern "C" bool mlx_scoped_physical_observer_is_current(mlx_scoped_physical_observer value) {
  const auto current = mlx::core::submission::current_physical_observer();
  return value.ctx && current.context == value.ctx &&
      current.callback == observe_scoped_backing;
}
extern "C" bool mlx_scoped_physical_observer_has_current(void) {
  return mlx::core::submission::current_physical_observer().callback != nullptr;
}
extern "C" int mlx_submission_scope_bind_physical_observer(
    mlx_submission_scope scope, mlx_scoped_physical_observer value) {
  auto* native = get_scope(scope);
  if (!native || !value.ctx) return 1;
  if (!native->bind_physical_observer(value.ctx, observe_scoped_backing, release_scoped_backing,
      static_cast<ScopedPhysicalObserver*>(value.ctx)->retain_in_cache,
      static_cast<ScopedPhysicalObserver*>(value.ctx)->reuse_cached_backing)) return 1;
  mlx_scoped_physical_observer_retain(value);
  return 0;
}

extern "C" int mlx_submission_scope_seal(mlx_submission_scope scope) {
  if (auto* value = get_scope(scope)) {
    value->seal();
    return 0;
  }
  return 1;
}

extern "C" int mlx_submission_scope_query(
    mlx_submission_status* out, mlx_submission_scope scope) {
  if (auto* value = get_scope(scope); value && out) {
    write_status(out, value);
    return 0;
  }
  return 1;
}

extern "C" int mlx_submission_scope_query_records(
    mlx_submission_record_status* out, mlx_submission_scope scope) {
  if (auto* value = get_scope(scope); value && out) {
    const auto status = value->query_records();
    *out = {status.pending, status.failed, status.blocked};
    return 0;
  }
  return 1;
}

extern "C" size_t mlx_submission_scope_control_bytes(void) {
  return sizeof(Scope);
}

extern "C" size_t mlx_submission_scope_retirement_control_bytes(void) {
  return mlx::core::submission::scope_retirement_control_bytes();
}

extern "C" int mlx_submission_scope_progress(
    mlx_submission_status* out, mlx_submission_scope scope) {
  if (auto* value = get_scope(scope); value && out) {
    if (value->scoped_observation_only()) {
      const auto result = value->progress_scoped();
      write_status(out, value);
      if (result != mlx::core::submission::ScopedProgress::observed) {
        // Legacy status consumers (including Recovery finish/drop/orphans)
        // must return promptly, retaining the same owner. This is not failure
        // or terminal evidence inferred from a polling refusal.
        out->activity = MLX_SUBMISSION_ACTIVITY_PENDING;
        out->blocked = true;
      }
    } else {
      value->progress();
      write_status(out, value);
    }
    return 0;
  }
  return 1;
}

extern "C" unsigned mlx_submission_scope_enable_scoped(mlx_submission_scope scope) {
  auto* value = get_scope(scope);
  return value && value->enable_scoped_observation() ? 0 : 2;
}
extern "C" unsigned mlx_submission_scope_progress_scoped(
    mlx_submission_status* out, mlx_submission_scope scope) {
  using mlx::core::submission::ScopedProgress;
  auto* value = get_scope(scope);
  if (!out || !value) return 2;
  const auto result = value->progress_scoped();
  write_status(out, value);
  switch (result) {
    case ScopedProgress::observed: return 0;
    case ScopedProgress::funded_progress: return 1;
    case ScopedProgress::busy: return 3;
    default: return 2;
  }
}

extern "C" int mlx_submission_scope_free(mlx_submission_scope scope) {
  if (auto* value = get_scope(scope)) {
    value->seal();
    value->release();
    return 0;
  }
  return 1;
}

extern "C" int mlx_submission_retire_completed(mlx_submission_retirement* out) {
  try {
    if (!out) {
      throw std::invalid_argument(
          "[mlx_submission_retire_completed] Requires an output pointer.");
    }
    const auto pass = mlx::core::submission::try_retire_records();
    *out = pass == mlx::core::submission::RetirementPass::complete_snapshot
        ? MLX_SUBMISSION_RETIREMENT_COMPLETE_SNAPSHOT
        : MLX_SUBMISSION_RETIREMENT_BUSY;
    return 0;
  } catch (const std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
}

extern "C" int mlx_submission_record_quota_minimum_capacity(size_t* out) {
  return out && mlx::core::submission::RecordQuota::minimum_capacity(*out) ? 0 : 1;
}
extern "C" int mlx_submission_record_quota_fresh_capacity(size_t* out, size_t extents) {
  return out && mlx::core::submission::RecordQuota::fresh_capacity_for_extents(extents, *out) ? 0 : 1;
}
extern "C" int mlx_submission_record_quota_allocation_extent(
    size_t* out, size_t requested, size_t alignment) {
  return out && mlx::core::submission::RecordQuota::allocation_extent(
      requested, alignment, *out) ? 0 : 1;
}
extern "C" int mlx_submission_record_quota_layout_for(
    mlx_submission_record_quota_layout* out, size_t capacity) {
  if (!out) return 1;
  mlx::core::submission::RecordQuotaLayout layout;
  if (!mlx::core::submission::RecordQuota::layout(capacity, layout)) return 1;
  *out = {layout.capacity, layout.allocation_bytes, layout.alignment, layout.retirement_controls};
  return 0;
}
extern "C" int mlx_submission_record_quota_new_retaining(
    mlx_submission_record_quota* out, size_t capacity, void* owner, void (*retire)(void*)) {
  if (!out || out->ctx || !owner || !retire) return 1;
  try {
    out->ctx = mlx::core::submission::RecordQuota::create(capacity, owner, retire);
    return 0;
  } catch (...) { return 1; }
}
extern "C" void mlx_submission_record_quota_retain(mlx_submission_record_quota quota) {
  if (quota.ctx) static_cast<mlx::core::submission::RecordQuota*>(quota.ctx)->retain();
}
extern "C" void mlx_submission_record_quota_release(mlx_submission_record_quota quota) {
  if (quota.ctx) static_cast<mlx::core::submission::RecordQuota*>(quota.ctx)->release();
}
extern "C" size_t mlx_submission_record_quota_occupied(mlx_submission_record_quota quota) {
  return quota.ctx ? static_cast<mlx::core::submission::RecordQuota*>(quota.ctx)->occupied_bytes() : 0;
}
extern "C" int mlx_submission_scope_new_retaining_with_quota(
    mlx_submission_scope* out, void* owner, void (*retire)(void*), mlx_submission_record_quota quota) {
  if (!out || out->ctx || !owner || !retire) return 1;
  try {
    out->ctx = new Scope(owner, retire, static_cast<mlx::core::submission::RecordQuota*>(quota.ctx));
    return 0;
  } catch (...) { return 1; }
}

extern "C" int mlx_submission_graph_quota_population_extent(
    size_t* out, size_t requested_bytes, size_t nonempty_attempts) {
  return out && mlx::core::submission::GraphQuota::allocation_population_extent(
      requested_bytes, nonempty_attempts, *out) ? 0 : 1;
}
extern "C" int mlx_submission_graph_quota_fresh_capacity(size_t* out, size_t extents) {
  return out && mlx::core::submission::GraphQuota::fresh_capacity_for_extents(extents, *out) ? 0 : 1;
}
extern "C" int mlx_submission_graph_quota_layout_for(
    mlx_submission_graph_quota_layout* out, size_t capacity) {
  if (!out) return 1;
  mlx::core::submission::GraphQuotaLayout layout;
  if (!mlx::core::submission::GraphQuota::layout(capacity, layout)) return 1;
  *out = {layout.capacity, layout.allocation_bytes, layout.alignment, layout.retirement_controls};
  return 0;
}
extern "C" int mlx_submission_graph_quota_new_retaining(
    mlx_submission_graph_quota* out, size_t capacity, void* owner, void (*retire)(void*)) {
  if (!out || out->ctx || !owner || !retire) return 1;
  out->ctx = mlx::core::submission::GraphQuota::try_create(capacity, owner, retire);
  return out->ctx ? 0 : 1;
}
extern "C" void mlx_submission_graph_quota_retain(mlx_submission_graph_quota quota) {
  if (quota.ctx) static_cast<mlx::core::submission::GraphQuota*>(quota.ctx)->retain();
}
extern "C" void mlx_submission_graph_quota_release(mlx_submission_graph_quota quota) {
  if (quota.ctx) static_cast<mlx::core::submission::GraphQuota*>(quota.ctx)->release();
}
extern "C" size_t mlx_submission_graph_quota_occupied(mlx_submission_graph_quota quota) {
  return quota.ctx ? static_cast<mlx::core::submission::GraphQuota*>(quota.ctx)->occupied_bytes() : 0;
}

extern "C" int mlx_submission_scope_new_retaining_with_arenas(
    mlx_submission_scope* out, void* owner, void (*retire)(void*),
    mlx_submission_record_quota records, mlx_submission_graph_quota graph) {
  if (!out || out->ctx || !owner || !retire) return 1;
  try {
    out->ctx = new Scope(owner, retire,
        static_cast<mlx::core::submission::RecordQuota*>(records.ctx),
        static_cast<mlx::core::submission::GraphQuota*>(graph.ctx));
    return 0;
  } catch (const mlx::core::submission::RecordQuotaError&) { return 2;
  } catch (const mlx::core::submission::GraphQuotaError&) { return 3;
  } catch (...) { return 1; }
}

extern "C" int mlx_submission_scope_new_original_child(
    mlx_submission_scope* out, void* owner, void (*retire)(void*),
    mlx_submission_record_quota records, mlx_submission_graph_quota graph,
    mlx_submission_observer parent) {
  if (!out || out->ctx || !owner || !retire) return 1;
  if (!parent.ctx || !records.ctx || !graph.ctx) return 4;
  try {
    out->ctx = Scope::original_child(*static_cast<Scope*>(parent.ctx), owner, retire,
        static_cast<mlx::core::submission::RecordQuota*>(records.ctx),
        static_cast<mlx::core::submission::GraphQuota*>(graph.ctx));
    return 0;
  } catch (const mlx::core::submission::NativeControlError&) { return 4;
  } catch (...) { return 1; }
}
extern "C" size_t mlx_submission_scope_original_child_control_bytes() {
  return Scope::original_child_control_bytes() + sizeof(mlx_submission_scope*) +
      sizeof(void*) + sizeof(void (*)(void*)) + sizeof(mlx_submission_record_quota) +
      sizeof(mlx_submission_graph_quota) + sizeof(mlx_submission_observer);
}

extern "C" unsigned mlx_submission_scope_enable_original_controls(mlx_submission_scope scope) {
  auto* value = get_scope(scope);
  return static_cast<unsigned>(value ? value->enable_original_controls()
      : mlx::core::submission::NativeControlFailure::missing_scope);
}

extern "C" unsigned mlx_submission_native_control_layout_for(
    mlx_submission_native_control_layout* out) {
  using namespace mlx::core;
  EventControlLayout event{};
  if (!out || !event_control_layout(event)) return static_cast<unsigned>(
      submission::NativeControlFailure::unsupported_backend);
  // These are the actual named fixed host/worker/dispatch representations.
  // The variable F body is reserved with sizeof(TaskNode<F>) at its producer.
  const size_t fixed = event.construction_controls + sizeof(event) +
      sizeof(mlx_submission_native_control_layout) + sizeof(mlx_submission_scope) +
      sizeof(submission::NativeControlBinding) + sizeof(submission::NativeControlFailure) +
      sizeof(submission::NativeControlError) + sizeof(submission::Scope*) +
      sizeof(submission::RecordDispatchGuard) + sizeof(scheduler::TaskContext) +
      sizeof(scheduler::TaskOwner) * 2 + sizeof(scheduler::CpuStreamToken) * 2 +
      sizeof(FailureCarrierRef) * 2 + sizeof(mlx_prefill_failure) +
      sizeof(FailureCarrier*) + sizeof(scheduler::StreamThread*) +
      sizeof(scheduler::Scheduler*) + sizeof(bool) + sizeof(unsigned);
  *out = {event.object_bytes, event.shared_bytes, event.shared_alignment,
          event.graph_extent, event.platform_events, sizeof(scheduler::Task),
          alignof(scheduler::Task), fixed};
  return 0;
}

extern "C" unsigned mlx_submission_scope_require_original_controls(mlx_submission_scope scope) {
  auto* value = get_scope(scope);
  return static_cast<unsigned>(value ? value->require_original_controls()
      : mlx::core::submission::NativeControlFailure::missing_scope);
}

extern "C" bool mlx_submission_runtime_preparation_allowed(void) {
  auto* scope = mlx::core::submission::current_scope();
  return !scope || !scope->original_controls_required();
}

extern "C" void mlx_submission_static_layout_for(mlx_submission_static_layout* out) {
  if (!out) return;
  const auto layout = mlx::core::submission::submission_runtime_layout();
  *out = {layout.module_bytes, layout.thread_bytes, layout.dynamic_registry_bytes,
      layout.constant_registry};
}

extern "C" unsigned mlx_submission_prepare_runtime(
    mlx_submission_runtime_baseline* out, mlx_stream operation, mlx_stream weights) {
  using namespace mlx::core;
  if (!out || !operation.ctx || !weights.ctx) return 3;
  if (!mlx_submission_runtime_preparation_allowed()) return 2;
  try {
    const std::array<Stream, 2> streams{mlx_stream_get_(operation), mlx_stream_get_(weights)};
    mlx_submission_runtime_baseline facts{};
    // GPU dispatch also uses the common Scheduler's counted-task accounting.
    // Prepare it here even when neither selected stream needs a CPU worker.
    (void)scheduler::scheduler();
    facts.selected_streams = streams[0] == streams[1] ? 1 : 2;
    for (size_t i = 0; i < facts.selected_streams; ++i) {
      prepare_event_runtime(streams[i].device);
      if (streams[i].device == Device::cpu) {
        (void)scheduler::prepare_cpu_stream(streams[i]);
        ++facts.cpu_workers;
      }
    }
    facts.scheduler_object_bytes = sizeof(scheduler::Scheduler);
    facts.worker_object_bytes = facts.cpu_workers * sizeof(scheduler::StreamThread);
    facts.event_runtime_object_bytes = event_runtime_object_bytes();
    facts.native_threads = facts.cpu_workers;
    // Bits name actual unpriced persistent populations, not guessed bytes:
    // 1=worker table, 2=thread/TLS startup, 4=fallback String/shared storage,
    // 8=device maps/libraries/platform resources. A zero mask is not a grant.
    facts.unpriced_populations = (facts.cpu_workers ? 7 : 1) |
        (facts.event_runtime_object_bytes ? 8 : 0);
    *out = facts;
    return 0;
  } catch (const std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
}

extern "C" unsigned mlx_pipeline_cache_layout_for(mlx_pipeline_cache_layout* out, size_t attempts) {
  using Cache = mlx::core::fast::PreparedPipelineCache;
  using Status = mlx::core::fast::PipelineCacheStatus;
  if (!out) return static_cast<unsigned>(Status::invalid);
  mlx::core::fast::PipelineCacheLayout layout;
  const auto cause = Cache::layout(attempts, layout);
  if (cause != Status::success) return static_cast<unsigned>(cause);
  constexpr size_t controls = sizeof(mlx_pipeline_cache_layout) +
      sizeof(mlx::core::fast::PipelineCacheLayout) + sizeof(mlx_pipeline_cache) * 3 +
      sizeof(mlx_submission_graph_quota) + sizeof(Cache*) * 3 +
      sizeof(Status) * 2 + sizeof(size_t) * 2 + sizeof(void (*)(void*));
  if (layout.control_bytes > SIZE_MAX - controls) return static_cast<unsigned>(Status::overflow);
  *out = {layout.allocation_bytes, layout.alignment, layout.control_bytes + controls};
  return 0;
}
extern "C" unsigned mlx_pipeline_cache_new_retaining(mlx_pipeline_cache* out,
    size_t attempts, void* owner, void (*retire)(void*)) {
  using Cache = mlx::core::fast::PreparedPipelineCache;
  using Status = mlx::core::fast::PipelineCacheStatus;
  if (!out || out->ctx) return static_cast<unsigned>(Status::invalid);
  Cache* value = nullptr;
  const auto cause = Cache::create(attempts, owner, retire, value);
  if (cause == Status::success) out->ctx = value;
  return static_cast<unsigned>(cause);
}
extern "C" unsigned mlx_pipeline_cache_install(mlx_pipeline_cache cache, mlx_submission_graph_quota graph) {
  using Status = mlx::core::fast::PipelineCacheStatus;
  if (!cache.ctx || !graph.ctx) return static_cast<unsigned>(Status::invalid);
  const bool installed = static_cast<mlx::core::submission::GraphQuota*>(graph.ctx)->install_pipeline_cache(
      static_cast<mlx::core::fast::PreparedPipelineCache*>(cache.ctx));
  return static_cast<unsigned>(installed ? Status::success : Status::already_bound);
}
extern "C" void mlx_pipeline_cache_free(mlx_pipeline_cache cache) {
  if (cache.ctx) static_cast<mlx::core::fast::PreparedPipelineCache*>(cache.ctx)->release();
}

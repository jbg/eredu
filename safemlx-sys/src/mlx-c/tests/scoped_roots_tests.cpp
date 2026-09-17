#include <thread>
#include "doctest/doctest.h"
#include "mlx/c/prefill_roots.h"
#include "mlx/c/private/array.h"
#include "mlx/failure.h"
#include "mlx/graph_quota.h"
#include "mlx/submission.h"
#include "mlx/ops.h"
#include "mlx/event.h"
#include "mlx/scheduler.h"
#include "mlx/scoped_evaluation.h"
#include "mlx/memory.h"
#include "mlx/c/error.h"
#include <atomic>
#include <chrono>
#include <vector>
#include <optional>
#include "mlx/c/event.h"
using namespace mlx::core;
namespace {
void release_count(void* raw) { ++*static_cast<std::atomic<size_t>*>(raw); }
struct NativeFailureOwner {
  mlx_prefill_failure value{};
  explicit NativeFailureOwner(std::atomic<size_t>& count) {
    REQUIRE(mlx_prefill_failure_new_retaining(&value, &count, release_count) == 0);
  }
  ~NativeFailureOwner() { mlx_prefill_failure_free(value); }
};
struct Arena {
  submission::GraphQuota* value{submission::GraphQuota::try_create(1 << 20, nullptr, nullptr)};
  ~Arena() { if (value) value->release(); }
};
struct Enter {
  submission::Scope* value;
  explicit Enter(submission::GraphQuota* arena, submission::RecordQuota* records = nullptr)
      : value(new submission::Scope(nullptr, nullptr, records, arena)) {}
  ~Enter() { value->seal(); value->release(); }
  mlx_submission_scope raw() const { return mlx_submission_scope{value}; }
};
struct Roots {
  mlx_prefill_roots value{};
  ~Roots() { mlx_prefill_roots_free(value); }
};
mlx_array view(array& value) { return mlx_array{&value, nullptr}; }
void settle(submission::Scope& scope) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (scope.query_records().pending) {
    const auto status = scope.progress_scoped();
    REQUIRE(status == submission::ScopedProgress::observed);
    REQUIRE(std::chrono::steady_clock::now() < end);
    std::this_thread::yield();
  }
}
}
TEST_CASE("scoped roots preserve nonzero output and observe the same sealed scope") {
  const auto stream = default_stream(Device::cpu);
  std::atomic<size_t> count{0};
  {
    Arena arena;
    REQUIRE(arena.value);
    NativeFailureOwner failure(count);
    Roots roots;
    REQUIRE(mlx_prefill_roots_new_owned(&roots.value, 2,
        mlx_submission_graph_quota{arena.value}, failure.value) == 0);
    Enter scope(arena.value);
    REQUIRE(scope.value->enable_scoped_observation());
    REQUIRE(mlx_prefill_roots_bind(roots.value, scope.raw()) == 0);
    CHECK(mlx_prefill_roots_bind(roots.value, scope.raw()) == 4);
    auto source = array({2.0f, -3.0f, 7.0f});
    auto output = add(source, array(4.0f), stream);
    REQUIRE(mlx_prefill_roots_append(roots.value, view(output)) == 0);
    REQUIRE(mlx_prefill_roots_append(roots.value, view(source)) == 0);
    {
      Enter nested(arena.value);
      CHECK(mlx_prefill_roots_submit_scoped(roots.value, nested.raw()) == 4);
      CHECK(mlx_prefill_roots_complete_current(roots.value) == 4);
    }
    REQUIRE(mlx_prefill_roots_submit_scoped(roots.value, scope.raw()) == 0);
    scope.value->seal();
    CHECK(mlx_prefill_roots_complete_current(roots.value) == 4);
    REQUIRE(mlx_prefill_roots_wait_scoped(roots.value, scope.raw()) == 0);
    REQUIRE(mlx_prefill_roots_query_scoped(roots.value, scope.raw()) == 0);
    REQUIRE(mlx_prefill_roots_validate_scoped(roots.value, scope.raw()) == 0);
    CHECK(output.data<float>()[0] == 6.0f);
    CHECK(output.data<float>()[1] == 1.0f);
    CHECK(output.data<float>()[2] == 11.0f);
    settle(*scope.value);
    CHECK(count == 0);
  }
  submission::progress_records();
  submission::retire_records();
  CHECK(count == 1);
}
TEST_CASE("scoped root submit failure retains its actual cause after independent retirement") {
  const auto stream = default_stream(Device::cpu);
  std::atomic<size_t> count{0};
  mlx_prefill_failure escaped{};
  {
    Arena arena;
    REQUIRE(arena.value);
    NativeFailureOwner failure(count);
    Roots roots;
    REQUIRE(mlx_prefill_roots_new_owned(&roots.value, 1,
        mlx_submission_graph_quota{arena.value}, failure.value) == 0);
    Enter scope(arena.value);
    REQUIRE(scope.value->enable_scoped_observation());
    REQUIRE(mlx_prefill_roots_bind(roots.value, scope.raw()) == 0);
    auto source = array({3.0f, 9.0f});
    auto output = add(source, array(2.0f), stream);
    REQUIRE(mlx_prefill_roots_append(roots.value, view(output)) == 0);
    // Test-only exhaustion of this actual original arena. No production fit
    // estimate or replacement allowance is inferred from this fixture capacity.
    std::vector<void*> filler;
    while (auto* value = arena.value->try_allocate(1, 1)) filler.push_back(value);
    CHECK(mlx_prefill_roots_submit_scoped(roots.value, scope.raw()) == 7);
    CHECK(mlx_prefill_roots_size(roots.value) == 1);
    CHECK(mlx_prefill_roots_query_scoped(roots.value, scope.raw()) == 5);
    CHECK(mlx_prefill_roots_wait_scoped(roots.value, scope.raw()) == 5);
    CHECK(mlx_prefill_roots_validate_scoped(roots.value, scope.raw()) == 5);
    mlx_prefill_failure_view cause{};
    REQUIRE(mlx_prefill_failure_view_get(&cause, failure.value) == 0);
    CHECK(cause.kind == unsigned(FailureKind::exception));
    const auto* original = static_cast<FailureCarrier*>(failure.value.ctx)->borrow();
    REQUIRE(original);
    CHECK(original->exception_type == &typeid(submission::GraphQuotaError));
    CHECK(cause.message_size > 0);
    CHECK(cause.source_type_size > 0);
    escaped = failure.value;
    mlx_prefill_failure_retain(escaped);
    for (auto* value : filler) arena.value->deallocate(value, 1, 1);
    scope.value->seal();
    settle(*scope.value); // Spent above was never used as terminal evidence.
  }
  submission::progress_records();
  submission::retire_records();
  CHECK(count == 0);
  mlx_prefill_failure_view cause{};
  REQUIRE(mlx_prefill_failure_view_get(&cause, escaped) == 0);
  CHECK(cause.kind == unsigned(FailureKind::exception));
  CHECK(static_cast<FailureCarrier*>(escaped.ctx)->borrow()->exception_type ==
      &typeid(submission::GraphQuotaError));
  CHECK(cause.message_size > 0);
  mlx_prefill_failure_free(escaped);
  CHECK(count == 1);
}

TEST_CASE("original roots execute nonzero Add with Graph Event and typed task controls") {
  using Failure = submission::NativeControlFailure;
  auto stream = new_stream(Device::cpu);
  scheduler::CpuStreamToken worker;
  REQUIRE(scheduler::prepared_cpu_stream(stream, worker) == Failure::missing_worker);
  mlx_submission_runtime_baseline baseline{};
  REQUIRE(mlx_submission_prepare_runtime(&baseline, mlx_stream{&stream},
      mlx_stream{&stream}) == 0);
  REQUIRE(scheduler::prepared_cpu_stream(stream, worker) == Failure::none);
  REQUIRE(scheduler::cpu_stream_progress(worker).accepted == 0);
  REQUIRE(baseline.cpu_workers == 1);
  std::atomic<size_t> count{0};
  struct Records {
    submission::RecordQuota* value{
        submission::RecordQuota::create(1 << 20, nullptr, nullptr)};
    ~Records() { value->release(); }
  } records;
  {
    Arena arena;
    NativeFailureOwner failure(count);
    Roots roots;
    REQUIRE(mlx_prefill_roots_new_owned(&roots.value, 2,
        mlx_submission_graph_quota{arena.value}, failure.value) == 0);
    Enter scope(arena.value, records.value);
    REQUIRE(scope.value->enable_scoped_observation());
    REQUIRE(scope.value->require_original_controls() == Failure::none);
    REQUIRE(mlx_prefill_roots_bind(roots.value, scope.raw()) == 0);
    REQUIRE(scope.value->enable_original_controls() == Failure::none);
    auto source = array({2.0f, -3.0f, 7.0f});
    auto output = add(source, array(4.0f), stream);
    REQUIRE(mlx_prefill_roots_append(roots.value, view(output)) == 0);
    REQUIRE(mlx_prefill_roots_append(roots.value, view(source)) == 0);
    const auto before = scheduler::cpu_stream_progress(worker).accepted;
    REQUIRE(mlx_prefill_roots_submit_scoped(roots.value, scope.raw()) == 0);
    CHECK(scheduler::cpu_stream_progress(worker).accepted > before);
    scope.value->seal();
    REQUIRE(mlx_prefill_roots_wait_scoped(roots.value, scope.raw()) == 0);
    REQUIRE(mlx_prefill_roots_validate_scoped(roots.value, scope.raw()) == 0);
    CHECK(output.data<float>()[0] == 6.0f);
    CHECK(output.data<float>()[1] == 1.0f);
    CHECK(output.data<float>()[2] == 11.0f);
    settle(*scope.value);
    CHECK(count == 0);
  }
  submission::progress_records();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (submission::try_retire_records() == submission::RetirementPass::busy) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
  CHECK(count == 1);
  CHECK(records.value->occupied_bytes() == 0);
}

namespace {
struct SyncRecords {
  submission::RecordQuota* value{submission::RecordQuota::create(1 << 20, nullptr, nullptr)};
  ~SyncRecords() { value->release(); }
};
struct ReturnedFailure {
  mlx_prefill_failure value{};
  ~ReturnedFailure() { mlx_prefill_failure_free(value); }
};
void configure_sync(Enter& scope, NativeFailureOwner& failure) {
  REQUIRE(scope.value->enable_scoped_observation());
  REQUIRE(scope.value->require_original_controls() == submission::NativeControlFailure::none);
  REQUIRE(mlx_prefill_failure_bind_original_scope(failure.value, scope.raw()) == 0);
  REQUIRE(scope.value->enable_original_controls() == submission::NativeControlFailure::none);
}
Stream first_sync_stream() {
  auto stream = new_stream(Device::cpu);
  scheduler::CpuStreamToken token;
  REQUIRE(scheduler::prepared_cpu_stream(stream, token) == submission::NativeControlFailure::missing_worker);
  mlx_submission_runtime_baseline facts{};
  REQUIRE(mlx_submission_prepare_runtime(&facts, mlx_stream{&stream}, mlx_stream{&stream}) == 0);
  REQUIRE(scheduler::prepared_cpu_stream(stream, token) == submission::NativeControlFailure::none);
  REQUIRE(scheduler::cpu_stream_progress(token).accepted == 0);
  return stream;
}
struct TextHandler {
  size_t calls{0};
  TextHandler() { mlx_set_error_handler([](const char*, void* raw) {
    ++*static_cast<size_t*>(raw);
  }, &calls, [](void*) {}); }
  ~TextHandler() { mlx_set_error_handler(nullptr, nullptr, nullptr); }
};
}
TEST_CASE("original synchronous evaluation first use preserves values and already available roots") {
  auto stream = first_sync_stream();
  std::atomic<size_t> retired{0};
  {
    Arena arena; SyncRecords records; NativeFailureOwner failure(retired);
    Enter scope(arena.value, records.value); configure_sync(scope, failure);
    auto source = array({2.0f, -3.0f, 7.0f});
    auto output = add(source, array(4.0f), stream);
    ReturnedFailure returned;
    REQUIRE(mlx_array_eval_scoped(view(output), &returned.value) == 0);
    CHECK(returned.value.ctx == failure.value.ctx);
    CHECK(output.status() == array::Status::available);
    CHECK(output.data<float>()[0] == 6.0f);
    CHECK(output.data<float>()[1] == 1.0f);
    CHECK(output.data<float>()[2] == 11.0f);
    const auto accepted = scheduler::cpu_stream_progress(stream)->accepted;
    ReturnedFailure again;
    REQUIRE(mlx_array_eval_scoped(view(output), &again.value) == 0);
    CHECK(scheduler::cpu_stream_progress(stream)->accepted == accepted);
    auto other = add(source, source, stream);
    other.eval(); // Native item/eval route must also avoid null scoped_outcome.
    CHECK(other.data<float>()[2] == 14.0f);
    CHECK(other.is_available());
    scope.value->seal(); settle(*scope.value);
  }
  submission::progress_records(); submission::retire_records();
  CHECK(retired == 1);
}
TEST_CASE("original synchronous evaluation observes an already scheduled same-scope root") {
  auto stream = first_sync_stream();
  std::atomic<size_t> retired{0};
  {
    Arena arena; SyncRecords records; NativeFailureOwner failure(retired);
    Enter scope(arena.value, records.value); configure_sync(scope, failure);
    auto output = add(array({1.0f, 4.0f}), array(3.0f), stream);
    Completion existing;
    REQUIRE(async_eval_with_completion_scoped({output}, *scope.value, existing) == submission::ScopedProgress::observed);
    ReturnedFailure returned;
    REQUIRE(mlx_array_eval_scoped(view(output), &returned.value) == 0);
    CHECK(output.data<float>()[0] == 4.0f);
    CHECK(output.data<float>()[1] == 7.0f);
    scope.value->seal(); settle(*scope.value);
  }
  submission::progress_records(); submission::retire_records();
  CHECK(retired == 1);
}
TEST_CASE("original synchronous evaluation missing Record refuses before native work") {
  auto stream = first_sync_stream();
  auto available = array({1.0f});
  std::atomic<size_t> retired{0};
  Arena arena; NativeFailureOwner failure(retired);
  Enter scope(arena.value);
  REQUIRE(scope.value->enable_scoped_observation());
  REQUIRE(scope.value->require_original_controls() == submission::NativeControlFailure::none);
  REQUIRE(mlx_prefill_failure_bind_original_scope(failure.value, scope.raw()) == 0);
  REQUIRE(scope.value->enable_original_controls() == submission::NativeControlFailure::missing_record);
  const auto occupied = arena.value->occupied_bytes();
  ReturnedFailure returned; TextHandler handler;
  CHECK(mlx_array_eval_scoped(view(available), &returned.value) == 4);
  CHECK(returned.value.ctx == failure.value.ctx);
  CHECK(arena.value->occupied_bytes() == occupied);
  CHECK(scheduler::cpu_stream_progress(stream)->accepted == 0);
  CHECK(handler.calls == 0);
}
TEST_CASE("original synchronous evaluation rejects foreign pending event without submitting") {
  auto stream = first_sync_stream();
  auto value = array({2.0f});
  Event foreign(stream);
  foreign.set_value(1);
  value.attach_event(foreign);
  value.set_status(array::Status::evaluated);
  std::atomic<size_t> retired{0};
  Arena arena; SyncRecords records; NativeFailureOwner failure(retired);
  Enter scope(arena.value, records.value); configure_sync(scope, failure);
  ReturnedFailure returned;
  CHECK(mlx_array_eval_scoped(view(value), &returned.value) == 4);
  CHECK(scheduler::cpu_stream_progress(stream)->accepted == 0);
  CHECK(value.event().valid());
  value.detach_event(); // Fixture-only event never had a producer.
}
TEST_CASE("original synchronous allocation failure keeps actual cause after scope and array retirement") {
  auto stream = first_sync_stream();
  std::atomic<size_t> retired{0};
  ReturnedFailure escaped;
  {
    Arena arena; SyncRecords records; NativeFailureOwner failure(retired);
    Enter scope(arena.value, records.value); configure_sync(scope, failure);
    auto output = add(array({3.0f, 9.0f}), array(2.0f), stream);
    std::vector<void*> filler; // Ordinary fixture-only exhaustion instrumentation.
    while (auto* value = arena.value->try_allocate(1, 1)) filler.push_back(value);
    TextHandler handler;
    CHECK(mlx_array_eval_scoped(view(output), &escaped.value) == 7);
    CHECK(handler.calls == 0);
    REQUIRE(escaped.value.ctx == failure.value.ctx);
    const auto* cause = static_cast<FailureCarrier*>(escaped.value.ctx)->borrow();
    REQUIRE(cause);
    CHECK(cause->exception_type == &typeid(submission::GraphQuotaError));
    CHECK(output.status() == array::Status::unscheduled);
    for (auto* value : filler) arena.value->deallocate(value, 1, 1);
    scope.value->seal(); settle(*scope.value);
  }
  submission::progress_records(); submission::retire_records();
  CHECK(retired == 0);
  mlx_prefill_failure_view view{};
  REQUIRE(mlx_prefill_failure_view_get(&view, escaped.value) == 0);
  CHECK(view.message_size > 0);
  auto alias = escaped.value; mlx_prefill_failure_retain(alias);
  mlx_prefill_failure_free(escaped.value); escaped.value = {};
  CHECK(retired == 0);
  mlx_prefill_failure_free(alias);
  CHECK(retired == 1);
}
TEST_CASE("original synchronous pressure returns funded progress without advancing unrelated work") {
  auto stream = first_sync_stream();
  auto unrelated = new_stream(Device::cpu);
  std::atomic<bool> release{false}, entered{false}, exited{false};
  scheduler::enqueue_counted(unrelated, [&] {
    entered = true;
    while (!release.load()) std::this_thread::yield();
    exited = true;
  });
  struct Unblock {
    std::atomic<bool>& release;
    ~Unblock() { release = true; }
  } unblock{release};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!entered.load()) { REQUIRE(std::chrono::steady_clock::now() < deadline); std::this_thread::yield(); }
  std::atomic<size_t> retired{0};
  {
    Arena arena; SyncRecords records; NativeFailureOwner failure(retired);
    Enter scope(arena.value, records.value); configure_sync(scope, failure);
    auto output = add(array({3.0f, 9.0f}), array(2.0f), stream);
    struct MemoryLimit { size_t old{set_memory_limit(0)}; ~MemoryLimit() { set_memory_limit(old); } } limit;
    ReturnedFailure returned;
    CHECK(mlx_array_eval_scoped(view(output), &returned.value) == 8);
    CHECK_FALSE(exited.load());
    CHECK(returned.value.ctx == failure.value.ctx);
    CHECK(scope.value->query().failed);
    // Refusal is not settlement. Unblock the independent fixture producer at
    // its ordinary boundary, then use exact retained Scope recovery evidence.
    release = true;
    scope.value->seal(); settle(*scope.value);
  }
  while (!exited.load()) { REQUIRE(std::chrono::steady_clock::now() < deadline); std::this_thread::yield(); }
  submission::progress_records(); submission::retire_records();
  CHECK(retired == 1);
}

namespace {
struct UnsupportedSyncRecord final : submission::Record {
  explicit UnsupportedSyncRecord(Allocation allocation) : Record(allocation) {}
  bool side_effects_terminal() const noexcept override { return true; }
};
}
TEST_CASE("original synchronous unobservable interval returns without a global cleanup fallback") {
  auto stream = first_sync_stream();
  std::atomic<size_t> retired{0};
  {
    Arena arena; SyncRecords records; NativeFailureOwner failure(retired);
    Enter scope(arena.value, records.value); configure_sync(scope, failure);
    auto unknown = submission::Record::create<UnsupportedSyncRecord>();
    unknown->enter(); auto* accepted = unknown.release(); accepted->finish(false);
    auto output = add(array({1.0f, 3.0f}), array(2.0f), stream);
    ReturnedFailure returned;
    CHECK(mlx_array_eval_scoped(view(output), &returned.value) == 9);
    CHECK(scope.value->query_records().pending);
    // Test-only cleanup after the refused interval. The production evaluator
    // did not run this ordinary global operation or infer terminal readiness.
    scope.value->seal();
    submission::progress_records();
    settle(*scope.value);
  }
  submission::progress_records(); submission::retire_records();
  CHECK(retired == 1);
}
TEST_CASE("original evaluation legacy timing and stream wait refuse before producer construction") {
  auto stream = first_sync_stream();
  std::atomic<size_t> retired{0};
  Arena arena; SyncRecords records; NativeFailureOwner failure(retired);
  Enter scope(arena.value, records.value); configure_sync(scope, failure);
  auto output = add(array({1.0f}), array(2.0f), stream);
  Completion empty;
  const auto before = scheduler::cpu_stream_progress(stream)->accepted;
  auto unobservable = [&](auto&& invoke) {
    bool caught = false;
    try { invoke(); }
    catch (const ScopedEvaluationError& error) {
      CHECK(error.outcome() == ScopedEvaluation::unobservable); caught = true;
    }
    REQUIRE(caught);
  };
  unobservable([&] { (void)async_eval_with_timing({output}, stream); });
  unobservable([&] { empty.wait(stream); });
  unobservable([&] { (void)empty.elapsed_seconds(); });
  unobservable([&] { (void)empty.try_elapsed_seconds(); });
  CHECK(scheduler::cpu_stream_progress(stream)->accepted == before);
  // Matching non-timed siblings use the same scoped evaluator and completion.
  auto completion = async_eval_with_completion({output});
  completion.wait();
  CHECK(completion.is_complete());
  output.wait();
  CHECK(output.data<float>()[0] == 3.0f);
  scope.value->seal(); settle(*scope.value);
}
TEST_CASE("ordinary synchronous native array and completion behavior stays available") {
  auto stream = default_stream(Device::cpu);
  auto output = add(array({1.0f, 4.0f}), array(3.0f), stream);
  output.eval();
  CHECK(output.data<float>()[0] == 4.0f);
  auto next = add(output, output, stream);
  auto completion = async_eval_with_completion({next});
  completion.wait();
  CHECK(completion.is_complete());
  next.wait();
  CHECK(next.data<float>()[1] == 14.0f);
  auto scalar = add(array(3.0f), array(4.0f), stream);
  CHECK(scalar.item<float>() == 7.0f);
}

TEST_CASE("original completion provenance survives sealed scope without legacy cleanup") {
  auto stream = first_sync_stream();
  std::atomic<size_t> retired{0};
  {
    Arena arena; SyncRecords records; NativeFailureOwner failure(retired);
    Enter scope(arena.value, records.value); configure_sync(scope, failure);
    auto output = add(array({2.0f}), array(5.0f), stream);
    auto completion = async_eval_with_completion({output});
    scope.value->seal();
    // The no-argument legacy wrapper has lost TLS but must not lose provenance.
    CHECK_THROWS_AS(completion.wait(), ScopedEvaluationError);
    CHECK_THROWS_AS(completion.is_complete(), ScopedEvaluationError);
    // Actual retained Scope evidence still supports the shared fixed API.
    REQUIRE(wait_scoped_completion(*scope.value, completion) == ScopedEvaluation::complete);
    REQUIRE(validate_scoped_array(output, *scope.value) == ScopedEvaluation::complete);
    CHECK(output.data<float>()[0] == 7.0f);
    settle(*scope.value);
  }
  submission::progress_records(); submission::retire_records();
  CHECK(retired == 1);
}

namespace {
void selected_stream_prepared_roots(Device device) {
  using Failure = submission::NativeControlFailure;
  auto selected = new_stream(device);
  auto producer = new_stream(device);
  auto missing = new_stream(Device::cpu);
  mlx_submission_runtime_baseline runtime{};
  REQUIRE(mlx_submission_prepare_runtime(&runtime, mlx_stream{&selected},
      mlx_stream{&producer}) == 0);
  scheduler::CpuStreamToken absent;
  REQUIRE(scheduler::prepared_cpu_stream(missing, absent) == Failure::missing_worker);
  array identity({2.0f, -3.0f, 7.0f});
  REQUIRE(identity.status() == array::Status::available);
  for (unsigned mode = 0; mode != 3; ++mode) {
    std::atomic<size_t> retired{0};
    {
      Arena arena; SyncRecords records; NativeFailureOwner failure(retired);
      Roots roots;
      REQUIRE(mlx_prefill_roots_new_owned(&roots.value, 1,
          mlx_submission_graph_quota{arena.value}, failure.value) == 0);
      Enter scope(arena.value, records.value);
      REQUIRE(scope.value->enable_scoped_observation());
      REQUIRE(scope.value->require_original_controls() == Failure::none);
      REQUIRE(mlx_prefill_roots_bind(roots.value, scope.raw()) == 0);
      REQUIRE(scope.value->enable_original_controls() == Failure::none);
      mlx_submission_observer observer{};
      REQUIRE(mlx_submission_observer_current(&observer) == 0);
      mlx_operation_event parent{};
      REQUIRE(mlx_operation_event_new(&parent, observer) == 0);
      REQUIRE(mlx_operation_event_append(parent, view(identity)) == 0);
      REQUIRE(mlx_operation_event_submit_on_stream(parent, mlx_stream{&producer}) == 0);
      REQUIRE(mlx_operation_event_wait(parent) == 0);
      settle(*scope.value);
      const auto graph_before = arena.value->occupied_bytes();
      const auto records_before = records.value->occupied_bytes();
      auto refusal = [&](mlx_stream stream) {
        CHECK(mlx_prefill_roots_complete_current_on_stream(roots.value, stream) == 4);
        CHECK(mlx_prefill_roots_size(roots.value) == 0);
        CHECK(arena.value->occupied_bytes() == graph_before);
        CHECK(records.value->occupied_bytes() == records_before);
        CHECK(!scope.value->query().failed);
      };
      refusal(mlx_stream{});
      refusal(mlx_stream{&missing});
      auto invalid = Stream{-1, device};
      refusal(mlx_stream{&invalid});
      auto wrong_device = Stream{selected.index, Device{device.type, 1000000}};
      refusal(mlx_stream{&wrong_device});
      {
        Enter child(arena.value, records.value);
        CHECK(mlx_prefill_roots_complete_current_on_stream(
            roots.value, mlx_stream{&selected}) == 4);
      }
      // Refusal left filling storage usable; all roots retain their actual
      // original population and no C wrapper is copied into a Rust/host Vec.
      std::optional<array> pending;
      if (mode == 1) REQUIRE(mlx_prefill_roots_append(roots.value, view(identity)) == 0);
      if (mode == 2) {
        pending.emplace(add(identity, identity, selected));
        REQUIRE(mlx_prefill_roots_append(roots.value, view(*pending)) == 0);
      }
      REQUIRE(mlx_operation_event_wait_stream(parent, mlx_stream{&selected}) == 0);
      CHECK(mlx_operation_event_wait(parent) == (device == Device::gpu ? 8u : 0u));
      REQUIRE(mlx_prefill_roots_complete_current_on_stream(
          roots.value, mlx_stream{&selected}) == 0);
      REQUIRE(mlx_operation_event_wait(parent) == 0);
      CHECK(mlx_prefill_roots_complete_current_on_stream(
          roots.value, mlx_stream{&selected}) == 5);
      if (pending) {
        CHECK(pending->data<float>()[0] == 4.0f);
        CHECK(pending->data<float>()[1] == -6.0f);
        CHECK(pending->data<float>()[2] == 14.0f);
      }
      CHECK(identity.data<float>()[0] == 2.0f);
      CHECK(scope.value->active_on_owner_thread());
      CHECK(submission::current_scope() == scope.value);
      settle(*scope.value);
      mlx_operation_event_free(parent);
      REQUIRE(mlx_submission_observer_retire(observer) == 0);
      mlx_submission_observer_release(observer);
    }
    submission::progress_records(); submission::retire_records();
    CHECK(retired == 1);
  }
}
}
TEST_CASE("prepared roots close exact CPU stream for empty available and pending outputs") {
  selected_stream_prepared_roots(Device::cpu);
}
#ifdef MLX_C_PATCH_TEST_METAL
TEST_CASE("prepared roots close exact Metal stream for empty available and pending outputs") {
  selected_stream_prepared_roots(Device::gpu);
}
#endif

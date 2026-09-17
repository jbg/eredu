// Cold accepted-record observations, registered in the central test filter.
#include "doctest/doctest.h"
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include "mlx/c/submission.h"
#include "mlx/mlx.h"
#include "mlx/scheduler.h"
#include "mlx/submission.h"
using namespace mlx::core;
extern size_t native_recovery_without_allocations(void (*)(void*), void*);
namespace {
struct Scope {
  mlx_submission_scope raw{nullptr};
  Scope() { REQUIRE(mlx_submission_scope_new(&raw) == 0); }
  ~Scope() { if (raw.ctx) mlx_submission_scope_free(raw); }
  void seal() { REQUIRE(mlx_submission_scope_seal(raw) == 0); }
  mlx_submission_record_status records() const {
    mlx_submission_record_status status{true, true, true};
    REQUIRE(mlx_submission_scope_query_records(&status, raw) == 0);
    return status;
  }
  mlx_submission_status lifetime() const {
    mlx_submission_status status{};
    REQUIRE(mlx_submission_scope_query(&status, raw) == 0);
    return status;
  }
};
struct Effects {
  std::atomic<bool> terminal{false};
  std::atomic<bool> failed{false};
  std::atomic<int> observed{0};
  std::atomic<int> destroyed{0};
};
struct Record : submission::Record {
  explicit Record(std::shared_ptr<Effects> value) : effects(std::move(value)) {}
  ~Record() override { ++effects->destroyed; }
  bool side_effects_terminal() const noexcept override {
    ++effects->observed;
    return effects->terminal.load(std::memory_order_acquire);
  }
  bool side_effects_failed() const noexcept override { return effects->failed.load(); }
  std::shared_ptr<Effects> effects;
};
Record* enter(const std::shared_ptr<Effects>& effects) {
  auto record = std::make_unique<Record>(effects);
  record->enter();
  return record.release(); // The canonical registry owns successful entry.
}
void progress_until(const Scope& scope) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (scope.records().pending && std::chrono::steady_clock::now() < end) {
    submission::progress_records(false);
    std::this_thread::yield();
  }
  REQUIRE_FALSE(scope.records().pending);
}
struct Gate {
  std::promise<void> release;
  std::shared_future<void> ready{release.get_future().share()};
  bool opened{false};
  void open() { if (!opened) { opened = true; release.set_value(); } }
  ~Gate() { open(); }
};
}
TEST_CASE("submission recovery record query excludes only live empty children") {
  Scope outer; Scope child; Scope descendant;
  outer.seal();
  CHECK(outer.lifetime().activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  CHECK_FALSE(outer.records().pending);
  CHECK_FALSE(child.records().pending);
  CHECK_FALSE(descendant.records().pending);
  auto effects = std::make_shared<Effects>();
  auto* record = enter(effects);
  CHECK(outer.records().pending);
  CHECK(child.records().pending);
  CHECK(descendant.records().pending);
  record->finish(false);
  effects->terminal = true;
  CHECK(outer.records().pending); // Query must not promote side-effect facts.
  CHECK(effects->observed == 0);
  progress_until(outer);
  CHECK_FALSE(child.records().pending);
  CHECK_FALSE(descendant.records().pending);
  CHECK(outer.lifetime().activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  CHECK(effects->destroyed == 0);
  descendant.seal(); child.seal();
  CHECK(outer.lifetime().activity == MLX_SUBMISSION_ACTIVITY_NONE);
  submission::retire_records();
  CHECK(effects->destroyed == 1);
}
TEST_CASE("submission recovery record query covers actual parent and descendant CPU work") {
  Scope outer;
  Gate parent_gate;
  auto parent_effects = std::make_shared<Effects>(); parent_effects->terminal = true;
  auto* parent_record = enter(parent_effects);
  auto first = new_stream(Device::cpu);
  parent_record->prepare_stream(first);
  scheduler::enqueue(first, [gate = parent_gate.ready] { gate.wait(); });
  parent_record->finish(false);
  Scope child;
  CHECK(outer.records().pending);
  CHECK_FALSE(child.records().pending); // New scopes do not cover earlier work.
  Gate child_gate;
  auto child_effects = std::make_shared<Effects>(); child_effects->terminal = true;
  auto* child_record = enter(child_effects);
  auto second = new_stream(Device::cpu);
  child_record->prepare_stream(second);
  scheduler::enqueue(second, [gate = child_gate.ready] { gate.wait(); });
  child_record->finish(false);
  outer.seal();
  CHECK(outer.records().pending);
  CHECK(child.records().pending);
  child_gate.open(); progress_until(child);
  CHECK(outer.records().pending);
  CHECK_FALSE(child.records().pending);
  parent_gate.open(); progress_until(outer);
  CHECK_FALSE(outer.records().failed);
  CHECK_FALSE(outer.records().blocked);
  CHECK(outer.lifetime().activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  child.seal();
  CHECK(outer.lifetime().activity == MLX_SUBMISSION_ACTIVITY_TERMINAL);
  CHECK(parent_effects->destroyed == 0);
  CHECK(child_effects->destroyed == 0);
  submission::retire_records();
  CHECK(parent_effects->destroyed == 1);
  CHECK(child_effects->destroyed == 1);
}
TEST_CASE("submission recovery record failure remains sticky after terminal evidence") {
  Scope outer; Scope child;
  Gate gate;
  auto effects = std::make_shared<Effects>(); effects->terminal = true;
  auto* record = enter(effects);
  auto stream = new_stream(Device::cpu);
  record->prepare_stream(stream);
  scheduler::enqueue(stream, [ready = gate.ready] { ready.wait(); });
  record->finish(true); // Real accepted stream work becomes blocked on failure.
  for (auto* scope : {&outer, &child}) {
    const auto status = scope->records();
    CHECK(status.pending); CHECK(status.failed); CHECK(status.blocked);
  }
  gate.open(); progress_until(outer);
  for (auto* scope : {&outer, &child}) {
    const auto status = scope->records();
    CHECK_FALSE(status.pending); CHECK(status.failed); CHECK(status.blocked);
  }
  child.seal(); outer.seal();
  CHECK(outer.lifetime().failed);
  CHECK(outer.lifetime().blocked);
  submission::retire_records();
  CHECK(effects->destroyed == 1);
}
TEST_CASE("submission recovery record query never polls allocates or retires") {
  Scope scope;
  auto effects = std::make_shared<Effects>();
  auto* record = enter(effects);
  record->finish(false);
  effects->terminal = true;
  struct Call { mlx_submission_scope scope; mlx_submission_record_status status{}; int result{-1}; } call{scope.raw};
  const auto attempts = native_recovery_without_allocations([](void* p) {
    auto& call = *static_cast<Call*>(p);
    call.result = mlx_submission_scope_query_records(&call.status, call.scope);
    (void)mlx_submission_scope_control_bytes();
  }, &call);
  CHECK(attempts == 0);
  CHECK(call.result == 0);
  CHECK(call.status.pending);
  CHECK(effects->observed == 0);
  CHECK(effects->destroyed == 0);
  CHECK(mlx_submission_scope_control_bytes() == sizeof(submission::Scope));
  progress_until(scope);
  CHECK_FALSE(scope.records().pending);
  CHECK(effects->destroyed == 0);
  submission::retire_records();
  CHECK(effects->destroyed == 1);
}
TEST_CASE("submission recovery record query rejects foreign thread and invalid arguments") {
  Scope scope;
  mlx_submission_record_status sentinel{true, false, true};
  CHECK(mlx_submission_scope_query_records(&sentinel, {nullptr}) == 1);
  CHECK(sentinel.pending);
  CHECK_FALSE(sentinel.failed);
  CHECK(sentinel.blocked);
  CHECK(mlx_submission_scope_query_records(nullptr, scope.raw) == 1);
  auto result = std::async(std::launch::async, [raw = scope.raw] {
    mlx_submission_record_status status{true, false, true};
    const auto code = mlx_submission_scope_query_records(&status, raw);
    return code == 1 && status.pending && !status.failed && status.blocked;
  });
  CHECK(result.get());
  CHECK_FALSE(scope.records().pending);
}
TEST_CASE("submission recovery record query does not acquire a contended registry") {
  struct LockedEffects {
    std::promise<void> entered;
    std::promise<void> release;
    std::shared_future<void> ready{release.get_future().share()};
    std::atomic<bool> held_until_release{false};
    std::atomic<bool> started{false};
  };
  struct LockedRecord : submission::Record {
    explicit LockedRecord(std::shared_ptr<LockedEffects> e) : effects(std::move(e)) {}
    bool side_effects_terminal() const noexcept override {
      effects->started = true;
      effects->entered.set_value();
      effects->held_until_release = effects->ready.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
      return true;
    }
    std::shared_ptr<LockedEffects> effects;
  };
  Scope scope;
  auto effects = std::make_shared<LockedEffects>();
  auto entered = effects->entered.get_future();
  auto record = std::make_unique<LockedRecord>(effects);
  record->enter(); record->finish(false); record.release();
  std::thread worker([effects] {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!effects->started && std::chrono::steady_clock::now() < end) {
      submission::progress_records(false);
      std::this_thread::yield();
    }
  });
  const auto entered_on_time = entered.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
  mlx_submission_record_status observed{true, true, true};
  const auto result = mlx_submission_scope_query_records(&observed, scope.raw);
  effects->release.set_value();
  worker.join();
  CHECK(result == 0);
  CHECK(entered_on_time);
  CHECK(observed.pending);
  CHECK(effects->held_until_release);
  CHECK_FALSE(scope.records().pending);
  submission::retire_records();
}
TEST_CASE("submission recovery record query preserves late secondary failure without progress") {
  Scope outer; Scope child;
  auto effects = std::make_shared<Effects>();
  auto* record = enter(effects);
  record->finish(false);
  effects->failed = true;
  const auto before = child.records();
  CHECK(before.pending);
  CHECK_FALSE(before.failed); // Failure has not yet reached the existing record.
  CHECK(effects->observed == 0);
  submission::progress_records(false);
  // Other test records can precede this one in the finite progress batch.
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!child.records().failed && std::chrono::steady_clock::now() < end) {
    submission::progress_records(false);
    std::this_thread::yield();
  }
  CHECK(outer.records().pending);
  CHECK(outer.records().failed);
  CHECK_FALSE(outer.records().blocked); // No stream work was accepted here.
  effects->terminal = true;
  CHECK(child.records().pending); // This query still cannot consume that fact.
  progress_until(outer);
  CHECK_FALSE(child.records().pending);
  CHECK(child.records().failed);
  CHECK(effects->destroyed == 0);
  submission::retire_records();
  CHECK(effects->destroyed == 1);
}
TEST_CASE("submission recovery record query retains actual asynchronous CPU failure") {
  Scope outer; Scope child;
  Gate gate;
  auto effects = std::make_shared<Effects>(); effects->terminal = true;
  auto* record = enter(effects);
  auto stream = new_stream(Device::cpu);
  record->prepare_stream(stream);
  scheduler::enqueue(stream, [ready = gate.ready] {
    ready.wait();
    throw std::runtime_error("accepted record readiness CPU fixture");
  });
  record->finish(false); // The submitting host call itself succeeded.
  CHECK(outer.records().pending);
  CHECK_FALSE(outer.records().failed);
  gate.open();
  progress_until(outer);
  for (auto* scope : {&outer, &child}) {
    const auto records = scope->records();
    CHECK_FALSE(records.pending);
    CHECK(records.failed);
    CHECK(records.blocked);
  }
  CHECK(outer.lifetime().activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  child.seal();
  CHECK(outer.lifetime().activity == MLX_SUBMISSION_ACTIVITY_TERMINAL);
  CHECK(outer.lifetime().failed);
  CHECK(outer.lifetime().blocked);
  CHECK(effects->destroyed == 0);
  submission::retire_records();
  CHECK(effects->destroyed == 1);
}

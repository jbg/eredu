// Constructor-owned Scope metadata. These tests do not infer completion from
// callback delivery or claim native Record/registry allocations are covered.
#include "doctest/doctest.h"
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include "mlx/c/submission.h"
#include "mlx/submission.h"

using namespace mlx::core;
extern size_t native_recovery_without_allocations(void (*)(void*), void*);
namespace {
void count(void* owner) { ++*static_cast<std::atomic<int>*>(owner); }
// A failed assertion must not leave a native Record with a pointer into a
// departed test stack. An unresolved owner is deliberately retained on failure.
struct Counter {
  std::unique_ptr<std::atomic<int>> value{new std::atomic<int>(0)};
  bool attached{false};
  ~Counter() {
    if (attached && value->load() == 0) (void)value.release();
  }
  operator int() const { return value->load(); }
  void* payload() const { return value.get(); }
};
struct ScopeOwner {
  mlx_submission_scope raw{nullptr};
  explicit ScopeOwner(Counter& retired) {
    REQUIRE(mlx_submission_scope_new_retaining(&raw, retired.payload(), count) == 0);
    retired.attached = true;
  }
  ~ScopeOwner() { if (raw.ctx) mlx_submission_scope_free(raw); }
  void seal() { REQUIRE(mlx_submission_scope_seal(raw) == 0); }
  void release() {
    REQUIRE(mlx_submission_scope_free(raw) == 0);
    raw.ctx = nullptr;
  }
  mlx_submission_record_status records() const {
    mlx_submission_record_status status{};
    REQUIRE(mlx_submission_scope_query_records(&status, raw) == 0);
    return status;
  }
};
struct Effects {
  std::atomic<bool> terminal{false};
  std::atomic<int> observed{0};
  std::atomic<int> destroyed{0};
};
struct Record : submission::Record {
  explicit Record(std::shared_ptr<Effects> e) : effects(std::move(e)) {}
  ~Record() override { ++effects->destroyed; }
  bool side_effects_terminal() const noexcept override {
    ++effects->observed;
    return effects->terminal.load();
  }
  std::shared_ptr<Effects> effects;
};
Record* enter(const std::shared_ptr<Effects>& effects) {
  auto record = std::make_unique<Record>(effects);
  record->enter();
  return record.release();
}
// Observe the allocation boundary after the complete derived/base destructor.
// The thread-local observer is installed only around ordinary synchronous
// retirement, so an unresolved record never retains a pointer into this stack.
struct DeletionProbe {
  Counter& parent;
  Counter& child;
  int destroyed{0};
  int deallocated{0};
  int parent_at_delete{-1};
  int child_at_delete{-1};
};
thread_local DeletionProbe* deletion_probe = nullptr;
struct DeletionRecord final : submission::Record {
  ~DeletionRecord() override {
    if (deletion_probe) ++deletion_probe->destroyed;
  }
  static void operator delete(void* allocation) noexcept {
    auto* probe = deletion_probe;
    ::operator delete(allocation);
    if (probe) {
      ++probe->deallocated;
      probe->parent_at_delete = probe->parent;
      probe->child_at_delete = probe->child;
    }
  }
};
struct ObserveDeletion {
  explicit ObserveDeletion(DeletionProbe& probe) { deletion_probe = &probe; }
  ~ObserveDeletion() { deletion_probe = nullptr; }
};

void settle(const ScopeOwner& scope) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (scope.records().pending && std::chrono::steady_clock::now() < end) {
    submission::progress_records(false);
    std::this_thread::yield();
  }
  REQUIRE_FALSE(scope.records().pending);
}
}

TEST_CASE("submission recovery owned scope rejects invalid handoffs without consumption") {
  Counter retired;
  mlx_submission_scope raw{nullptr};
  CHECK(mlx_submission_scope_new_retaining(nullptr, retired.payload(), count) == 1);
  CHECK(mlx_submission_scope_new_retaining(&raw, nullptr, count) == 1);
  CHECK(mlx_submission_scope_new_retaining(&raw, retired.payload(), nullptr) == 1);
  CHECK(raw.ctx == nullptr);
  CHECK(mlx_submission_scope_new(&raw) == 0); // Existing plain constructor.
  const auto original = raw.ctx;
  CHECK(mlx_submission_scope_new_retaining(&raw, retired.payload(), count) == 1);
  CHECK(raw.ctx == original);
  CHECK(retired == 0);
  CHECK(mlx_submission_scope_free(raw) == 0);
  CHECK(retired == 0);
}

TEST_CASE("submission recovery owned scope native allocation failure preserves owner and ancestry") {
  Counter parent_retired;
  Counter failed_retired;
  ScopeOwner parent(parent_retired); // Initialize registry and owner tag first.
  struct Attempt {
    mlx_submission_scope raw{nullptr};
    void* owner;
    int result{-1};
  } attempt{{nullptr}, failed_retired.payload()};
  const auto attempts = native_recovery_without_allocations([](void* context) {
    auto& a = *static_cast<Attempt*>(context);
    a.result = mlx_submission_scope_new_retaining(&a.raw, a.owner, count);
  }, &attempt);
  CHECK(attempts > 0);
  CHECK(attempt.result == 1);
  CHECK(attempt.raw.ctx == nullptr);
  CHECK(failed_retired == 0);
  CHECK_FALSE(parent.records().pending);
  // The exact original payload can be retried, still nested in the same parent.
  REQUIRE(mlx_submission_scope_new_retaining(&attempt.raw, failed_retired.payload(), count) == 0);
  failed_retired.attached = true;
  parent.seal(); parent.release();
  CHECK(parent_retired == 0);
  CHECK(mlx_submission_scope_free(attempt.raw) == 0);
  CHECK(parent_retired == 1);
  CHECK(failed_retired == 1);
}

TEST_CASE("submission recovery owned parent outlives quiescent sealed plain descendant") {
  Counter retired;
  ScopeOwner parent(retired);
  mlx_submission_scope child{nullptr};
  REQUIRE(mlx_submission_scope_new(&child) == 0);
  parent.seal();
  CHECK_FALSE(parent.records().pending);
  mlx_submission_status status{};
  REQUIRE(mlx_submission_scope_query(&status, parent.raw) == 0);
  CHECK(status.activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  REQUIRE(mlx_submission_scope_seal(child) == 0);
  REQUIRE(mlx_submission_scope_query(&status, parent.raw) == 0);
  CHECK(status.activity == MLX_SUBMISSION_ACTIVITY_NONE);
  parent.release();
  CHECK(retired == 0); // Settled is not final intrusive-reference retirement.
  REQUIRE(mlx_submission_scope_free(child) == 0);
  CHECK(retired == 1);
}

TEST_CASE("submission recovery owned scope retains exact record until ordinary deletion") {
  for (const bool failed : {false, true}) {
    Counter parent_retired;
    Counter child_retired;
    ScopeOwner parent(parent_retired);
    ScopeOwner child(child_retired);
    auto effects = std::make_shared<Effects>();
    auto* record = enter(effects);
    parent.seal(); child.seal();
    CHECK(parent.records().pending);
    CHECK(child.records().pending);
    record->finish(failed);
    CHECK(parent.records().failed == failed);
    effects->terminal = true;
    CHECK(child.records().pending); // Cold query cannot promote this fact.
    CHECK(effects->observed == 0);
    settle(child);
    CHECK_FALSE(parent.records().pending);
    CHECK(parent.records().failed == failed);
    CHECK(effects->destroyed == 0);
    parent.release(); child.release();
    CHECK(parent_retired == 0);
    CHECK(child_retired == 0);
    submission::retire_records();
    CHECK(effects->destroyed == 1);
    CHECK(parent_retired == 1);
    CHECK(child_retired == 1);
  }
}

TEST_CASE("submission recovery owned scope foreign free cannot consume original custody") {
  Counter retired;
  ScopeOwner scope(retired);
  const auto raw = scope.raw;
  const auto result = std::async(std::launch::async, [raw] {
    return mlx_submission_scope_free(raw);
  }).get();
  CHECK(result == 1);
  CHECK(retired == 0);
  scope.release();
  CHECK(retired == 1);
}

TEST_CASE("submission recovery owned scope cold layout and final callback need no allocation") {
  Counter retired;
  ScopeOwner scope(retired);
  struct Call { mlx_submission_scope raw; int result{-1}; size_t scope{0}; size_t controls{0}; } call{scope.raw};
  const auto attempts = native_recovery_without_allocations([](void* context) {
    auto& c = *static_cast<Call*>(context);
    c.scope = mlx_submission_scope_control_bytes();
    c.controls = mlx_submission_scope_retirement_control_bytes();
    c.result = mlx_submission_scope_free(c.raw);
  }, &call);
  scope.raw.ctx = nullptr;
  CHECK(attempts == 0);
  CHECK(call.result == 0);
  CHECK(call.scope > 0);
  CHECK(call.controls > 0);
  CHECK(retired == 1);
}

TEST_CASE("submission recovery owned scope custody survives the complete record allocation") {
  for (const bool failed : {false, true}) {
    Counter parent_retired;
    Counter child_retired;
    ScopeOwner parent(parent_retired);
    ScopeOwner child(child_retired);
    auto prepared = std::make_unique<DeletionRecord>();
    prepared->enter();
    auto* record = prepared.release();
    parent.seal(); child.seal();
    record->finish(failed);
    settle(child);
    CHECK_FALSE(parent.records().pending);
    CHECK(child.records().failed == failed);
    parent.release(); child.release();
    CHECK(parent_retired == 0);
    CHECK(child_retired == 0);
    DeletionProbe probe{parent_retired, child_retired};
    {
      ObserveDeletion observe(probe);
      submission::retire_records();
    }
    CHECK(probe.destroyed == 1);
    CHECK(probe.deallocated == 1);
    CHECK(probe.parent_at_delete == 0);
    CHECK(probe.child_at_delete == 0);
    CHECK(parent_retired == 1);
    CHECK(child_retired == 1);
    submission::retire_records();
    CHECK(parent_retired == 1);
    CHECK(child_retired == 1);
  }
}

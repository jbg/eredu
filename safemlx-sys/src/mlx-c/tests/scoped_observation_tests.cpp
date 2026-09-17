#include <thread>
#include "doctest/doctest.h"
#include "mlx/submission.h"
#include "mlx/failure.h"
#include "mlx/scheduler.h"
#include "mlx/event.h"
#include <atomic>
#include <chrono>
#include <memory>
using namespace mlx::core::submission;
namespace {
struct ScopeOwner {
  Scope* value{new Scope};
  ~ScopeOwner() { value->seal(); value->release(); }
};
struct Counts {
  std::atomic<size_t> observed{0};
  std::atomic<size_t> destroyed{0};
};
struct Observable final : Record {
  Counts& counts;
  Observable(Allocation allocation, Counts& value) : Record(allocation), counts(value) {}
  ~Observable() override { ++counts.destroyed; }
  bool scoped_observation_supported() const noexcept override { return true; }
  bool side_effects_terminal() const noexcept override { ++counts.observed; return true; }
};
struct Unknown final : Record {
  Counts& counts;
  Unknown(Allocation allocation, Counts& value) : Record(allocation), counts(value) {}
  ~Unknown() override { ++counts.destroyed; }
  bool side_effects_terminal() const noexcept override { ++counts.observed; return true; }
};
template<class T>
T* entered(Counts& counts) {
  auto owner=Record::create<T>(counts);
  owner->enter();
  return owner.release();
}

namespace {
struct FailureLifetime {
  std::atomic<size_t> payloads{0};
  std::atomic<size_t> retired{0};
  std::atomic<bool> early{false};
};
struct ExceptionPayload {
  FailureLifetime& lifetime;
  explicit ExceptionPayload(FailureLifetime& value) : lifetime(value) {
    ++lifetime.payloads;
  }
  ~ExceptionPayload() { --lifetime.payloads; }
};
struct OwnedWorkerError final : std::exception {
  std::shared_ptr<ExceptionPayload> payload;
  explicit OwnedWorkerError(FailureLifetime& value)
      : payload(std::make_shared<ExceptionPayload>(value)) {}
  const char* what() const noexcept override { return "actual scoped worker cause"; }
};
void retire_failure(void* raw) {
  auto& value = *static_cast<FailureLifetime*>(raw);
  if (value.payloads.load() != 0) value.early = true;
  ++value.retired;
}
mlx::core::FailureCarrierRef failure_owner(FailureLifetime& value) {
  return mlx::core::FailureCarrier::create(&value, retire_failure);
}
void await_completed(mlx::core::scheduler::StreamThread& worker, uint64_t count) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (worker.progress().completed < count) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
}
}

TEST_CASE("scoped first-failure slot retains its owner through reset and escaped aliases") {
  using namespace mlx::core;
  FailureLifetime first, later;
  scheduler::StreamThread worker;
  auto one = failure_owner(first), two = failure_owner(later);
  REQUIRE(one);
  REQUIRE(two);
  worker.record_error(std::make_exception_ptr(OwnedWorkerError(first)), one);
  worker.record_error(std::make_exception_ptr(OwnedWorkerError(later)), two);
  REQUIRE(one.get()->borrow());
  REQUIRE(two.get()->borrow());
  CHECK(one.get()->borrow()->exception_type == &typeid(OwnedWorkerError));
  CHECK(two.get()->borrow()->exception_type == &typeid(OwnedWorkerError));
  CHECK(worker.first_failure.get() == one.get());
  auto escaped = one;
  one = {};
  two = {};
  CHECK(later.retired == 1);
  CHECK_FALSE(later.early);
  CHECK(first.retired == 0);
  REQUIRE(worker.error(true)); // Explicit ordinary legacy extraction boundary.
  CHECK_FALSE(worker.first_failure);
  CHECK_FALSE(worker.first_error);
  CHECK(first.retired == 0);
  escaped = {};
  CHECK(first.retired == 1);
  CHECK_FALSE(first.early);
}

TEST_CASE("scheduler teardown destroys the first native cause before its final custody") {
  using namespace mlx::core;
  FailureLifetime value;
  {
    scheduler::StreamThread worker;
    auto owner = failure_owner(value);
    REQUIRE(owner);
    worker.record_error(std::make_exception_ptr(OwnedWorkerError(value)), owner);
    owner = {};
    CHECK(value.retired == 0);
    CHECK(value.payloads == 1);
    // No reset: the real StreamThread destructor drains/joins before member
    // retirement. first_error precedes first_failure in destruction order.
  }
  CHECK(value.retired == 1);
  CHECK(value.payloads == 0);
  CHECK_FALSE(value.early);
}

TEST_CASE("failing final queued task retains its cause after its capture retires") {
  using namespace mlx::core;
  FailureLifetime value;
  scheduler::StreamThread worker;
  std::atomic<bool> captured{false};
  auto owner = failure_owner(value);
  REQUIRE(owner);
  {
    ScopeOwner scope;
    REQUIRE(scope.value->enable_scoped_observation());
    REQUIRE(scope.value->bind_failure(owner));
    worker.enqueue([&worker, &value, &captured, owner]() mutable {
      try { throw OwnedWorkerError(value); }
      catch (...) { worker.record_error(std::current_exception(), owner); }
      captured.store(true, std::memory_order_release);
    });
    owner = {};
    // The actual task owns its carrier independently of this scope.
  }
  await_completed(worker, 1);
  CHECK(captured.load(std::memory_order_acquire));
  CHECK(worker.progress().failed);
  CHECK(value.retired == 0);
  CHECK(value.payloads == 1);
  REQUIRE(worker.first_failure.get()->borrow());
  REQUIRE(worker.error(true));
  CHECK(value.retired == 1);
  CHECK(value.payloads == 0);
  CHECK_FALSE(value.early);
}

TEST_CASE("scoped event copy and move replacement keep each displaced carrier alive") {
  using namespace mlx::core;
  const auto stream = default_stream(Device::cpu);
  FailureLifetime first, second;
  Event one, two;
  {
    auto owner = failure_owner(first);
    REQUIRE(owner);
    ScopeOwner scope;
    REQUIRE(scope.value->enable_scoped_observation());
    REQUIRE(scope.value->bind_failure(owner));
    one = Event(stream);
  }
  {
    auto owner = failure_owner(second);
    REQUIRE(owner);
    ScopeOwner scope;
    REQUIRE(scope.value->enable_scoped_observation());
    REQUIRE(scope.value->bind_failure(owner));
    two = Event(stream);
  }
  CHECK(first.retired == 0);
  CHECK(second.retired == 0);
  Event alias = one;
  one = two;
  CHECK(first.retired == 0);
  alias = Event{};
  CHECK(first.retired == 1);
  two = Event{};
  CHECK(second.retired == 0);
  one = Event{};
  CHECK(second.retired == 1);
}
}

TEST_CASE("scoped observation keeps unrelated pending and terminal owners untouched") {
  Counts pending, terminal, selected;
  {
    ScopeOwner first;
    entered<Observable>(terminal)->finish(false);
    first.value->progress();
    REQUIRE(terminal.observed > 0);
    CHECK(terminal.destroyed == 0);
    first.value->seal();
    ScopeOwner other;
    entered<Observable>(pending)->finish(false);
    other.value->seal();
    {
      ScopeOwner target;
      REQUIRE(target.value->enable_scoped_observation());
      entered<Observable>(selected)->finish(false);
      CHECK(target.value->progress_scoped() == ScopedProgress::observed);
      CHECK(selected.observed > 0);
      CHECK(pending.observed == 0);
      CHECK(pending.destroyed == 0);
      CHECK(terminal.destroyed == 0);
      CHECK(try_retire_records() == RetirementPass::busy);
      CHECK(terminal.destroyed == 0);
      progress_records();
      retire_records();
      CHECK(pending.observed == 0);
      CHECK(terminal.destroyed == 0);
    }
  }
  progress_records();
  CHECK(try_retire_records() == RetirementPass::complete_snapshot);
  CHECK(pending.destroyed == 1);
  CHECK(terminal.destroyed == 1);
  CHECK(selected.destroyed == 1);
}

TEST_CASE("scoped observation includes genuine descendants and refuses unknown callbacks") {
  Counts child, unknown;
  {
    ScopeOwner parent;
    REQUIRE(parent.value->enable_scoped_observation());
    const auto identity=parent.value->identity();
    REQUIRE(identity != 0);
    {
      ScopeOwner nested;
      CHECK(nested.value->scoped_observation_only());
      CHECK(nested.value->identity() != identity);
      entered<Observable>(child)->finish(false);
    }
    CHECK(parent.value->progress_scoped() == ScopedProgress::observed);
    CHECK(child.observed > 0);
    CHECK(child.destroyed == 0);
    entered<Unknown>(unknown)->finish(false);
    CHECK(parent.value->progress_scoped() == ScopedProgress::unobservable);
    CHECK(unknown.observed == 0);
    CHECK(unknown.destroyed == 0);
    CHECK(parent.value->query().activity == Activity::pending);
  }
  progress_records();
  CHECK(try_retire_records() == RetirementPass::complete_snapshot);
  CHECK(child.destroyed == 1);
  CHECK(unknown.destroyed == 1);
}

TEST_CASE("scoped observation cannot adopt work accepted before its mode") {
  Counts existing;
  {
    ScopeOwner scope;
    entered<Observable>(existing)->finish(false);
    CHECK_FALSE(scope.value->enable_scoped_observation());
    CHECK_FALSE(scope.value->scoped_observation_only());
    CHECK(scope.value->progress_scoped() == ScopedProgress::unobservable);
    CHECK(existing.observed == 0);
  }
  progress_records();
  CHECK(try_retire_records() == RetirementPass::complete_snapshot);
  CHECK(existing.destroyed == 1);
}

TEST_CASE("scoped record throttle observes only its own accepted CPU frontiers") {
  using namespace mlx::core;
  Counts selected;
  const auto stream = new_stream(Device::cpu);
  std::atomic<bool> completed{false};
  {
    ScopeOwner scope;
    REQUIRE(scope.value->enable_scoped_observation());
    auto* record = entered<Observable>(selected);
    record->prepare_stream(stream);
    scheduler::enqueue(stream, [&] { completed.store(true, std::memory_order_release); });
    CHECK(record->wait_for_scoped_progress() == ScopedProgress::observed);
    CHECK(completed.load(std::memory_order_acquire));
    CHECK(selected.destroyed == 0);
    record->finish(false);
    CHECK(scope.value->progress_scoped() == ScopedProgress::observed);
  }
  progress_records();
  CHECK(try_retire_records() == RetirementPass::complete_snapshot);
  CHECK(selected.destroyed == 1);
}

TEST_CASE("scoped record throttle returns actual failed frontier without throwing") {
  using namespace mlx::core;
  Counts selected;
  const auto stream = new_stream(Device::cpu);
  {
    ScopeOwner scope;
    REQUIRE(scope.value->enable_scoped_observation());
    auto* record = entered<Observable>(selected);
    record->prepare_stream(stream);
    scheduler::enqueue(stream, [] { throw std::runtime_error("scoped worker cause"); });
    CHECK(record->wait_for_scoped_progress() == ScopedProgress::failed);
    CHECK(selected.destroyed == 0);
    record->finish(true);
    // Failure alone is not completion; publish actual terminal task evidence.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scope.value->query_records().pending) {
      REQUIRE(std::chrono::steady_clock::now() < deadline);
      CHECK(scope.value->progress_scoped() == ScopedProgress::observed);
      std::this_thread::yield();
    }
    CHECK(scope.value->query().failed);
  }
  progress_records();
  CHECK(try_retire_records() == RetirementPass::complete_snapshot);
  CHECK(selected.destroyed == 1);
}

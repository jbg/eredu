#include "doctest/doctest.h"
#include "mlx/c/submission.h"
#include "mlx/c/error.h"
#include "mlx/event.h"
#include "mlx/failure.h"
#include "mlx/graph_quota.h"
#include "mlx/record_quota.h"
#include "mlx/scheduler.h"
#include "mlx/submission.h"
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// Test-only access to the native shared control. Production exports no weak
// Event handle; the test observes the allocator's actual final-header lifetime.
namespace mlx::core {
struct EventControlTestAccess {
  static std::weak_ptr<void> weak(const Event& value) { return value.event_; }
};
}
namespace mlx::core::scheduler {
struct SchedulerControlTestAccess {
  static bool stopping(const Scheduler& value) {
    return value.stopping_.load(std::memory_order_acquire);
  }
  static bool registry_unlocked(Scheduler& value) {
    std::unique_lock lock(value.threads_mtx_, std::try_to_lock);
    return lock.owns_lock();
  }
};
}
using namespace mlx::core;
using namespace mlx::core::submission;
namespace {
using Failure = NativeControlFailure;
struct Counts { std::atomic<size_t> retired{0}; };
void retired(void* raw) { ++static_cast<Counts*>(raw)->retired; }
struct Graph {
  GraphQuota* value;
  Graph(size_t capacity, Counts& count)
      : value(GraphQuota::create(capacity, &count, retired)) {}
  ~Graph() { reset(); }
  void reset() { if (auto* old = std::exchange(value, nullptr)) old->release(); }
};
struct Records {
  // This is an explicit finite fixture ceiling, not a required-fit claim.
  RecordQuota* value{RecordQuota::create(1 << 20, nullptr, nullptr)};
  ~Records() { value->release(); }
};
struct Enter {
  Scope* value;
  Enter(GraphQuota* graph = nullptr, RecordQuota* record = nullptr, Counts* count = nullptr)
      : value(new Scope(count, count ? retired : nullptr, record, graph)) {}
  ~Enter() { reset(); }
  void reset() {
    if (auto* old = std::exchange(value, nullptr)) { old->seal(); old->release(); }
  }
};
FailureCarrierRef carrier(Counts& count) {
  return FailureCarrier::create(&count, retired);
}
void configure(Scope& scope, const FailureCarrierRef& failure) {
  REQUIRE(scope.enable_scoped_observation());
  REQUIRE(scope.require_original_controls() == Failure::none);
  REQUIRE(scope.bind_failure(failure));
  REQUIRE(scope.enable_original_controls() == Failure::none);
}
template<class F> void refuses(F&& body, Failure expected) {
  try { body(); FAIL_CHECK("original operation unexpectedly succeeded"); }
  catch (const NativeControlError& error) { CHECK(error.failure() == expected); }
}
struct Observed final : Record {
  explicit Observed(Allocation allocation) : Record(allocation) {}
  bool scoped_observation_supported() const noexcept override { return true; }
};
Observed* enter_record() {
  auto pending = Record::create<Observed>();
  pending->enter();
  return pending.release();
}
void settle(Scope& scope) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (scope.query_records().pending) {
    REQUIRE(scope.progress_scoped() == ScopedProgress::observed);
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
}
void retire_registry() {
  progress_records();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (try_retire_records() == RetirementPass::busy) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
}
Stream initialized_cpu() {
  auto stream = default_stream(Device::cpu);
  mlx_submission_runtime_baseline baseline{};
  REQUIRE(mlx_submission_prepare_runtime(&baseline, mlx_stream{&stream},
      mlx_stream{&stream}) == 0);
  REQUIRE(baseline.cpu_workers == 1);
  return stream;
}

void await(scheduler::CpuStreamToken worker, uint64_t count) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (scheduler::cpu_stream_progress(worker).completed < count) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
}
}

TEST_CASE("original control mode refuses incomplete or spent scope without ordinary fallback") {
  const auto stream = initialized_cpu();
  Counts graph_count, failure_count;
  Graph graph(1 << 20, graph_count);
  Records records;
  auto failure = carrier(failure_count);
  {
    Enter scope(nullptr, records.value);
    REQUIRE(scope.value->enable_scoped_observation());
    CHECK(scope.value->enable_original_controls() == Failure::missing_graph);
    refuses([&] { Event event(stream); }, Failure::invalid_scope);
    CHECK(scope.value->enable_original_controls() == Failure::already_configured);
  }
  {
    Enter scope(graph.value);
    REQUIRE(scope.value->enable_scoped_observation());
    CHECK(scope.value->enable_original_controls() == Failure::missing_record);
    refuses([&] { scheduler::enqueue(stream, [] {}); }, Failure::invalid_scope);
    CHECK(graph.value->occupied_bytes() == 0);
  }
  {
    Enter scope(graph.value, records.value);
    REQUIRE(scope.value->enable_scoped_observation());
    CHECK(scope.value->enable_original_controls() == Failure::missing_failure);
    refuses([&] { Event event(stream); }, Failure::invalid_scope);
  }
  {
    Enter scope(graph.value, records.value);
    configure(*scope.value, failure);
    CHECK(scope.value->enable_original_controls() == Failure::already_configured);
    Event event(stream);
    CHECK(graph.value->occupied_bytes() > 0);
    CHECK_FALSE(scope.value->bind_failure(carrier(failure_count)));
    scope.value->seal();
    CHECK(scope.value->enable_original_controls() == Failure::invalid_scope);
  }
  CHECK(graph.value->occupied_bytes() == 0);
  {
    // Existing C2 Graph-only diagnostics deliberately remain ordinary.
    Enter scope(graph.value);
    auto ordinary_failure = carrier(failure_count);
    REQUIRE(scope.value->enable_scoped_observation());
    REQUIRE(scope.value->bind_failure(ordinary_failure));
    Event event(stream);
    CHECK_FALSE(current_native_controls().original);
    CHECK(graph.value->occupied_bytes() == 0);
    CHECK(scope.value->require_original_controls() == Failure::invalid_scope);
    CHECK(scope.value->enable_original_controls() == Failure::invalid_scope);
  }
  {
    Enter scope(graph.value, records.value);
    auto* record = enter_record();
    CHECK(scope.value->require_original_controls() == Failure::invalid_scope);
    record->finish(false);
  }
  retire_registry();
  CHECK(graph.value->occupied_bytes() == 0);
}

TEST_CASE("original child binds only its own fresh carrier before work and isolates causes") {
  const auto stream = initialized_cpu();
  Counts graph_count, parent_count, child_count, rejected_count, parent_scope, child_scope;
  Graph graph(1 << 20, graph_count);
  Records records;
  auto parent_failure = carrier(parent_count), child_failure = carrier(child_count);
  Event parent_event, child_event;
  uint64_t parent_identity = 0, child_identity = 0;
  {
    Enter parent(graph.value, records.value, &parent_scope);
    configure(*parent.value, parent_failure);
    parent_identity = parent.value->identity();
    parent_event = Event(stream);
    {
      Enter child(graph.value, records.value, &child_scope);
      child_identity = child.value->identity();
      CHECK(child_identity != parent_identity);
      CHECK(child.value->original_controls_required());
      CHECK_FALSE(child.value->original_controls_enabled());
      refuses([&] { Event event(stream); }, Failure::invalid_scope);
      REQUIRE(child.value->bind_failure(child_failure));
      CHECK_FALSE(child.value->bind_failure(carrier(rejected_count)));
      CHECK(rejected_count.retired == 1);
      REQUIRE(child.value->enable_original_controls() == Failure::none);
      child_event = Event(stream);
      child_failure.get()->capture_exception(
          std::make_exception_ptr(std::range_error("child native cause")));
      CHECK(child_event.observe() == EventObservation::failed);
      CHECK_FALSE(parent_failure.get()->borrow());
      CHECK(parent_event.observe() == EventObservation::ready);
      CHECK(parent_failure.get()->scope_identity() == parent_identity);
      CHECK(child_failure.get()->scope_identity() == child_identity);
      parent.reset(); // Durable child ancestry outlives the parent's host guard.
      CHECK(parent_scope.retired == 0);
      CHECK(child_scope.retired == 0);
    }
    CHECK(parent_scope.retired == 1);
    CHECK(child_scope.retired == 1);
  }
  parent_failure = {};
  child_failure = {};
  CHECK(parent_count.retired == 0);
  CHECK(child_count.retired == 0);
  child_event = Event{};
  CHECK(child_count.retired == 1);
  CHECK(parent_count.retired == 0);
  parent_event = Event{};
  CHECK(parent_count.retired == 1);
  CHECK(graph.value->occupied_bytes() == 0);

  Counts other_count;
  auto other = carrier(other_count);
  {
    Enter parent(graph.value, records.value);
    configure(*parent.value, other);
    Enter child(graph.value, records.value);
    // An inherited binding is not this child's own binding capability.
    CHECK(child.value->enable_original_controls() == Failure::foreign_domain);
    CHECK_FALSE(child.value->bind_failure(carrier(child_count)));
    refuses([&] { Event event(stream); }, Failure::invalid_scope);
  }
}

TEST_CASE("original Event object and shared header use one arena through final weak alias") {
  const auto stream = initialized_cpu();
  Counts graph_count, failure_count;
  Graph graph(1 << 20, graph_count);
  Records records;
  EventControlLayout layout{};
  REQUIRE(event_control_layout(layout));
  CHECK(layout.shared_bytes >= layout.object_bytes);
  CHECK(layout.graph_extent >= layout.shared_bytes);
  Event event;
  std::weak_ptr<void> weak;
  {
    Enter scope(graph.value, records.value);
    configure(*scope.value, carrier(failure_count));
    event = Event(stream);
    weak = EventControlTestAccess::weak(event);
    CHECK(graph.value->occupied_bytes() > 0);
    CHECK(graph.value->occupied_bytes() <= layout.graph_extent);
  }
  Event alias = event;
  graph.reset();
  CHECK(graph_count.retired == 0);
  std::thread first([value = std::move(event)]() mutable { value = Event{}; });
  std::thread second([value = std::move(alias)]() mutable { value = Event{}; });
  first.join(); second.join();
  CHECK(weak.expired());
  CHECK(failure_count.retired == 1);
  CHECK(graph_count.retired == 0); // allocate_shared allocator remains in header.
  weak.reset();
  CHECK(graph_count.retired == 1);
}

TEST_CASE("original Event refusal reflects occupied arena and preserves the first owner") {
  const auto stream = initialized_cpu();
  Counts graph_count, first_count, second_count;
  Graph graph(1 << 20, graph_count);
  Records records;
  Event held;
  std::vector<void*> fill;
  {
    Enter scope(graph.value, records.value);
    configure(*scope.value, carrier(first_count));
    held = Event(stream);
    while (auto* value = graph.value->try_allocate(1, 1)) fill.push_back(value);
    const auto occupied = graph.value->occupied_bytes();
    CHECK_THROWS_AS(Event{stream}, GraphQuotaError);
    CHECK(graph.value->occupied_bytes() == occupied);
    CHECK(held.valid());
  }
  for (auto* value : fill) graph.value->deallocate(value, 1, 1);
  held = Event{};
  CHECK(graph.value->occupied_bytes() == 0);
  CHECK(first_count.retired == 1);
  // A fresh same-content event in another domain has independent custody.
  Counts foreign_count;
  Graph foreign(1 << 20, foreign_count);
  {
    Enter scope(foreign.value, records.value);
    configure(*scope.value, carrier(second_count));
    held = Event(stream);
  }
  foreign.reset();
  CHECK(foreign_count.retired == 0);
  held = Event{};
  CHECK(foreign_count.retired == 1);
  CHECK(second_count.retired == 1);
}

TEST_CASE("original task dispatch requires exact entered receipt and restores nested guards") {
  const auto stream = initialized_cpu();
  const auto other = new_stream(Device::cpu);
  (void)scheduler::prepare_cpu_stream(other);
  Counts graph_count, failure_count;
  Graph graph(1 << 20, graph_count);
  Records records;
  auto failure = carrier(failure_count);
  {
    Enter scope(graph.value, records.value);
    configure(*scope.value, failure);
    auto* first = enter_record();
    first->prepare_stream(stream);
    auto* second = enter_record();
    second->prepare_stream(other);
    const auto occupied = graph.value->occupied_bytes();
    refuses([&] { scheduler::enqueue(stream, [] {}); }, Failure::missing_receipt);
    CHECK(graph.value->occupied_bytes() == occupied);
    scheduler::CpuStreamToken receipt;
    {
      RecordDispatchGuard outer(*first);
      REQUIRE(current_dispatch_stream(stream, receipt) == Failure::none);
      CHECK(current_dispatch_stream(other, receipt) == Failure::missing_receipt);
      try {
        RecordDispatchGuard nested(*second);
        CHECK(current_dispatch_stream(stream, receipt) == Failure::missing_receipt);
        REQUIRE(current_dispatch_stream(other, receipt) == Failure::none);
        throw 19;
      } catch (int value) { CHECK(value == 19); }
      REQUIRE(current_dispatch_stream(stream, receipt) == Failure::none);
      {
        Enter nested(graph.value, records.value);
        auto own = carrier(failure_count);
        REQUIRE(nested.value->bind_failure(own));
        REQUIRE(nested.value->enable_original_controls() == Failure::none);
        // Lexical Record restoration cannot authorize a foreign child Scope.
        refuses([&] { scheduler::enqueue(stream, [] {}); }, Failure::foreign_domain);
        refuses([&] { RecordDispatchGuard wrong(*first); }, Failure::foreign_domain);
      }
      REQUIRE(current_dispatch_stream(stream, receipt) == Failure::none);
      scheduler::enqueue(stream, [] {});
    }
    CHECK(current_dispatch_stream(stream, receipt) == Failure::missing_receipt);
    first->finish(false); second->finish(false);
    scope.value->seal();
    settle(*scope.value);
  }
  retire_registry();
  CHECK(graph.value->occupied_bytes() == 0);
}

namespace {
struct Gate {
  std::atomic<bool> open{false};
  std::atomic<bool> entered{false};
  void wait() {
    entered.store(true, std::memory_order_release);
    while (!open.load(std::memory_order_acquire)) std::this_thread::yield();
  }
  ~Gate() { open.store(true, std::memory_order_release); }
};
struct BigTask {
  std::array<uint64_t, 128> inline_payload{};
  std::array<int, 3>* order;
  std::atomic<size_t>* next;
  std::atomic<size_t>* destroyed;
  bool armed{true};
  BigTask(std::array<int, 3>& order, std::atomic<size_t>& next,
          std::atomic<size_t>& destroyed)
      : order(&order), next(&next), destroyed(&destroyed) { inline_payload[127] = 37; }
  BigTask(BigTask&& other) noexcept
      : inline_payload(other.inline_payload), order(other.order), next(other.next),
        destroyed(other.destroyed), armed(std::exchange(other.armed, false)) {}
  ~BigTask() { if (armed) ++*destroyed; }
  void operator()() {
    (*order)[next->fetch_add(1)] = static_cast<int>(inline_payload[127]);
  }
};
struct ThrowingMove {
  explicit ThrowingMove(std::atomic<size_t>& moves) : moves(moves) {}
  std::atomic<size_t>& moves;
  ThrowingMove(ThrowingMove&& other) : moves(other.moves) {
    ++moves;
    throw std::range_error("concrete task move failure");
  }
  void operator()() {}
};
}

TEST_CASE("one FIFO preserves ordinary original ordinary order and retires inline task before completion") {
  const auto stream = initialized_cpu();
  const auto worker = scheduler::prepare_cpu_stream(stream);
  const auto before = scheduler::cpu_stream_progress(worker).accepted;
  Counts graph_count, failure_count;
  Graph graph(1 << 20, graph_count);
  Records records;
  Gate gate;
  std::array<int, 3> order{};
  std::atomic<size_t> next{0}, destroyed{0};
  scheduler::enqueue(stream, [&] { gate.wait(); order[next.fetch_add(1)] = 11; });
  {
    Enter scope(graph.value, records.value);
    configure(*scope.value, carrier(failure_count));
    auto* record = enter_record();
    record->prepare_stream(stream);
    const auto initial = graph.value->occupied_bytes();
    size_t extent = 0;
    REQUIRE(scheduler::task_graph_extent<BigTask>(extent));
    CHECK(extent >= sizeof(scheduler::TaskNode<BigTask>));
    {
      RecordDispatchGuard dispatch(*record);
      scheduler::enqueue_counted(stream, BigTask(order, next, destroyed));
    }
    CHECK(graph.value->occupied_bytes() > initial);
    CHECK(graph.value->occupied_bytes() <= initial + extent);
    CHECK(destroyed == 0);
    CHECK(scheduler::scheduler().n_active_tasks() > 0);
    record->finish(false);
    scope.value->seal();
    scheduler::enqueue(stream, [&] { order[next.fetch_add(1)] = 59; });
    CHECK(scheduler::cpu_stream_progress(worker).accepted == before + 3);
    gate.open.store(true, std::memory_order_release);
    await(worker, before + 3);
    CHECK(destroyed == 1);
    CHECK(scheduler::scheduler().n_active_tasks() == 0);
    CHECK((order == std::array<int, 3>{11, 37, 59}));
    settle(*scope.value);
  }
  retire_registry();
  CHECK(graph.value->occupied_bytes() == 0);
  CHECK(failure_count.retired == 1);
}

TEST_CASE("original task construction refusal preserves accepted frontier and returns its actual block") {
  const auto stream = initialized_cpu();
  const auto worker = scheduler::prepare_cpu_stream(stream);
  Counts graph_count, failure_count;
  Graph graph(1 << 20, graph_count);
  Records records;
  std::atomic<size_t> moves{0};
  {
    Enter scope(graph.value, records.value);
    configure(*scope.value, carrier(failure_count));
    auto* record = enter_record();
    record->prepare_stream(stream);
    const auto before = scheduler::cpu_stream_progress(worker).accepted;
    const auto occupied = graph.value->occupied_bytes();
    {
      RecordDispatchGuard dispatch(*record);
      CHECK_THROWS_AS(scheduler::enqueue(stream, ThrowingMove(moves)), std::range_error);
      CHECK(moves == 1);
      CHECK(graph.value->occupied_bytes() == occupied);
      CHECK(scheduler::cpu_stream_progress(worker).accepted == before);
      std::vector<void*> filler;
      while (auto* value = graph.value->try_allocate(1, 1)) filler.push_back(value);
      CHECK_THROWS_AS(scheduler::enqueue(stream, ThrowingMove(moves)), GraphQuotaError);
      CHECK(moves == 1); // Refusal preceded concrete callable construction.
      CHECK(scheduler::cpu_stream_progress(worker).accepted == before);
      for (auto* value : filler) graph.value->deallocate(value, 1, 1);
    }
    record->finish(false);
    scope.value->seal();
    settle(*scope.value);
  }
  retire_registry();
  CHECK(graph.value->occupied_bytes() == 0);
}

TEST_CASE("original Event and task native layout report is concrete and has no lazy device work") {
  mlx_submission_native_control_layout c{};
  REQUIRE(mlx_submission_native_control_layout_for(&c) == 0);
  EventControlLayout event{};
  REQUIRE(event_control_layout(event));
  CHECK(c.event_object_bytes == event.object_bytes);
  CHECK(c.event_shared_bytes == event.shared_bytes);
  CHECK(c.event_shared_alignment == event.shared_alignment);
  CHECK(c.event_graph_extent == event.graph_extent);
  CHECK(c.event_platform_objects == event.platform_events);
  CHECK(c.task_header_bytes == sizeof(scheduler::Task));
  CHECK(c.task_header_alignment == alignof(scheduler::Task));
  CHECK(c.fixed_controls >= event.construction_controls);
  size_t minimum = 0, maximum = 0;
  REQUIRE(GraphQuota::minimum_allocation_extent(
      sizeof(scheduler::TaskNode<BigTask>), alignof(scheduler::TaskNode<BigTask>), minimum));
  REQUIRE(scheduler::task_graph_extent<BigTask>(maximum));
  CHECK(minimum <= maximum);
}

namespace {
struct TaskCounts {
  std::atomic<size_t> ran{0}, dropped{0}, unlocked{0};
};
struct TrackedTask {
  scheduler::StreamThread* worker;
  TaskCounts* counts;
  bool block_on_move;
  bool throw_on_call;
  bool armed{true};
  TrackedTask(scheduler::StreamThread& worker, TaskCounts& counts,
              bool block = false, bool fail = false)
      : worker(&worker), counts(&counts), block_on_move(block), throw_on_call(fail) {}
  TrackedTask(TrackedTask&& other) noexcept
      : worker(other.worker), counts(other.counts), block_on_move(other.block_on_move),
        throw_on_call(other.throw_on_call), armed(std::exchange(other.armed, false)) {
    if (block_on_move) worker->block();
  }
  ~TrackedTask() {
    if (armed) {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      std::unique_lock lock(worker->mtx, std::try_to_lock);
      while (!lock.owns_lock() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
        lock.try_lock();
      }
      if (lock.owns_lock()) ++counts->unlocked;
      ++counts->dropped;
    }
  }
  void operator()() {
    ++counts->ran;
    if (throw_on_call) throw std::range_error("actual typed queued task failure");
  }
};
scheduler::StreamThread& worker_for(scheduler::CpuStreamToken token) {
  return *const_cast<scheduler::StreamThread*>(
      static_cast<const scheduler::StreamThread*>(token.state));
}
}

TEST_CASE("original enqueue refusal destroys concrete captures outside the queue mutex") {
  (void)initialized_cpu();
  const auto stream = new_stream(Device::cpu);
  const auto token = scheduler::prepare_cpu_stream(stream);
  auto& worker = worker_for(token);
  Counts graph_count, failure_count;
  Graph graph(1 << 20, graph_count);
  Records records;
  TaskCounts counts;
  {
    Enter scope(graph.value, records.value);
    configure(*scope.value, carrier(failure_count));
    auto* record = enter_record();
    record->prepare_stream(stream);
    const auto occupied = graph.value->occupied_bytes();
    const auto before = worker.progress().accepted;
    {
      RecordDispatchGuard dispatch(*record);
      // The concrete move blocks after exact receipt validation but before
      // queue acceptance, exercising the post-construction refusal edge.
      refuses([&] { scheduler::enqueue_counted(stream, TrackedTask(worker, counts, true)); },
              Failure::stopped_worker);
    }
    CHECK(worker.progress().accepted == before);
    CHECK(counts.ran == 0);
    CHECK(counts.dropped == 1);
    CHECK(counts.unlocked == 1);
    CHECK(graph.value->occupied_bytes() == occupied);
    CHECK(scheduler::scheduler().n_active_tasks() == 0);
    record->finish(false);
    scope.value->seal();
    settle(*scope.value);
  }
  retire_registry();
  CHECK(graph.value->occupied_bytes() == 0);
}

TEST_CASE("original accepted task failure keeps its native source through final capture and reset") {
  (void)initialized_cpu();
  const auto stream = new_stream(Device::cpu);
  const auto token = scheduler::prepare_cpu_stream(stream);
  auto& worker = worker_for(token);
  Counts graph_count, failure_count;
  Graph graph(1 << 20, graph_count);
  Records records;
  TaskCounts counts;
  auto failure = carrier(failure_count);
  FailureCarrierRef escaped;
  {
    Enter scope(graph.value, records.value);
    configure(*scope.value, failure);
    auto* record = enter_record();
    record->prepare_stream(stream);
    const auto before = worker.progress().accepted;
    {
      RecordDispatchGuard dispatch(*record);
      scheduler::enqueue_counted(stream, TrackedTask(worker, counts, false, true));
    }
    record->finish(false);
    scope.value->seal();
    await(token, before + 1);
    CHECK(counts.ran == 1);
    CHECK(counts.dropped == 1);
    CHECK(counts.unlocked == 1);
    CHECK(scheduler::scheduler().n_active_tasks() == 0);
    REQUIRE(failure.get()->borrow());
    CHECK(failure.get()->borrow()->exception_type == &typeid(std::range_error));
    CHECK(worker.first_failure.get() == failure.get());
    escaped = failure;
    failure = {};
    settle(*scope.value);
  }
  retire_registry();
  CHECK(graph.value->occupied_bytes() == 0);
  CHECK(failure_count.retired == 0);
  REQUIRE(worker.error(true)); // Ordinary diagnostic extraction after settlement.
  CHECK_FALSE(worker.first_failure);
  CHECK(failure_count.retired == 0);
  REQUIRE(escaped.get()->borrow());
  CHECK(escaped.get()->borrow()->exception_type == &typeid(std::range_error));
  escaped = {};
  CHECK(failure_count.retired == 1);
}

TEST_CASE("original task worker lookup is fixed for absent foreign failed and exhausted streams") {
  const auto stream = initialized_cpu();
  scheduler::CpuStreamToken result;
  CHECK(scheduler::prepared_cpu_stream(Stream(0x3fffffff, Device::cpu), result) ==
        Failure::missing_worker);
  CHECK(scheduler::prepared_cpu_stream(Stream(stream.index, Device::gpu), result) ==
        Failure::foreign_domain);
  const auto isolated = new_stream(Device::cpu);
  const auto token = scheduler::prepare_cpu_stream(isolated);
  auto& worker = worker_for(token);
  const auto before = worker.admission.load(std::memory_order_acquire);
  worker.admission.store(scheduler::StreamThread::sequence_mask, std::memory_order_release);
  CHECK(scheduler::prepared_cpu_stream(isolated, result) == Failure::sequence_exhausted);
  worker.admission.store(before, std::memory_order_release);
  worker.block();
  CHECK(scheduler::prepared_cpu_stream(isolated, result) == Failure::stopped_worker);
  // This stream is permanently failed and is never reused by another case.
  worker.record_error(std::make_exception_ptr(std::range_error("lookup failure")));
  CHECK(scheduler::prepared_cpu_stream(isolated, result) == Failure::failed_worker);
  REQUIRE(worker.error(true));
}

TEST_CASE("original enqueue block race has one acceptance and no counted or destruction leak") {
  (void)initialized_cpu();
  const auto stream = new_stream(Device::cpu);
  const auto token = scheduler::prepare_cpu_stream(stream);
  auto& worker = worker_for(token);
  Counts graph_count, failure_count;
  Graph graph(1 << 20, graph_count);
  Records records;
  for (size_t iteration = 0; iteration < 32; ++iteration) {
    std::atomic<bool> go{false};
    TaskCounts counts;
    const auto before = worker.progress().accepted;
    struct JoinBlocker {
      std::atomic<bool>& go;
      std::thread worker;
      ~JoinBlocker() {
        go.store(true, std::memory_order_release);
        if (worker.joinable()) worker.join();
      }
    } blocker{go, std::thread([&] {
      while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
      worker.block();
    })};
    {
      Enter scope(graph.value, records.value);
      configure(*scope.value, carrier(failure_count));
      auto* record = enter_record();
      record->prepare_stream(stream);
      bool accepted = false;
      {
        RecordDispatchGuard dispatch(*record);
        go.store(true, std::memory_order_release);
        try {
          scheduler::enqueue_counted(stream, TrackedTask(worker, counts));
          accepted = true;
        } catch (const NativeControlError& error) {
          CHECK((error.failure() == Failure::stopped_worker ||
                 error.failure() == Failure::blocked_during_submit));
        }
      }
      blocker.worker.join();
      const auto after = worker.progress().accepted;
      CHECK(after == before + size_t(accepted));
      await(token, after);
      CHECK(counts.ran == size_t(accepted));
      CHECK(counts.dropped == 1);
      CHECK(counts.unlocked == 1);
      CHECK(scheduler::scheduler().n_active_tasks() == 0);
      // Test-only reset after join and exact accepted frontier completion.
      worker.admission.fetch_and(scheduler::StreamThread::sequence_mask,
                                 std::memory_order_acq_rel);
      record->finish(false);
      scope.value->seal();
      settle(*scope.value);
    }
    retire_registry();
    CHECK(graph.value->occupied_bytes() == 0);
  }
  CHECK(failure_count.retired == 32);
}

TEST_CASE("original Event signals and waits its nonzero value after the host guard is sealed") {
  const auto stream = initialized_cpu();
  Counts graph_count, failure_count;
  Graph graph(1 << 20, graph_count);
  Records records;
  Event escaped;
  {
    Enter scope(graph.value, records.value);
    configure(*scope.value, carrier(failure_count));
    escaped = Event(stream);
    escaped.set_value(73);
    CHECK(escaped.observe() == EventObservation::pending);
    auto* record = enter_record();
    record->prepare_stream(stream);
    {
      RecordDispatchGuard dispatch(*record);
      escaped.signal(stream);
      escaped.wait(stream);
    }
    record->finish(false);
    scope.value->seal();
    settle(*scope.value);
    CHECK(escaped.observe() == EventObservation::ready);
  }
  retire_registry();
  CHECK(failure_count.retired == 0);
  graph.reset();
  CHECK(graph_count.retired == 0);
  CHECK(escaped.observe() == EventObservation::ready);
  escaped = Event{};
  CHECK(failure_count.retired == 1);
  CHECK(graph_count.retired == 1);
}

TEST_CASE("original typed task uses proven minimum and maximum block extents for quota refusal") {
  const auto stream = initialized_cpu();
  const auto worker = scheduler::prepare_cpu_stream(stream);
  using Node = scheduler::TaskNode<BigTask>;
  size_t minimum = 0, maximum = 0;
  REQUIRE(GraphQuota::minimum_allocation_extent(sizeof(Node), alignof(Node), minimum));
  REQUIRE(scheduler::task_graph_extent<BigTask>(maximum));
  REQUIRE(minimum > 1);
  GraphQuotaLayout checked{};
  REQUIRE(GraphQuota::layout(minimum - 1, checked));
  for (const bool enough : {false, true}) {
    Counts graph_count, failure_count;
    Graph graph(enough ? maximum : minimum - 1, graph_count);
    Records records;
    std::array<int, 3> output{};
    std::atomic<size_t> next{0}, destroyed{0};
    {
      Enter scope(graph.value, records.value);
      configure(*scope.value, carrier(failure_count));
      auto* record = enter_record();
      record->prepare_stream(stream);
      REQUIRE(graph.value->occupied_bytes() == 0);
      const auto before = scheduler::cpu_stream_progress(worker).accepted;
      {
        RecordDispatchGuard dispatch(*record);
        if (enough) {
          scheduler::enqueue(stream, BigTask(output, next, destroyed));
        } else {
          CHECK_THROWS_AS(scheduler::enqueue(stream, BigTask(output, next, destroyed)),
                          GraphQuotaError);
          CHECK(scheduler::cpu_stream_progress(worker).accepted == before);
        }
      }
      record->finish(false);
      scope.value->seal();
      await(worker, before + size_t(enough));
      CHECK(next == size_t(enough));
      if (enough) CHECK(output[0] == 37);
      CHECK(destroyed == 1);
      settle(*scope.value);
    }
    retire_registry();
    CHECK(graph.value->occupied_bytes() == 0);
    CHECK(failure_count.retired == 1);
  }
}

TEST_CASE("selected runtime first use prepares exact workers without Event or task warmup") {
  auto operation = new_stream(Device::cpu);
  auto weights = new_stream(Device::cpu);
  scheduler::CpuStreamToken first, second;
  CHECK(scheduler::prepared_cpu_stream(operation, first) == Failure::missing_worker);
  CHECK(scheduler::prepared_cpu_stream(weights, second) == Failure::missing_worker);
  mlx_submission_runtime_baseline baseline{};
  REQUIRE(mlx_submission_prepare_runtime(&baseline, mlx_stream{&operation},
      mlx_stream{&weights}) == 0);
  REQUIRE(scheduler::prepared_cpu_stream(operation, first) == Failure::none);
  REQUIRE(scheduler::prepared_cpu_stream(weights, second) == Failure::none);
  CHECK(first.state != second.state);
  CHECK(baseline.selected_streams == 2);
  CHECK(baseline.cpu_workers == 2);
  CHECK(baseline.native_threads == 2);
  CHECK(baseline.scheduler_object_bytes == sizeof(scheduler::Scheduler));
  CHECK(baseline.worker_object_bytes == 2 * sizeof(scheduler::StreamThread));
  CHECK(baseline.event_runtime_object_bytes == event_runtime_object_bytes());
  CHECK((baseline.unpriced_populations & 7) == 7);
  CHECK(scheduler::cpu_stream_progress(first).accepted == 0);
  CHECK(scheduler::cpu_stream_progress(second).accepted == 0);
  REQUIRE(mlx_submission_prepare_runtime(&baseline, mlx_stream{&operation},
      mlx_stream{&operation}) == 0);
  CHECK(baseline.selected_streams == 1);
  CHECK(baseline.worker_object_bytes == sizeof(scheduler::StreamThread));
  CHECK(scheduler::cpu_stream_progress(first).accepted == 0);
  CHECK(sizeof(mlx_submission_runtime_baseline) == 7 * sizeof(size_t));
}

TEST_CASE("runtime initialization rejects original scope and null before changing output or workers") {
  auto stream = new_stream(Device::cpu);
  mlx_submission_runtime_baseline out{19, 23, 29, 31, 37, 41, 43};
  CHECK(mlx_submission_prepare_runtime(&out, mlx_stream{}, mlx_stream{&stream}) == 3);
  CHECK(out.selected_streams == 19);
  {
    Enter scope;
    REQUIRE(scope.value->enable_scoped_observation());
    REQUIRE(scope.value->require_original_controls() == Failure::none);
    CHECK_FALSE(mlx_submission_runtime_preparation_allowed());
    CHECK(mlx_submission_prepare_runtime(&out, mlx_stream{&stream},
        mlx_stream{&stream}) == 2);
    CHECK(out.selected_streams == 19);
    CHECK(out.unpriced_populations == 43);
    scheduler::CpuStreamToken worker;
    CHECK(scheduler::prepared_cpu_stream(stream, worker) == Failure::missing_worker);
  }
  CHECK(mlx_submission_runtime_preparation_allowed());
  REQUIRE(mlx_submission_prepare_runtime(&out, mlx_stream{&stream},
      mlx_stream{&stream}) == 0);
  CHECK(out.selected_streams == 1);
}

TEST_CASE("runtime initialization failure retains its ordinary prefix and exact native diagnostic") {
  auto operation = new_stream(Device::cpu);
  auto weights = new_stream(Device::cpu);
  const auto worker = scheduler::prepare_cpu_stream(weights);
  worker_for(worker).block();
  struct ErrorCapture {
    std::string text;
    ErrorCapture() {
      mlx_set_error_handler([](const char* text, void* raw) {
        static_cast<ErrorCapture*>(raw)->text = text;
      }, this, [](void*) {});
    }
    ~ErrorCapture() { mlx_set_error_handler(nullptr, nullptr, nullptr); }
  } error;
  mlx_submission_runtime_baseline out{19, 23, 29, 31, 37, 41, 43};
  REQUIRE(mlx_submission_prepare_runtime(&out, mlx_stream{&operation},
      mlx_stream{&weights}) == 1);
  CHECK(error.text.find("CPU stream is blocked after submission failure.") == 0);
  CHECK(out.selected_streams == 19);
  CHECK(out.unpriced_populations == 43);
  scheduler::CpuStreamToken partial;
  REQUIRE(scheduler::prepared_cpu_stream(operation, partial) == Failure::none);
  CHECK(scheduler::cpu_stream_progress(partial).accepted == 0);
  CHECK(scheduler::prepared_cpu_stream(weights, partial) == Failure::stopped_worker);
  // No rollback/replacement of persistent native resources and no new request.
  REQUIRE(mlx_submission_prepare_runtime(&out, mlx_stream{&operation},
      mlx_stream{&operation}) == 0);
  CHECK(out.cpu_workers == 1);
}

TEST_CASE("scheduler final failure retirement releases registry and cause before host reentry") {
  auto stream = new_stream(Device::cpu);
  auto destination = new_stream(Device::cpu);
  mlx_submission_runtime_baseline baseline{};
  REQUIRE(mlx_submission_prepare_runtime(&baseline, mlx_stream{&stream},
      mlx_stream{&stream}) == 0);
  scheduler::CpuStreamToken token;
  REQUIRE(scheduler::prepared_cpu_stream(stream, token) == Failure::none);
  auto& worker = worker_for(token);
  struct State {
    Stream destination;
    scheduler::StreamThread* worker;
    std::atomic<size_t> live_causes{0};
    size_t callbacks{0};
    bool registry_unlocked{false}, error_unlocked{false}, cause_retired{false};
    bool reentered{false};
  } state{destination, &worker};
  struct Cause final : std::exception {
    std::atomic<size_t>* live;
    explicit Cause(std::atomic<size_t>& value) : live(&value) { ++*live; }
    Cause(const Cause& other) : live(other.live) { ++*live; }
    ~Cause() override { --*live; }
    const char* what() const noexcept override { return "final original cause"; }
  };
  auto failure = FailureCarrier::create(&state, [](void* raw) {
    auto& state = *static_cast<State*>(raw);
    ++state.callbacks;
    state.cause_retired = state.live_causes.load() == 0;
    try {
      state.registry_unlocked = scheduler::SchedulerControlTestAccess::registry_unlocked(
          scheduler::scheduler());
      {
        std::unique_lock lock(state.worker->error_mtx, std::try_to_lock);
        state.error_unlocked = lock.owns_lock();
      }
      if (state.registry_unlocked && state.error_unlocked) {
        auto result = scheduler::prepare_cpu_stream(state.destination);
        state.reentered = result.state != nullptr;
      }
    } catch (...) { state.reentered = false; }
  });
  Counts graph_count;
  Graph graph(1 << 20, graph_count);
  Records records;
  {
    Enter scope(graph.value, records.value);
    configure(*scope.value, failure);
    auto* record = enter_record();
    record->prepare_stream(stream);
    const auto before = worker.progress().accepted;
    {
      RecordDispatchGuard dispatch(*record);
      scheduler::enqueue_counted(stream, [&state] { throw Cause(state.live_causes); });
    }
    record->finish(false);
    scope.value->seal();
    await(token, before + 1);
    REQUIRE(failure.get()->borrow());
    REQUIRE(state.live_causes == 1);
    CHECK(failure.get()->borrow()->exception_type == &typeid(Cause));
    settle(*scope.value);
  }
  retire_registry();
  failure = {}; // The process-lived first-failure slot is now the final owner.
  REQUIRE(state.callbacks == 0);
  auto message = scheduler::scheduler().take_stream_error(stream);
  REQUIRE(message);
  CHECK(*message == "final original cause");
  CHECK(state.callbacks == 1);
  CHECK(state.cause_retired);
  CHECK(state.registry_unlocked);
  CHECK(state.error_unlocked);
  CHECK(state.reentered);
  CHECK(state.live_causes == 0);
  CHECK_FALSE(worker.first_failure);
  CHECK_FALSE(scheduler::scheduler().take_stream_error(stream));
  CHECK(state.callbacks == 1);
}

TEST_CASE("distinct stream exception retires outside both locks before carrier first-cause custody") {
  auto stream = new_stream(Device::cpu);
  const auto token = scheduler::prepare_cpu_stream(stream);
  auto& worker = worker_for(token);
  struct State {
    scheduler::StreamThread* worker;
    size_t first{0}, later{0}, custody{0};
    bool armed{false}, later_registry_unlocked{false}, later_error_unlocked{false};
    bool both_retired_before_custody{false};
  } state{&worker};
  struct First final : std::exception {
    State* state;
    explicit First(State& value) : state(&value) { ++state->first; }
    First(const First& other) : state(other.state) { ++state->first; }
    ~First() override { --state->first; }
    const char* what() const noexcept override { return "carrier first cause"; }
  };
  struct Later final : std::exception {
    State* state;
    explicit Later(State& value) : state(&value) { ++state->later; }
    Later(const Later& other) : state(other.state) { ++state->later; }
    ~Later() override {
      if (--state->later == 0 && state->armed) {
        try {
          state->later_registry_unlocked =
              scheduler::SchedulerControlTestAccess::registry_unlocked(scheduler::scheduler());
          std::unique_lock lock(state->worker->error_mtx, std::try_to_lock);
          state->later_error_unlocked = lock.owns_lock();
        } catch (...) {}
      }
    }
    const char* what() const noexcept override { return "distinct later stream cause"; }
  };
  auto failure = FailureCarrier::create(&state, [](void* raw) {
    auto& state = *static_cast<State*>(raw);
    ++state.custody;
    state.both_retired_before_custody = state.first == 0 && state.later == 0;
  });
  try { throw First(state); }
  catch (...) { failure.get()->capture_current_exception(); }
  worker.record_error(std::make_exception_ptr(Later(state)), failure);
  REQUIRE(state.first == 1);
  REQUIRE(state.later == 1);
  REQUIRE(failure.get()->borrow()->exception_type == &typeid(First));
  state.armed = true;
  failure = {};
  CHECK(state.custody == 0);
  auto message = scheduler::scheduler().take_stream_error(stream);
  REQUIRE(message);
  CHECK(*message == "distinct later stream cause");
  CHECK(state.later_registry_unlocked);
  CHECK(state.later_error_unlocked);
  CHECK(state.both_retired_before_custody);
  CHECK(state.custody == 1);
  CHECK(state.first == 0);
  CHECK(state.later == 0);
}

TEST_CASE("stack scheduler drains queued counted tasks before notification and custody teardown") {
  auto stream = new_stream(Device::cpu);
  std::atomic<bool> entered{false}, release{false};
  std::atomic<size_t> called{0}, retired{0};
  std::array<int, 3> order{};
  std::thread unlocker;
  struct Join {
    std::thread& thread;
    std::atomic<bool>& release;
    ~Join() { release = true; if (thread.joinable()) thread.join(); }
  } join{unlocker, release};
  bool refused{false}, zero_before_members{false};
  {
    scheduler::Scheduler owner;
    // This lexical helper releases the barrier on assertion/exception paths
    // before owner destruction. Normal destruction uses the stopping observer.
    struct UnwindRelease {
      std::atomic<bool>& release;
      int exceptions{std::uncaught_exceptions()};
      ~UnwindRelease() {
        if (std::uncaught_exceptions() > exceptions) release = true;
      }
    } unwind{release};
    auto custody = std::shared_ptr<int>(new int(1), [&](int* value) {
      delete value;
      ++retired;
      zero_before_members = owner.n_active_tasks() == 0;
    });
    owner.enqueue(stream, [&, custody] {
      entered = true;
      while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
      order[called++] = 11;
      try { owner.enqueue(stream, [] {}); }
      catch (const std::runtime_error& error) {
        refused = std::string_view(error.what()) ==
            "Cannot enqueue work after CPU stream is stopped or blocked.";
      }
    }, true);
    owner.enqueue(stream, [&, custody] { order[called++] = 23; }, true);
    owner.enqueue(stream, [&, custody] { order[called++] = 47; }, true);
    custody.reset();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!entered.load(std::memory_order_acquire)) {
      REQUIRE(std::chrono::steady_clock::now() < deadline);
      std::this_thread::yield();
    }
    REQUIRE(owner.n_active_tasks() == 3);
    unlocker = std::thread([&owner, &release] {
      while (!scheduler::SchedulerControlTestAccess::stopping(owner))
        std::this_thread::yield();
      release = true;
    });
  }
  unlocker.join();
  CHECK(called == 3);
  CHECK((order == std::array<int, 3>{11, 23, 47}));
  CHECK(refused);
  CHECK(retired == 1);
  // Callable custody precedes counted notification, so the last capture dies
  // with one counted task still active; the task notifies before the join ends.
  CHECK_FALSE(zero_before_members);
}


TEST_CASE("independent original control child preserves strict nesting and failed ancestry") {
  (void)initialized_cpu();
  const auto stream = new_stream(Device::cpu);
  const auto token = scheduler::prepare_cpu_stream(stream);
  auto& worker = worker_for(token);
  Counts parent_graph_count, child_graph_count, parent_scope_count, child_scope_count;
  Counts parent_failure_count, child_failure_count, rejected_count;
  Graph parent_graph(1 << 20, parent_graph_count), child_graph(1 << 20, child_graph_count);
  Records parent_records, child_records;
  auto parent_failure = carrier(parent_failure_count), child_failure = carrier(child_failure_count);
  Enter parent(parent_graph.value, parent_records.value, &parent_scope_count);
  configure(*parent.value, parent_failure);
  mlx_submission_observer parent_observer{nullptr}, child_observer{nullptr};
  REQUIRE(mlx_submission_observer_current(&parent_observer) == 0);
  mlx_submission_scope child{nullptr}, rejected{nullptr};
  CHECK(mlx_submission_scope_new_retaining_with_arenas(&rejected, &rejected_count, retired,
      mlx_submission_record_quota{child_records.value},
      mlx_submission_graph_quota{child_graph.value}) == 2);
  CHECK(rejected.ctx == nullptr);
  CHECK(rejected_count.retired == 0);
  CHECK(current_scope() == parent.value);
  const auto parent_occupied = parent_graph.value->occupied_bytes();

  REQUIRE(mlx_submission_scope_new_original_child(&child, &child_scope_count, retired,
      mlx_submission_record_quota{child_records.value},
      mlx_submission_graph_quota{child_graph.value}, parent_observer) == 0);
  auto* active_child = static_cast<Scope*>(child.ctx);
  configure(*active_child, child_failure);
  REQUIRE(mlx_submission_observer_current(&child_observer) == 0);
  CHECK(current_graph_quota() == child_graph.value);
  CHECK(current_record_quota() == child_records.value);
  // Same retained parent, now stale as a current-parent loan.
  CHECK(mlx_submission_scope_new_original_child(&rejected, &rejected_count, retired,
      mlx_submission_record_quota{child_records.value},
      mlx_submission_graph_quota{child_graph.value}, parent_observer) == 4);
  CHECK(rejected.ctx == nullptr);
  CHECK(rejected_count.retired == 0);

  std::atomic<int> result{0};
  auto* record = enter_record();
  record->prepare_stream(stream);
  const auto before = worker.progress().accepted;
  {
    RecordDispatchGuard dispatch(*record);
    CHECK(mlx_submission_scope_new_original_child(&rejected, &rejected_count, retired,
        mlx_submission_record_quota{child_records.value},
        mlx_submission_graph_quota{child_graph.value}, child_observer) == 4);
    CHECK(current_scope() == active_child);
    scheduler::enqueue_counted(stream, [&result] {
      result.store(7 + 13 + 23, std::memory_order_release);
      throw std::range_error("independent control child failure");
    });
  }
  record->finish(false);
  REQUIRE(mlx_submission_scope_seal(child) == 0);
  CHECK(current_scope() == parent.value);
  CHECK(current_graph_quota() == parent_graph.value);
  CHECK(current_record_quota() == parent_records.value);
  CHECK(parent_graph.value->occupied_bytes() == parent_occupied);
  mlx_submission_observer_release(parent_observer);
  parent_observer.ctx = nullptr;
  parent.reset();
  CHECK(parent_scope_count.retired == 0);
  mlx_submission_scope_free(child);
  child.ctx = nullptr;
  CHECK(child_scope_count.retired == 0);
  await(token, before + 1);
  CHECK(result.load(std::memory_order_acquire) == 43);
  REQUIRE(child_failure.get()->borrow());
  CHECK_FALSE(parent_failure.get()->borrow());
  settle(*active_child);
  CHECK(active_child->query().failed);
  CHECK(mlx_submission_observer_retire(child_observer) == 0);
  mlx_submission_observer_release(child_observer);
  child_observer.ctx = nullptr;
  CHECK(child_scope_count.retired == 1);
  CHECK(parent_scope_count.retired == 1);
  CHECK(current_scope() == nullptr);
  CHECK(child_graph.value->occupied_bytes() == 0);
  REQUIRE(worker.error(true));
  child_failure = {};
  parent_failure = {};
  CHECK(child_failure_count.retired == 1);
  CHECK(parent_failure_count.retired == 1);
  CHECK(rejected_count.retired == 0);
}

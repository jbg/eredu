#include "doctest/doctest.h"
#include "mlx/c/event.h"
#include "mlx/c/host_transfer.h"
#include "mlx/c/submission.h"
#include "mlx/failure.h"
#include "mlx/graph_quota.h"
#include "mlx/record_quota.h"
#include "mlx/submission.h"
#include "mlx/host_transfer.h"
#include "mlx/scoped_evaluation.h"
#include "mlx/c/private/array.h"
#include <chrono>
#include <atomic>
#include <algorithm>
#include <iterator>
#include <thread>
#include <optional>
#include <exception>
#include <limits>
#include <condition_variable>
#include <mutex>
using namespace mlx::core;
using namespace mlx::core::submission;
namespace {
struct Owner {
  GraphQuota* graph{GraphQuota::create(1 << 20, nullptr, nullptr)};
  RecordQuota* records{RecordQuota::create(1 << 20, nullptr, nullptr)};
  Scope* scope{new Scope(nullptr, nullptr, records, graph)};
  unsigned retired{0};
  FailureCarrierRef failure{FailureCarrier::create(&retired,
      [](void* value) { ++*static_cast<unsigned*>(value); })};
  explicit Owner(size_t record_capacity = 1 << 20)
      : records(RecordQuota::create(record_capacity, nullptr, nullptr)) {
    REQUIRE(scope->enable_scoped_observation());
    REQUIRE(scope->require_original_controls() == NativeControlFailure::none);
    REQUIRE(scope->bind_failure(failure));
    REQUIRE(scope->enable_original_controls() == NativeControlFailure::none);
  }
  ~Owner() { scope->seal(); scope->release(); records->release(); graph->release(); }
};
struct PinnedRecord final : Record {
  PinnedRecord(Allocation allocation, std::atomic<unsigned>& count) : Record(allocation), destroyed(count) {}
  ~PinnedRecord() override { ++destroyed; }
  bool scoped_observation_supported() const noexcept override { return true; }
  std::atomic<unsigned>& destroyed;
};
// A completion Event can signal before its stream's task/handler frontier is
// fully retired. Observed only says that polling is permitted; the exact
// structural owner status is the separate terminal predicate.
void await_original_terminal(Owner& owner) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  for (;;) {
    REQUIRE(owner.scope->progress_scoped() == ScopedProgress::observed);
    const auto status = owner.scope->query();
    REQUIRE_FALSE(status.failed);
    REQUIRE_FALSE(status.blocked);
    if (status.activity != Activity::pending) return;
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
}
void terminal_record(std::atomic<unsigned>& count) {
  auto record = Record::create<PinnedRecord>(count);
  record->enter();
  auto* entered = record.release();
  entered->finish(false);
}
}

TEST_CASE("original observer retires terminal record storage while its role remains active") {
  Owner owner;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  CHECK(observer.ctx == owner.scope);
  CHECK(current_scope() == owner.scope);
  const auto record_start = owner.records->occupied_bytes();
  std::atomic<unsigned> destroyed{0};
  terminal_record(destroyed);
  REQUIRE(owner.scope->progress_scoped() == ScopedProgress::observed);
  CHECK(owner.scope->query().activity != Activity::pending);
  CHECK(destroyed == 0);
  CHECK(owner.records->occupied_bytes() > record_start);
  REQUIRE(mlx_submission_observer_retire(observer) == 0);
  CHECK(destroyed == 1);
  CHECK(owner.records->occupied_bytes() == record_start);
  CHECK(current_scope() == owner.scope);
  CHECK(owner.scope->active_on_owner_thread());
  mlx_submission_observer_release(observer);
  CHECK(owner.scope->active_on_owner_thread());
}

TEST_CASE("original observer retains structural child lifetime and rejects child submission authority") {
  Owner owner;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  auto* child = new Scope;
  mlx_submission_observer refused{};
  CHECK(mlx_submission_observer_current(&refused) == 4);
  CHECK(refused.ctx == nullptr);
  mlx_operation_event event{};
  CHECK(mlx_operation_event_new(&event, observer) == 4);
  CHECK(event.ctx == nullptr);
  mlx_submission_status status{};
  REQUIRE(mlx_submission_observer_query(&status, observer) == 0);
  CHECK(status.activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  CHECK(!owner.scope->query_records().pending);
  child->seal(); child->release();
  REQUIRE(mlx_submission_observer_query(&status, observer) == 0);
  CHECK(status.activity != MLX_SUBMISSION_ACTIVITY_PENDING);
  owner.scope->seal();
  CHECK(mlx_operation_event_new(&event, observer) == 4);
  CHECK(mlx_submission_observer_progress(&status, observer) == 0);
  CHECK(mlx_submission_observer_retire(observer) == 0);
  mlx_submission_observer_release(observer);
}

TEST_CASE("original observer does not retire another role and refuses foreign host observations") {
  Owner first;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  std::atomic<unsigned> first_destroyed{0}, second_destroyed{0};
  terminal_record(first_destroyed);
  REQUIRE(first.scope->progress_scoped() == ScopedProgress::observed);
  first.scope->seal();
  {
    Owner second;
    terminal_record(second_destroyed);
    REQUIRE(second.scope->progress_scoped() == ScopedProgress::observed);
    REQUIRE(mlx_submission_observer_retire(observer) == 0);
    CHECK(first_destroyed == 1);
    CHECK(second_destroyed == 0);
    REQUIRE(second.scope->retire_scoped_records() == ScopedProgress::observed);
  }
  CHECK(second_destroyed == 1);
  std::thread foreign([observer] {
    mlx_submission_status untouched{MLX_SUBMISSION_ACTIVITY_PENDING, true, true};
    CHECK(mlx_submission_observer_query(&untouched, observer) == 4);
    CHECK(untouched.activity == MLX_SUBMISSION_ACTIVITY_PENDING);
    CHECK((untouched.failed && untouched.blocked));
    CHECK(mlx_submission_observer_retain(observer) == 4);
  });
  foreign.join();
  mlx_submission_observer_release(observer);
}

TEST_CASE("original deferred completion uses paid wrapper and exact role retirement") {
  Owner owner;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  const auto before = owner.graph->occupied_bytes();
  mlx_operation_event event{};
  REQUIRE(mlx_operation_event_new(&event, observer) == 0);
  REQUIRE(mlx_operation_event_submit(event) == 0);
  CHECK(mlx_operation_event_query(event) == 0);
  CHECK(owner.graph->occupied_bytes() > before);
  mlx_operation_event_defer(event);
  CHECK(owner.graph->occupied_bytes() > before);
  CHECK(owner.scope->query().activity != Activity::pending);
  REQUIRE(mlx_submission_observer_retire(observer) == 0);
  CHECK(owner.graph->occupied_bytes() == before);
  CHECK(owner.scope->active_on_owner_thread());
  mlx_submission_observer_release(observer);
}

void original_host_copy_roundtrip(Device::DeviceType device) {
  auto stream = default_stream(device);
  mlx_submission_runtime_baseline runtime{};
  REQUIRE(mlx_submission_prepare_runtime(&runtime, mlx_stream{&stream}, mlx_stream{&stream}) == 0);
  HostTransferBuffer source(Shape{3}, float32, HostTransferPolicy::transfer);
  const float expected[] = {2.0f, -3.0f, 7.0f};
  std::copy(std::begin(expected), std::end(expected), static_cast<float*>(source.data()));
  Owner owner;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  const auto baseline = owner.graph->occupied_bytes();
  for (unsigned iteration = 0; iteration != 2; ++iteration) {
    mlx_operation_event producer{}, consumer{};
    mlx_array value{};
    mlx_host_transfer_buffer host{};
    REQUIRE(mlx_operation_event_new(&producer, observer) == 0);
    REQUIRE(mlx_copy_from_host_operation(&value, producer,
        mlx_host_transfer_buffer{&source, nullptr}, mlx_stream{&stream}) == 0);
    auto wrong_device = Stream{stream.index, Device{device == Device::cpu ? Device::gpu : Device::cpu}};
    const auto occupied = owner.graph->occupied_bytes();
    CHECK(mlx_operation_event_wait_stream(producer, mlx_stream{&wrong_device}) == 4);
    CHECK(owner.graph->occupied_bytes() == occupied);
    REQUIRE(mlx_operation_event_wait_stream(producer, mlx_stream{&stream}) == 0);
    // A Metal stream wait deliberately leaves a consumer receipt open. Its
    // producer cannot establish whole-owner completion until actual consumer
    // work commits that same stream. Fixed funded progress is not failure.
    CHECK(mlx_operation_event_wait(producer) == (device == Device::gpu ? 8u : 0u));
    CHECK(!owner.scope->query().failed);
    REQUIRE(mlx_operation_event_new(&consumer, observer) == 0);
    REQUIRE(mlx_copy_to_host_operation(&host, consumer, value,
        MLX_HOST_TRANSFER_POLICY_TRANSFER, mlx_stream{&stream}) == 0);
    REQUIRE(mlx_operation_event_wait(consumer) == 0);
    REQUIRE(mlx_operation_event_wait(producer) == 0);
    REQUIRE(validate_scoped_array(mlx_array_get_(value), *owner.scope) == ScopedEvaluation::complete);
    for (unsigned i = 0; i != 3; ++i) CHECK(mlx_array_get_(value).data<float>()[i] == expected[i]);
    auto* output = static_cast<HostTransferBuffer*>(host.ctx);
    for (unsigned i = 0; i != 3; ++i) CHECK(static_cast<const float*>(output->data())[i] == expected[i]);
    REQUIRE(host.prepared_owner != nullptr);
    mlx_host_transfer_buffer_free(host);
    mlx_array_free(value);
    mlx_operation_event_free(producer);
    mlx_operation_event_free(consumer);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (owner.scope->query().activity == Activity::pending) {
      REQUIRE(owner.scope->progress_scoped() == ScopedProgress::observed);
      REQUIRE(std::chrono::steady_clock::now() < deadline);
      std::this_thread::yield();
    }
    REQUIRE(mlx_submission_observer_retire(observer) == 0);
    CHECK(owner.graph->occupied_bytes() == baseline);
    CHECK(owner.scope->active_on_owner_thread());
  }
  mlx_submission_observer_release(observer);
}

TEST_CASE("original native CPU host copies preserve values and reclaim an active role window") {
  original_host_copy_roundtrip(Device::cpu);
}

namespace {
// Keeps a real thread-local GPU encoder alive on its creating thread. Its
// globally assigned scalar index is not authority to use it from this thread.
struct ForeignThreadStream {
  std::optional<Stream> value;
  std::exception_ptr error;
  std::mutex mutex;
  std::condition_variable changed;
  bool ready{false}, release{false};
  std::thread worker;
  explicit ForeignThreadStream(Device::DeviceType device) : worker([this, device] {
    try { value = new_stream(device); } catch (...) { error = std::current_exception(); }
    std::unique_lock lock(mutex);
    ready = true;
    changed.notify_all();
    changed.wait(lock, [this] { return release; });
  }) {
    std::unique_lock lock(mutex);
    changed.wait(lock, [this] { return ready; });
    if (error) {
      release = true;
      lock.unlock();
      changed.notify_all();
      worker.join();
      std::rethrow_exception(error);
    }
  }
  ~ForeignThreadStream() {
    { std::lock_guard lock(mutex); release = true; }
    changed.notify_all();
    worker.join();
  }
};
}

// Covers empty groups, evaluated identity results, and a result carrying an
// event from another stream. No global synchronization/flush participates.
void original_selected_stream_frontier(Device::DeviceType device) {
  auto producer_stream = new_stream(device);
  auto second_stream = new_stream(device);
  auto consumer_stream = new_stream(device);
  REQUIRE(producer_stream != consumer_stream);
  REQUIRE(second_stream != consumer_stream);
  REQUIRE(producer_stream != second_stream);
  mlx_submission_runtime_baseline runtime{};
  REQUIRE(mlx_submission_prepare_runtime(&runtime, mlx_stream{&producer_stream},
      mlx_stream{&consumer_stream}) == 0);
  REQUIRE(mlx_submission_prepare_runtime(&runtime, mlx_stream{&second_stream},
      mlx_stream{&consumer_stream}) == 0);
  HostTransferBuffer source(Shape{3}, float32, HostTransferPolicy::transfer);
  const float expected[] = {2.0f, -3.0f, 7.0f};
  std::copy(std::begin(expected), std::end(expected), static_cast<float*>(source.data()));
  array identity(expected, Shape{3}, float32);
  REQUIRE(identity.status() == array::Status::available);
  REQUIRE_FALSE(identity.event().valid());
  std::optional<Stream> cross_cpu_stream;
  std::optional<ForeignThreadStream> foreign;
  if (device == Device::gpu) {
    cross_cpu_stream = new_stream(Device::cpu);
    REQUIRE(mlx_submission_prepare_runtime(&runtime, mlx_stream{&*cross_cpu_stream},
        mlx_stream{&consumer_stream}) == 0);
    foreign.emplace(device);
  }
  Owner owner;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  const auto graph_start = owner.graph->occupied_bytes();
  const auto record_start = owner.records->occupied_bytes();
  for (unsigned mode = 0; mode != 3; ++mode) {
    mlx_operation_event first{}, second{}, old_identity{}, selected{};
    mlx_array left{}, right{};
    REQUIRE(mlx_operation_event_new(&first, observer) == 0);
    REQUIRE(mlx_copy_from_host_operation(&left, first,
        mlx_host_transfer_buffer{&source, nullptr}, mlx_stream{&producer_stream}) == 0);
    REQUIRE(mlx_operation_event_new(&second, observer) == 0);
    REQUIRE(mlx_copy_from_host_operation(&right, second,
        mlx_host_transfer_buffer{&source, nullptr}, mlx_stream{&second_stream}) == 0);
    REQUIRE(mlx_operation_event_wait(first) == 0);
    REQUIRE(mlx_operation_event_wait(second) == 0);
    // Parent Event readiness does not imply that its queued task controls
    // have been destroyed. Reach the exact terminal frontier before taking a
    // stable no-allocation refusal snapshot, then open the real consumer waits.
    await_original_terminal(owner);

    REQUIRE(mlx_operation_event_new(&selected, observer) == 0);
    if (mode == 1) REQUIRE(mlx_operation_event_append(selected, mlx_array{&identity}) == 0);
    if (mode == 2) REQUIRE(mlx_operation_event_append(selected, left) == 0);
    const auto before_refusal = owner.graph->occupied_bytes();
    const auto records_before_refusal = owner.records->occupied_bytes();
    auto refuse = [&](mlx_stream invalid) {
      CHECK(mlx_operation_event_submit_on_stream(selected, invalid) == 4);
      CHECK(mlx_operation_event_query(selected) == 5); // still unconsumed
      CHECK(owner.graph->occupied_bytes() == before_refusal);
      CHECK(owner.records->occupied_bytes() == records_before_refusal);
      CHECK(!owner.scope->query().failed);
    };
    refuse(mlx_stream{});
    auto missing = Stream{-1, consumer_stream.device};
    refuse(mlx_stream{&missing});
    auto wrong_type = Stream{consumer_stream.index,
        Device{device == Device::cpu ? Device::gpu : Device::cpu}};
    refuse(mlx_stream{&wrong_type});
    auto wrong_device = Stream{consumer_stream.index,
        Device{device, std::numeric_limits<int>::max()}};
    refuse(mlx_stream{&wrong_device});
    if (foreign) refuse(mlx_stream{&*foreign->value});
    auto* child = new Scope;
    refuse(mlx_stream{&consumer_stream});
    child->seal(); child->release();
    // Both prerequisites can append waits before the selected consumer is
    // submitted. A whole-role funded outcome is not a failure of the second.
    REQUIRE(mlx_operation_event_wait_stream(first, mlx_stream{&consumer_stream}) == 0);
    REQUIRE(mlx_operation_event_wait_stream(second, mlx_stream{&consumer_stream}) == 0);

    REQUIRE(mlx_operation_event_new(&old_identity, observer) == 0);
    if (mode == 1) REQUIRE(mlx_operation_event_append(old_identity, mlx_array{&identity}) == 0);
    if (mode == 2) REQUIRE(mlx_operation_event_append(old_identity, left) == 0);
    REQUIRE(mlx_operation_event_submit(old_identity) == 0);
    CHECK(mlx_operation_event_wait(old_identity) == (device == Device::gpu ? 8u : 0u));
    CHECK(!owner.scope->query().failed);

    // Refusal did not consume the event or authorize the unconfigured child.
    REQUIRE(mlx_operation_event_submit_on_stream(selected, mlx_stream{&consumer_stream}) == 0);
    REQUIRE(mlx_operation_event_wait(selected) == 0);
    REQUIRE(mlx_operation_event_wait(old_identity) == 0);
    REQUIRE(mlx_operation_event_wait(first) == 0);
    REQUIRE(mlx_operation_event_wait(second) == 0);
    REQUIRE(validate_scoped_array(mlx_array_get_(left), *owner.scope) == ScopedEvaluation::complete);
    REQUIRE(validate_scoped_array(mlx_array_get_(right), *owner.scope) == ScopedEvaluation::complete);
    for (unsigned i = 0; i != 3; ++i) {
      CHECK(mlx_array_get_(left).data<float>()[i] == expected[i]);
      CHECK(mlx_array_get_(right).data<float>()[i] == expected[i]);
      CHECK(identity.data<float>()[i] == expected[i]);
    }
    mlx_operation_event_free(selected);
    mlx_operation_event_free(old_identity);
    mlx_operation_event_free(second);
    mlx_operation_event_free(first);
    mlx_array_free(right);
    mlx_array_free(left);
    await_original_terminal(owner);
    REQUIRE(mlx_submission_observer_retire(observer) == 0);
    CHECK(owner.records->occupied_bytes() == record_start);
    CHECK(owner.graph->occupied_bytes() == graph_start);
    CHECK(owner.scope->active_on_owner_thread());
    CHECK(current_scope() == owner.scope);
  }
  if (cross_cpu_stream) {
    // Valid cross-device roots are still handled by the shared evaluator's
    // device-switch fence/event path. Selected-stream identity does not impose
    // a new output-device equality rule.
    {
      auto output = negative(identity, *cross_cpu_stream);
      mlx_operation_event selected{};
      REQUIRE(mlx_operation_event_new(&selected, observer) == 0);
      REQUIRE(mlx_operation_event_append(selected, mlx_array{&output}) == 0);
      REQUIRE(mlx_operation_event_submit_on_stream(selected, mlx_stream{&consumer_stream}) == 0);
      REQUIRE(mlx_operation_event_wait(selected) == 0);
      REQUIRE(validate_scoped_array(output, *owner.scope) == ScopedEvaluation::complete);
      for (unsigned i = 0; i != 3; ++i) CHECK(output.data<float>()[i] == -expected[i]);
      mlx_operation_event_free(selected);
    }
    await_original_terminal(owner);
    REQUIRE(mlx_submission_observer_retire(observer) == 0);
    CHECK(owner.records->occupied_bytes() == record_start);
    CHECK(owner.graph->occupied_bytes() == graph_start);
    CHECK(owner.scope->active_on_owner_thread());
  }
  mlx_submission_observer_release(observer);
}

TEST_CASE("original CPU selected stream closes empty identity and foreign-stream result frontiers") {
  original_selected_stream_frontier(Device::cpu);
}

void original_prepared_global_stream_frontier(Stream stream) {
  Owner owner;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  const auto graph_start = owner.graph->occupied_bytes();
  const auto record_start = owner.records->occupied_bytes();
  mlx_operation_event selected{};
  REQUIRE(mlx_operation_event_new(&selected, observer) == 0);
  REQUIRE(mlx_operation_event_submit_on_stream(selected, mlx_stream{&stream}) == 0);
  REQUIRE(mlx_operation_event_wait(selected) == 0);
  mlx_operation_event_free(selected);
  await_original_terminal(owner);
  REQUIRE(mlx_submission_observer_retire(observer) == 0);
  CHECK(owner.graph->occupied_bytes() == graph_start);
  CHECK(owner.records->occupied_bytes() == record_start);
  CHECK(owner.scope->active_on_owner_thread());
  mlx_submission_observer_release(observer);
}

void original_completed_array_validation(Device::DeviceType device) {
  auto stream = new_stream(device);
  mlx_submission_runtime_baseline runtime{};
  REQUIRE(mlx_submission_prepare_runtime(&runtime, mlx_stream{&stream}, mlx_stream{&stream}) == 0);
  const float expected[] = {2.0f, -3.0f, 7.0f};
  array source(expected, Shape{3}, float32);
  Owner owner;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  const auto graph_start = owner.graph->occupied_bytes();
  const auto record_start = owner.records->occupied_bytes();
  {
    auto lazy = negative(source, stream);
    auto output = negative(source, stream);
    const auto graph_before = owner.graph->occupied_bytes();
    const auto record_before = owner.records->occupied_bytes();
    CHECK(mlx_submission_observer_validate_array(observer, mlx_array{}) == 4);
    CHECK(mlx_submission_observer_validate_array(observer, mlx_array{&lazy}) == 1);
    CHECK(lazy.status() == array::Status::unscheduled);
    CHECK(!lazy.event().valid());
    CHECK(owner.graph->occupied_bytes() == graph_before);
    CHECK(owner.records->occupied_bytes() == record_before);
    mlx_operation_event event{};
    REQUIRE(mlx_operation_event_new(&event, observer) == 0);
    REQUIRE(mlx_operation_event_append(event, mlx_array{&output}) == 0);
    REQUIRE(mlx_operation_event_submit_on_stream(event, mlx_stream{&stream}) == 0);
    REQUIRE(mlx_operation_event_wait(event) == 0);
    await_original_terminal(owner);
    REQUIRE(output.event().valid());
    owner.scope->seal();
    const auto graph_before_validation = owner.graph->occupied_bytes();
    const auto record_before_validation = owner.records->occupied_bytes();
    REQUIRE(mlx_submission_observer_validate_array(observer, mlx_array{&output}) == 0);
    CHECK(output.status() == array::Status::available);
    CHECK(!output.event().valid());
    CHECK(owner.graph->occupied_bytes() <= graph_before_validation);
    CHECK(owner.records->occupied_bytes() == record_before_validation);
    for (unsigned i = 0; i != 3; ++i) CHECK(output.data<float>()[i] == -expected[i]);
    CHECK(mlx_submission_observer_validate_array(observer, mlx_array{&lazy}) == 1);
    CHECK(lazy.status() == array::Status::unscheduled);
    CHECK(!lazy.event().valid());
    REQUIRE(mlx_submission_observer_validate_array(observer, mlx_array{&output}) == 0);
    mlx_operation_event_free(event);
    // The first role is sealed and retained. A new independent owner now
    // receives its own arenas/carrier; a nested owner cannot replace its
    // parent's arenas. Validation still rejects its descriptor through the
    // first retained observer while this independent role is current.
    {
      Owner foreign;
      mlx_submission_observer other{};
      REQUIRE(mlx_submission_observer_current(&other) == 0);
      {
        auto value = negative(source, stream);
        mlx_operation_event event{};
        REQUIRE(mlx_operation_event_new(&event, other) == 0);
        REQUIRE(mlx_operation_event_append(event, mlx_array{&value}) == 0);
        REQUIRE(mlx_operation_event_submit_on_stream(event, mlx_stream{&stream}) == 0);
        // Finish native callbacks without touching this descriptor. Its
        // scheduled/event state still requires exact-owner validation.
        await_original_terminal(foreign);
        // Descriptor state stays scheduled even if the hardware finishes early.
        REQUIRE(value.status() != array::Status::available);
        REQUIRE(value.event().valid());
        const auto status = value.status();
        const auto* carrier = value.event().failure_owner().get();
        const auto foreign_graph = foreign.graph->occupied_bytes();
        const auto foreign_records = foreign.records->occupied_bytes();
        CHECK(mlx_submission_observer_validate_array(observer, mlx_array{&value}) == 4);
        CHECK(value.status() == status);
        REQUIRE(value.event().valid());
        CHECK(value.event().failure_owner().get() == carrier);
        CHECK(foreign.graph->occupied_bytes() == foreign_graph);
        CHECK(foreign.records->occupied_bytes() == foreign_records);
        REQUIRE(mlx_operation_event_wait(event) == 0);
        foreign.scope->seal();
        REQUIRE(mlx_submission_observer_validate_array(other, mlx_array{&value}) == 0);
        CHECK(value.status() == array::Status::available);
        CHECK(!value.event().valid());
        for (unsigned i = 0; i != 3; ++i) CHECK(value.data<float>()[i] == -expected[i]);
        mlx_operation_event_free(event);
      }
      await_original_terminal(foreign);
      REQUIRE(mlx_submission_observer_retire(other) == 0);
      mlx_submission_observer_release(other);
    }
  }
  await_original_terminal(owner);
  REQUIRE(mlx_submission_observer_retire(observer) == 0);
  CHECK(owner.graph->occupied_bytes() == graph_start);
  CHECK(owner.records->occupied_bytes() == record_start);
  mlx_submission_observer_release(observer);
}

TEST_CASE("original CPU completed array validation preserves refusal and detaches after seal") {
  original_completed_array_validation(Device::cpu);
}

#include "operation_observer/exact_roots.cpp"
#include "operation_observer/prepared_clones.cpp"

#include "operation_observer/evaluator_streams.cpp"

#include "operation_observer/owned_host_copy.cpp"

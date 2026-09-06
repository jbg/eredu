#include "doctest/doctest.h"

#include "mlx/c/error.h"
#include "mlx/c/event.h"
#include "mlx/c/host_transfer.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/event.h"
#include "mlx/c/private/host_transfer.h"
#include "mlx/c/private/prepared_array_output.h"
#include "mlx/c/transforms.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/submission.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <new>

using namespace mlx::core;

namespace {

struct AllocationBudget {
  size_t remaining;
  size_t attempts{0};
  size_t frees{0};
};

thread_local AllocationBudget* active_allocation_budget = nullptr;

struct TrackAllocations {
  AllocationBudget* previous;

  explicit TrackAllocations(AllocationBudget& budget)
      : previous(active_allocation_budget) {
    active_allocation_budget = &budget;
  }

  ~TrackAllocations() {
    active_allocation_budget = previous;
  }
};

} // namespace

// Test-binary-only allocation injection. The budget is inactive except around
// direct publication-guard operations; assertions execute outside that scope.
void* operator new(std::size_t size) {
  if (active_allocation_budget) {
    ++active_allocation_budget->attempts;
    if (active_allocation_budget->remaining == 0) {
      throw std::bad_alloc();
    }
    --active_allocation_budget->remaining;
  }
  if (auto* storage = std::malloc(size == 0 ? 1 : size)) {
    return storage;
  }
  throw std::bad_alloc();
}

void operator delete(void* storage) noexcept {
  if (storage && active_allocation_budget) {
    ++active_allocation_budget->frees;
  }
  std::free(storage);
}

void operator delete(void* storage, std::size_t) noexcept {
  ::operator delete(storage);
}

size_t native_recovery_without_allocations(void (*operation)(void*), void* context) {
  AllocationBudget budget{0};
  {
    TrackAllocations tracking(budget);
    operation(context);
  }
  return budget.attempts;
}

namespace {

struct OwnedEvent {
  mlx_event value{nullptr};

  ~OwnedEvent() {
    mlx_event_free(value);
  }
};

struct OwnedHostBuffer {
  mlx_host_transfer_buffer value{nullptr};

  ~OwnedHostBuffer() {
    mlx_host_transfer_buffer_free(value);
  }
};

struct OwnedArray {
  mlx_array value{nullptr};

  ~OwnedArray() {
    mlx_array_free(value);
  }
};

void initialize_host_buffer(OwnedHostBuffer& buffer, float value) {
  CHECK(mlx_host_transfer_buffer_new(
            &buffer.value,
            nullptr,
            0,
            MLX_FLOAT32,
            MLX_HOST_TRANSFER_POLICY_TRANSFER) == 0);
  void* data = nullptr;
  CHECK(mlx_host_transfer_buffer_data_mut(&data, buffer.value) == 0);
  *static_cast<float*>(data) = value;
}

struct CountErrors {
  int count{0};

  CountErrors() {
    mlx_set_error_handler(
        [](const char*, void* count) { ++*static_cast<int*>(count); },
        &count,
        [](void*) {});
  }

  ~CountErrors() {
    mlx_set_error_handler(nullptr, nullptr, nullptr);
  }
};

class PublicationProbe : public UnaryPrimitive {
 public:
  PublicationProbe(
      Stream stream,
      const mlx_event& destination,
      bool& prepared_before_submission,
      std::atomic<int>& completed,
      bool fail_before_work = false,
      void* const* related_output = nullptr)
      : UnaryPrimitive(stream),
        destination_(destination),
        prepared_before_submission_(prepared_before_submission),
        completed_(completed),
        fail_before_work_(fail_before_work),
        related_output_(related_output) {}

  void eval_cpu(const std::vector<array>& inputs, array& output) override {
    prepared_before_submission_ = destination_.ctx != nullptr &&
        (!related_output_ || *related_output_ != nullptr);
    if (fail_before_work_) {
      throw std::runtime_error("producer failed before queued work");
    }
    output.copy_shared_buffer(inputs[0]);
    scheduler::enqueue(stream(), [completed = &completed_]() {
      completed->fetch_add(1);
    });
  }

  void eval_gpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("PublicationProbe requires a CPU stream");
  }

  const char* name() const override {
    return "PublicationProbe";
  }

 private:
  const mlx_event& destination_;
  bool& prepared_before_submission_;
  std::atomic<int>& completed_;
  bool fail_before_work_;
  void* const* related_output_;
};

} // namespace

TEST_CASE("test C completion preparation restores storage on failure") {
  for (bool existing_storage : {false, true}) {
    CAPTURE(existing_storage);
    OwnedEvent destination;
    if (existing_storage) {
      destination.value = mlx_event_new();
    }
    const auto previous_storage = destination.value.ctx;
    CHECK_THROWS_WITH_AS(
        [&]() {
          mlx_event_preparation_ preparation(destination.value);
          CHECK(destination.value.ctx != nullptr);
          throw std::runtime_error("producer failed before publication");
        }(),
        "producer failed before publication",
        std::runtime_error);
    CHECK(destination.value.ctx == previous_storage);
  }
}

TEST_CASE("test C completion storage precedes native submission") {
  for (bool timed : {false, true}) {
    for (bool existing_storage : {false, true}) {
      CAPTURE(timed);
      CAPTURE(existing_storage);
      OwnedEvent destination;
      if (existing_storage) {
        destination.value = mlx_event_new();
      }
      const auto previous_storage = destination.value.ctx;
      auto stream = new_thread_unsafe_stream(Device::cpu);
      bool prepared_before_submission = false;
      std::atomic<int> completed{0};
      auto primitive = std::make_shared<PublicationProbe>(
          stream, destination.value, prepared_before_submission, completed);
      auto output = array({}, float32, primitive, {array(7.0f)});
      std::vector<array> outputs{output};

      // These scoped C views borrow the native values for the duration of the
      // call; the owning C producer must allocate its destination first.
      const mlx_vector_array c_outputs{&outputs};
      const mlx_stream c_stream{&stream};
      const int result = timed
          ? mlx_async_eval_timed(&destination.value, c_outputs, c_stream)
          : mlx_async_eval_with_event(&destination.value, c_outputs);

      CHECK(result == 0);
      CHECK(prepared_before_submission);
      CHECK(destination.value.ctx != nullptr);
      if (existing_storage) {
        CHECK(destination.value.ctx == previous_storage);
      }
      CHECK(mlx_event_synchronize(destination.value) == 0);
      CHECK(completed.load() == 1);
      CHECK(output.item<float>() == 7.0f);
      if (timed) {
        double seconds = -1.0;
        CHECK(mlx_event_elapsed(&seconds, destination.value) == 0);
        CHECK(seconds >= 0.0);
      }
    }
  }
}

TEST_CASE("test C completion producer failure preserves output storage") {
  for (bool timed : {false, true}) {
    for (bool existing_storage : {false, true}) {
      CAPTURE(timed);
      CAPTURE(existing_storage);
      OwnedEvent destination;
      if (existing_storage) {
        destination.value = mlx_event_new();
      }
      const auto previous_storage = destination.value.ctx;
      auto stream = new_thread_unsafe_stream(Device::cpu);
      bool prepared_before_submission = false;
      std::atomic<int> completed{0};
      auto primitive = std::make_shared<PublicationProbe>(
          stream, destination.value, prepared_before_submission, completed, true);
      auto output = array({}, float32, primitive, {array(7.0f)});
      std::vector<array> outputs{output};
      const mlx_vector_array c_outputs{&outputs};
      const mlx_stream c_stream{&stream};
      CountErrors errors;
      auto token = scheduler::prepare_cpu_stream(stream);
      const int result = timed
          ? mlx_async_eval_timed(&destination.value, c_outputs, c_stream)
          : mlx_async_eval_with_event(&destination.value, c_outputs);

      CHECK(result != 0);
      CHECK(errors.count == 1);
      CHECK(prepared_before_submission);
      CHECK(destination.value.ctx == previous_storage);
      CHECK(completed.load() == 0);
      if (existing_storage) {
        CHECK(mlx_event_synchronize(destination.value) == 0);
      }
      // A timed call may accept its start marker before the failing primitive.
      // The recovery registry owns that prefix; no new sentinel can be queued
      // onto a poisoned stream, so observe its existing native frontier.
      const auto frontier = scheduler::cpu_stream_progress(token).accepted;
      auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (scheduler::cpu_stream_progress(token).completed < frontier &&
             std::chrono::steady_clock::now() < deadline) {
        submission::progress_records();
        std::this_thread::yield();
      }
      CHECK(scheduler::cpu_stream_progress(token).completed >= frontier);
      submission::progress_records();
    }
  }
}

TEST_CASE("test C host transfer prepares both publication outputs") {
  for (bool existing_buffer : {false, true}) {
    for (bool existing_event : {false, true}) {
      for (bool fail_before_work : {false, true}) {
        CAPTURE(existing_buffer);
        CAPTURE(existing_event);
        CAPTURE(fail_before_work);
        OwnedHostBuffer buffer;
        OwnedEvent event;
        if (existing_buffer) {
          initialize_host_buffer(buffer, 13.0f);
        }
        if (existing_event) {
          event.value = mlx_event_new();
        }
        const auto previous_buffer = buffer.value.ctx;
        const auto previous_event = event.value.ctx;
        auto stream = new_thread_unsafe_stream(Device::cpu);
        bool prepared_before_submission = false;
        std::atomic<int> completed{0};
        auto primitive = std::make_shared<PublicationProbe>(
            stream,
            event.value,
            prepared_before_submission,
            completed,
            fail_before_work,
            &buffer.value.ctx);
        auto source = array({}, float32, primitive, {array(7.0f)});
        const mlx_array c_source{&source};
        const mlx_stream c_stream{&stream};
        CountErrors errors;
        const int result = mlx_copy_to_host(
            &buffer.value,
            &event.value,
            c_source,
            MLX_HOST_TRANSFER_POLICY_TRANSFER,
            c_stream);

        CHECK(prepared_before_submission);
        if (fail_before_work) {
          CHECK(result != 0);
          CHECK(errors.count == 1);
          CHECK(buffer.value.ctx == previous_buffer);
          CHECK(event.value.ctx == previous_event);
          CHECK(completed.load() == 0);
          if (existing_buffer) {
            const void* data = nullptr;
            CHECK(mlx_host_transfer_buffer_data(&data, buffer.value) == 0);
            CHECK(*static_cast<const float*>(data) == 13.0f);
          }
        } else {
          CHECK(result == 0);
          CHECK(errors.count == 0);
          CHECK(mlx_event_synchronize(event.value) == 0);
          CHECK(completed.load() == 1);
          const void* data = nullptr;
          CHECK(mlx_host_transfer_buffer_data(&data, buffer.value) == 0);
          CHECK(*static_cast<const float*>(data) == 7.0f);
          if (existing_buffer) {
            CHECK(buffer.value.ctx == previous_buffer);
          }
          if (existing_event) {
            CHECK(event.value.ctx == previous_event);
          }
        }
        synchronize(stream);
      }
    }
  }
}

TEST_CASE("test C host transfer device publication retains source storage") {
  for (bool existing_array : {false, true}) {
    for (bool existing_event : {false, true}) {
      CAPTURE(existing_array);
      CAPTURE(existing_event);
      OwnedArray output;
      OwnedEvent event;
      if (existing_array) {
        output.value = mlx_array_new_(array(13.0f));
      }
      if (existing_event) {
        event.value = mlx_event_new();
      }
      const auto previous_array = output.value.ctx;
      const auto previous_event = event.value.ctx;
      auto stream = new_thread_unsafe_stream(Device::cpu);
      const mlx_stream c_stream{&stream};
      auto release_copy = std::make_shared<std::promise<void>>();
      auto copy_released = release_copy->get_future().share();
      scheduler::enqueue(stream, [copy_released]() { copy_released.wait(); });
      {
        OwnedHostBuffer source;
        initialize_host_buffer(source, 7.0f);
        CHECK(mlx_copy_from_host(
                  &output.value, &event.value, source.value, c_stream) == 0);
      }
      // The original C source owner is already gone. Submitted work and its
      // output must retain the actual backing storage independently.
      release_copy->set_value();
      CHECK(mlx_event_synchronize(event.value) == 0);
      CHECK(mlx_array_get_(output.value).item<float>() == 7.0f);
      if (existing_array) {
        CHECK(output.value.ctx == previous_array);
      }
      if (existing_event) {
        CHECK(event.value.ctx == previous_event);
      }
    }
  }
}

TEST_CASE("test C host transfer device failure rolls back paired outputs") {
  for (bool existing_array : {false, true}) {
    for (bool existing_event : {false, true}) {
      CAPTURE(existing_array);
      CAPTURE(existing_event);
      OwnedArray output;
      OwnedEvent event;
      if (existing_array) {
        output.value = mlx_array_new_(array(13.0f));
      }
      if (existing_event) {
        event.value = mlx_event_new();
      }
      const auto previous_array = output.value.ctx;
      const auto previous_event = event.value.ctx;
      auto stream = new_thread_unsafe_stream(Device::cpu);
      const mlx_stream c_stream{&stream};
      HostTransferBuffer empty_source;
      const mlx_host_transfer_buffer c_source{&empty_source};
      CountErrors errors;
      CHECK(mlx_copy_from_host(
                &output.value, &event.value, c_source, c_stream) != 0);
      CHECK(errors.count == 1);
      CHECK(output.value.ctx == previous_array);
      CHECK(event.value.ctx == previous_event);
      if (existing_array) {
        CHECK(mlx_array_get_(output.value).item<float>() == 13.0f);
      }
      if (existing_event) {
        CHECK(mlx_event_synchronize(event.value) == 0);
      }
    }
  }
}

TEST_CASE("test host transfer noexcept moves preserve heap backed shapes") {
  static_assert(std::is_nothrow_move_constructible_v<HostTransferBuffer>);
  static_assert(std::is_nothrow_move_assignable_v<HostTransferBuffer>);
  HostTransferBuffer original(Shape(11, 1), float32);
  auto* storage = original.data();
  *static_cast<float*>(storage) = 7.0f;
  auto copied = original;
  HostTransferBuffer copy_assigned;
  copy_assigned = original;
  HostTransferBuffer moved(std::move(original));
  HostTransferBuffer move_assigned(Shape(12, 1), float32);
  move_assigned = std::move(moved);

  CHECK(copied.data() == storage);
  CHECK(copy_assigned.data() == storage);
  CHECK(move_assigned.data() == storage);
  CHECK(move_assigned.shape() == Shape(11, 1));
  CHECK(*static_cast<float*>(move_assigned.data()) == 7.0f);
}

TEST_CASE("test C host transfer array reservation constructs no placeholder") {
  static_assert(!std::is_copy_constructible_v<mlx_array_output_preparation_>);
  static_assert(!std::is_copy_assignable_v<mlx_array_output_preparation_>);
  static_assert(!std::is_move_constructible_v<mlx_array_output_preparation_>);
  static_assert(!std::is_move_assignable_v<mlx_array_output_preparation_>);
  OwnedArray output;
  AllocationBudget reservation{1};
  bool stayed_empty = false;
  {
    TrackAllocations tracked(reservation);
    mlx_array_output_preparation_ preparation(output.value);
    stayed_empty = output.value.ctx == nullptr;
  }
  CHECK(stayed_empty);
  CHECK(output.value.ctx == nullptr);
  CHECK(reservation.attempts == 1);
  CHECK(reservation.frees == 1);

  AllocationBudget rejected{0};
  bool allocation_failed = false;
  {
    TrackAllocations tracked(rejected);
    try {
      mlx_array_output_preparation_ preparation(output.value);
    } catch (const std::bad_alloc&) {
      allocation_failed = true;
    }
  }
  CHECK(allocation_failed);
  CHECK(output.value.ctx == nullptr);
  CHECK(rejected.attempts == 1);
  CHECK(rejected.frees == 0);
}

TEST_CASE("test C host transfer array publication allocates nothing") {
  for (bool existing_array : {false, true}) {
    CAPTURE(existing_array);
    OwnedArray output;
    // Isolate wrapper publication from last-owner native-buffer destruction,
    // whose existing cache-recycling path may allocate cache entries.
    auto previous_value = array(13.0f);
    if (existing_array) {
      output.value = mlx_array_new_(previous_value);
    }
    const auto previous_storage = output.value.ctx;
    auto value = array(7.0f);
    const auto descriptor = value.id();
    AllocationBudget reservation{existing_array ? 0U : 1U};
    AllocationBudget publication{0};
    bool preserved_before_publication = false;
    {
      TrackAllocations tracked(reservation);
      mlx_array_output_preparation_ preparation(output.value);
      preserved_before_publication = output.value.ctx == previous_storage;
      {
        TrackAllocations tracked_publication(publication);
        preparation.publish(std::move(value));
      }
    }
    CHECK(preserved_before_publication);
    CHECK(reservation.attempts == (existing_array ? 0 : 1));
    CHECK(publication.attempts == 0);
    CHECK(mlx_array_get_(output.value).id() == descriptor);
    CHECK(mlx_array_get_(output.value).item<float>() == 7.0f);
    CHECK(previous_value.item<float>() == 13.0f);
    if (existing_array) {
      CHECK(output.value.ctx == previous_storage);
    }
  }
}

#include "doctest/doctest.h"

#include <atomic>
#include <chrono>
#include <future>
#include <optional>
#include <thread>

#include "mlx/c/submission.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/submission.h"

using namespace mlx::core;

// Implemented by the publication test translation unit's test-only allocator.
extern size_t native_recovery_without_allocations(void (*)(void*), void*);

namespace {
struct OwnedScope {
  mlx_submission_scope value{nullptr};
  OwnedScope() {
    if (mlx_submission_scope_new(&value)) {
      throw std::runtime_error("could not create test submission scope");
    }
  }
  ~OwnedScope() {
    if (value.ctx) {
      mlx_submission_scope_free(value);
    }
  }
  mlx_submission_status status() const {
    mlx_submission_status result{};
    REQUIRE(mlx_submission_scope_query(&result, value) == 0);
    return result;
  }
  mlx_submission_status progress() const {
    mlx_submission_status result{};
    REQUIRE(mlx_submission_scope_progress(&result, value) == 0);
    return result;
  }
  void drop() {
    REQUIRE(mlx_submission_scope_free(value) == 0);
    value.ctx = nullptr;
  }
};

struct Work {
  std::promise<void> release;
  std::shared_future<void> gate{release.get_future().share()};
  std::atomic<bool> opened{false};
  std::atomic<bool> started{false};
  std::atomic<bool> finished{false};
  std::atomic<int> destroyed{0};
  std::atomic<float> observed{0};
  void* allocation{nullptr};
  void open() {
    if (!opened.exchange(true)) {
      release.set_value();
    }
  }
};

struct Gate {
  std::shared_ptr<Work> work{std::make_shared<Work>()};
  ~Gate() { work->open(); }
};

class PartialPrimitive : public Primitive {
 public:
  PartialPrimitive(Stream stream, std::shared_ptr<Work> work, bool fail)
      : Primitive(stream), work_(std::move(work)), fail_(fail) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>& outputs) override {
    // This temporary is not a graph input/output when the primitive throws.
    // A task with a raw pointer is accepted before the later synchronous error.
    auto temp = array(Shape{1}, float32, nullptr, {});
    temp.set_data(allocator::malloc(sizeof(float)), [work = work_](auto buffer) {
      ++work->destroyed;
      allocator::free(buffer);
    });
    *temp.data<float>() = 17.0f;
    work_->allocation = temp.data<float>();
    scheduler::enqueue(stream(), [work = work_, pointer = temp.data<float>()] {
      work->started.store(true, std::memory_order_release);
      work->gate.wait();
      work->observed.store(*pointer, std::memory_order_release);
      work->finished.store(true, std::memory_order_release);
    });
    if (fail_) {
      throw std::bad_alloc();
    }
    outputs[0].copy_shared_buffer(temp);
  }
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("CPU-only recovery test primitive");
  }
  DEFINE_NAME(PartialPrimitive);

 private:
  std::shared_ptr<Work> work_;
  bool fail_;
};

class LateFailurePrimitive : public Primitive {
 public:
  LateFailurePrimitive(Stream stream, std::shared_ptr<Work> trigger,
                       std::shared_ptr<Work> pending)
      : Primitive(stream), trigger_(std::move(trigger)), pending_(std::move(pending)) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>& outputs) override {
    outputs[0].set_data(allocator::malloc(sizeof(float)), [pending = pending_](auto buffer) {
      ++pending->destroyed;
      allocator::free(buffer);
    });
    *outputs[0].data<float>() = 23.0f;
    scheduler::enqueue(stream(), [trigger = trigger_] {
      trigger->gate.wait();
      throw std::runtime_error("late producer failure after host wait starts");
    });
    scheduler::enqueue(stream(), [pending = pending_, pointer = outputs[0].data<float>()] {
      pending->started = true;
      pending->gate.wait();
      pending->observed = *pointer;
      pending->finished = true;
    });
  }
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("CPU-only recovery test primitive");
  }
  DEFINE_NAME(LateFailurePrimitive);
 private:
  std::shared_ptr<Work> trigger_;
  std::shared_ptr<Work> pending_;
};

array partial(Stream stream, const Gate& gate, bool fail) {
  return array(
      Shape{1}, float32,
      std::make_shared<PartialPrimitive>(stream, gate.work, fail), {});
}

bool settled(const OwnedScope& scope) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < deadline) {
    if (scope.progress().activity != MLX_SUBMISSION_ACTIVITY_PENDING) {
      submission::retire_records();
      return true;
    }
    std::this_thread::yield();
  }
  return false;
}

bool wait_finished(const std::shared_ptr<Work>& work) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < deadline) {
    if (work->finished.load(std::memory_order_acquire)) {
      return true;
    }
    std::this_thread::yield();
  }
  return false;
}
} // namespace

TEST_CASE("submission recovery retains partial primitive temporary until terminal") {
  OwnedScope scope;
  Gate gate;
  auto stream = new_stream(Device::cpu);
  {
    auto output = partial(stream, gate, true);
    CHECK_THROWS_AS(async_eval_with_completion({output}), std::bad_alloc);
  }
  REQUIRE(mlx_submission_scope_seal(scope.value) == 0);
  auto pending = scope.progress();
  CHECK(pending.activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  CHECK(pending.failed);
  CHECK(pending.blocked);
  CHECK(gate.work->destroyed == 0);
  CHECK_THROWS(scheduler::enqueue(stream, [] {}));
  gate.work->open();
  CHECK(settled(scope));
  CHECK(scope.status().activity == MLX_SUBMISSION_ACTIVITY_TERMINAL);
  CHECK(gate.work->observed == 17.0f);
  CHECK(gate.work->destroyed == 1);
}

TEST_CASE("submission recovery preserves success donation before later child failure") {
  OwnedScope scope;
  Gate successful;
  Gate failing;
  auto stream = new_stream(Device::cpu);
  std::optional<array> first(partial(stream, successful, false));
  auto donated = negative(*first, stream);
  first.reset();
  auto second_completion = async_eval_with_completion({donated});
  CHECK(donated.data<float>() == successful.work->allocation);
  CHECK(donated.data_shared_ptr()->recovery_pins == 0);
  CHECK(successful.work->destroyed == 0);
  {
    auto failed = partial(stream, failing, true);
    CHECK_THROWS_AS(async_eval_with_completion({failed}), std::bad_alloc);
  }
  CHECK(scope.progress().activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  CHECK(successful.work->destroyed == 0);
  CHECK(failing.work->destroyed == 0);
  successful.work->open();
  CHECK(wait_finished(successful.work));
  CHECK(scope.progress().activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  failing.work->open();
  CHECK(settled(scope));
  CHECK(donated.item<float>() == -17.0f);
  CHECK(failing.work->destroyed == 1);
}

TEST_CASE("submission recovery keeps output pins when a later primitive in the same eval fails") {
  OwnedScope scope;
  Gate successful;
  Gate failing;
  auto stream = new_stream(Device::cpu);
  std::optional<array> first(partial(stream, successful, false));
  auto donated = negative(*first, stream);
  first.reset();
  {
    auto failed = array(
        Shape{1}, float32,
        std::make_shared<PartialPrimitive>(stream, failing.work, true),
        {donated});
    CHECK_THROWS_AS(async_eval_with_completion({failed}), std::bad_alloc);
  }
  CHECK(donated.data<float>() == successful.work->allocation);
  CHECK(donated.data_shared_ptr()->recovery_pins > 0);
  CHECK(scope.progress().activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  CHECK(successful.work->destroyed == 0);
  CHECK(failing.work->destroyed == 0);
  successful.work->open();
  CHECK(wait_finished(successful.work));
  failing.work->open();
  CHECK(settled(scope));
  CHECK(donated.data_shared_ptr()->recovery_pins == 0);
  // This node shares the failed evaluation's event, so ordinary value access
  // must keep reporting that failure. Terminal recovery permits inspecting the
  // still-owned allocation without requesting evaluation again.
  CHECK_THROWS(donated.item<float>());
  CHECK(*donated.data<float>() == -17.0f);
  CHECK(failing.work->destroyed == 1);
}

TEST_CASE("submission recovery waits for every stream and survives dropped scope") {
  OwnedScope outer;
  Gate first;
  Gate second;
  {
    OwnedScope inner;
    auto first_output = partial(new_stream(Device::cpu), first, true);
    CHECK_THROWS_AS(async_eval_with_completion({first_output}), std::bad_alloc);
    auto second_output = partial(new_stream(Device::cpu), second, true);
    CHECK_THROWS_AS(async_eval_with_completion({second_output}), std::bad_alloc);
    inner.drop();
  }
  CHECK(outer.progress().activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  first.work->open();
  CHECK(wait_finished(first.work));
  CHECK(outer.progress().activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  CHECK(second.work->destroyed == 0);
  second.work->open();
  CHECK(settled(outer));
  CHECK(first.work->destroyed == 1);
  CHECK(second.work->destroyed == 1);
}

TEST_CASE("submission recovery supports out of order scope release") {
  OwnedScope outer;
  OwnedScope inner;
  outer.drop();
  Gate gate;
  {
    auto output = partial(new_stream(Device::cpu), gate, true);
    CHECK_THROWS_AS(async_eval_with_completion({output}), std::bad_alloc);
  }
  CHECK(inner.progress().activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  gate.work->open();
  CHECK(settled(inner));
  CHECK(gate.work->destroyed == 1);
}

TEST_CASE("submission recovery active child excludes premature outer settlement") {
  OwnedScope outer;
  OwnedScope child;
  REQUIRE(mlx_submission_scope_seal(outer.value) == 0);
  CHECK(outer.status().activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  CHECK(child.status().activity == MLX_SUBMISSION_ACTIVITY_NONE);
  Gate gate;
  {
    auto output = partial(new_stream(Device::cpu), gate, true);
    CHECK_THROWS_AS(async_eval_with_completion({output}), std::bad_alloc);
  }
  REQUIRE(mlx_submission_scope_seal(child.value) == 0);
  CHECK(outer.progress().activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  gate.work->open();
  CHECK(settled(outer));
  CHECK(child.status().activity == MLX_SUBMISSION_ACTIVITY_TERMINAL);
}

TEST_CASE("submission recovery concurrent physical pins never hide ordinary owners") {
  auto source = array(3.0f);
  {
    submission::Record physical_only;
    physical_only.reserve_outputs(1);
    physical_only.retain_output(source.data_shared_ptr());
    CHECK(source.is_donatable());
  }
  auto alias = array(Shape{}, float32, nullptr, {});
  alias.copy_shared_buffer(source);
  std::atomic<bool> finished{false};
  std::thread mutator([&] {
    for (int i = 0; i < 20000; ++i) {
      submission::Record pins;
      pins.reserve_outputs(4);
      for (int j = 0; j < 4; ++j) {
        pins.retain_output(source.data_shared_ptr());
      }
    }
    finished.store(true, std::memory_order_release);
  });
  bool incorrectly_donatable = false;
  while (!finished.load(std::memory_order_acquire)) {
    incorrectly_donatable |= source.is_donatable();
  }
  mutator.join();
  CHECK_FALSE(incorrectly_donatable);
  CHECK(source.data_shared_ptr()->recovery_pins == 0);
  CHECK_FALSE(source.data_shared_ptr()->recovery_pin_lock.test_and_set());
  source.data_shared_ptr()->recovery_pin_lock.clear();
}

TEST_CASE("submission recovery donation declines a contended pin snapshot without waiting") {
  auto source = array(3.0f);
  submission::Record physical_only;
  physical_only.reserve_outputs(1);
  physical_only.retain_output(source.data_shared_ptr());
  CHECK(source.is_donatable());
  auto* data = source.data_shared_ptr().get();
  CHECK_FALSE(data->recovery_pin_lock.test_and_set());
  // The pin mutator owns this guard on the same thread. Blocking acquisition
  // would deadlock, while a coherent nonblocking snapshot must decline.
  CHECK_FALSE(source.is_donatable());
  data->recovery_pin_lock.clear();
  CHECK(source.is_donatable());
  auto alias = array(Shape{}, float32, nullptr, {});
  alias.copy_shared_buffer(source);
  CHECK_FALSE(source.is_donatable());
}

TEST_CASE("submission recovery host wait observes late failure without releasing pending payload") {
  OwnedScope scope;
  Gate trigger;
  Gate pending;
  auto stream = new_stream(Device::cpu);
  std::optional<array> output(std::in_place, Shape{1}, float32,
      std::make_shared<LateFailurePrimitive>(stream, trigger.work, pending.work),
      std::vector<array>{});
  auto completion = async_eval_with_completion({*output});
  CHECK_FALSE(scope.progress().failed);
  CHECK_FALSE(completion.is_complete());
  std::promise<void> returned;
  auto observed_return = returned.get_future();
  std::atomic<bool> watchdog_released{false};
  std::thread releaser([&] {
    trigger.work->open();
    // Bound even a regressed native blocking wait: the watchdog frees it and
    // the assertions fail, instead of leaving the test process hung.
    if (observed_return.wait_for(std::chrono::milliseconds(500)) !=
        std::future_status::ready) {
      watchdog_released = true;
      pending.work->open();
    }
  });
  CHECK_THROWS_AS(completion.wait(), std::runtime_error);
  returned.set_value();
  releaser.join();
  CHECK_FALSE(watchdog_released);
  output.reset();
  CHECK(scope.progress().activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  CHECK(pending.work->destroyed == 0);
  pending.work->open();
  CHECK(settled(scope));
  CHECK(pending.work->observed == 23.0f);
  CHECK(pending.work->destroyed == 1);
}

TEST_CASE("submission recovery registry owns work without a scope handle") {
  Gate gate;
  {
    OwnedScope scope;
    auto output = partial(new_stream(Device::cpu), gate, true);
    CHECK_THROWS_AS(async_eval_with_completion({output}), std::bad_alloc);
    scope.drop();
  }
  CHECK(gate.work->destroyed == 0);
  gate.work->open();
  CHECK(wait_finished(gate.work));
  OwnedScope observer;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (gate.work->destroyed == 0 && std::chrono::steady_clock::now() < deadline) {
    observer.progress();
    submission::retire_records();
    std::this_thread::yield();
  }
  CHECK(gate.work->destroyed == 1);
  CHECK(observer.status().activity == MLX_SUBMISSION_ACTIVITY_NONE);
}

TEST_CASE("submission recovery counts ordering waits on an independent consumer scope") {
  OwnedScope producer;
  Gate gate;
  std::optional<array> output(partial(new_stream(Device::cpu), gate, false));
  auto completion = async_eval_with_completion({*output});
  REQUIRE(mlx_submission_scope_seal(producer.value) == 0);
  OwnedScope consumer;
  completion.wait(new_stream(Device::cpu));
  REQUIRE(mlx_submission_scope_seal(consumer.value) == 0);
  output.reset();
  CHECK(consumer.progress().activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  CHECK(gate.work->destroyed == 0);
  gate.work->open();
  CHECK(settled(consumer));
  CHECK(settled(producer));
  CHECK(gate.work->destroyed == 1);
}

TEST_CASE("submission recovery terminal cleanup tolerates secondary allocation failure") {
  OwnedScope scope;
  Gate gate;
  {
    auto output = partial(new_stream(Device::cpu), gate, true);
    CHECK_THROWS_AS(async_eval_with_completion({output}), std::bad_alloc);
  }
  gate.work->open();
  REQUIRE(wait_finished(gate.work));
  struct Observation {
    mlx_submission_scope scope;
    mlx_submission_status status{};
    int result{1};
  } observation{scope.value};
  size_t attempts = 0;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < deadline) {
    attempts += native_recovery_without_allocations([](void* context) {
      auto& observation = *static_cast<Observation*>(context);
      observation.result = mlx_submission_scope_progress(
          &observation.status, observation.scope);
      if (observation.status.activity == MLX_SUBMISSION_ACTIVITY_TERMINAL) {
        submission::retire_records();
      }
    }, &observation);
    if (observation.status.activity == MLX_SUBMISSION_ACTIVITY_TERMINAL) {
      break;
    }
    std::this_thread::yield();
  }
  CHECK(observation.result == 0);
  CHECK(observation.status.activity == MLX_SUBMISSION_ACTIVITY_TERMINAL);
  CHECK(observation.status.failed);
  CHECK(gate.work->destroyed == 1);
  CHECK(attempts > 0); // Optional buffer-cache insertion failed and fell back.
}

TEST_CASE("submission recovery scheduler throttle rejects known failed blocked progress") {
  OwnedScope scope;
  Gate producer;
  Gate counted;
  Gate current;
  auto counted_stream = new_stream(Device::cpu);
  auto counted_token = scheduler::prepare_cpu_stream(counted_stream);
  for (int i = 0; i < 12; ++i) {
    scheduler::enqueue_counted(counted_stream, [work = counted.work] {
      work->gate.wait();
    });
  }
  {
    auto failed = partial(new_stream(Device::cpu), producer, true);
    CHECK_THROWS_AS(async_eval_with_completion({failed}), std::bad_alloc);
  }
  std::promise<void> returned;
  auto observed_return = returned.get_future();
  std::atomic<bool> watchdog_released{false};
  std::thread watchdog([&] {
    if (observed_return.wait_for(std::chrono::milliseconds(500)) !=
        std::future_status::ready) {
      watchdog_released = true;
      producer.work->open();
      counted.work->open();
      current.work->open();
    }
  });
  {
    auto output = partial(new_stream(Device::cpu), current, false);
    CHECK_THROWS_WITH_AS(async_eval_with_completion({output}),
        "Native scheduler progress is blocked by an unresolved failed submission.",
        std::runtime_error);
  }
  returned.set_value();
  watchdog.join();
  CHECK_FALSE(watchdog_released);
  CHECK(current.work->destroyed == 0);
  producer.work->open();
  current.work->open();
  counted.work->open();
  CHECK(settled(scope));
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  const auto frontier = scheduler::cpu_stream_progress(counted_token).accepted;
  while (scheduler::cpu_stream_progress(counted_token).completed < frontier &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  CHECK(scheduler::cpu_stream_progress(counted_token).completed >= frontier);
  CHECK(current.work->destroyed == 1);
}

TEST_CASE("submission recovery secondary native receipt controls terminal ownership") {
  struct Effects {
    std::atomic<bool> terminal{false};
    std::atomic<bool> failed{false};
    std::atomic<int> destroyed{0};
  } effects;
  struct SecondaryRecord : submission::Record {
    explicit SecondaryRecord(Effects& value) : effects(value) {}
    ~SecondaryRecord() override { ++effects.destroyed; }
    bool side_effects_terminal() const noexcept override {
      return effects.terminal.load(std::memory_order_acquire);
    }
    bool side_effects_failed() const noexcept override {
      return effects.failed.load(std::memory_order_acquire);
    }
    Effects& effects;
  };
  OwnedScope scope;
  auto stream = new_stream(Device::cpu);
  auto token = scheduler::prepare_cpu_stream(stream);
  auto* record = new SecondaryRecord(effects);
  record->enter();
  record->prepare_stream(stream);
  scheduler::enqueue(stream, [] {});
  record->finish(false);
  const auto frontier = scheduler::cpu_stream_progress(token).accepted;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (scheduler::cpu_stream_progress(token).completed < frontier &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  REQUIRE(scheduler::cpu_stream_progress(token).completed >= frontier);
  CHECK(scope.progress().activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  effects.failed = true;
  auto unresolved = scope.progress();
  CHECK(unresolved.activity == MLX_SUBMISSION_ACTIVITY_PENDING);
  CHECK(unresolved.failed);
  CHECK(unresolved.blocked);
  CHECK(effects.destroyed == 0);
  effects.terminal = true;
  // A worker may publish terminal evidence, but cannot destroy host owners.
  std::thread worker([] { submission::progress_records(false); });
  worker.join();
  CHECK(scope.status().activity == MLX_SUBMISSION_ACTIVITY_TERMINAL);
  CHECK(effects.destroyed == 0);
  scope.progress();
  CHECK(effects.destroyed == 0);
  submission::retire_records();
  CHECK(effects.destroyed == 1);
}

TEST_CASE("submission recovery owner identity never follows recycled thread ids") {
  submission::Scope* retired_scope = nullptr;
  std::thread::id retired_id;
  std::thread original([&] {
    retired_scope = new submission::Scope;
    retired_id = std::this_thread::get_id();
    retired_scope->seal();
  });
  original.join();
  bool id_reused = false;
  bool unauthorized = false;
  for (int i = 0; i < 32; ++i) {
    std::thread replacement([&] {
      auto* current = new submission::Scope;
      id_reused |= std::this_thread::get_id() == retired_id;
      unauthorized |= retired_scope->on_owner_thread();
      current->seal();
      current->release();
    });
    replacement.join();
  }
  CAPTURE(id_reused); // Reuse is OS-dependent, but may never grant authority.
  CHECK_FALSE(unauthorized);
  CHECK_FALSE(retired_scope->on_owner_thread());
  retired_scope->release(); // Sealed metadata has no thread-affine resources.
}

TEST_CASE("submission recovery scope progress defers arbitrary primitive destruction") {
  class BlockingRetirement : public UnaryPrimitive {
   public:
    BlockingRetirement(Stream stream, std::shared_ptr<Work> work)
        : UnaryPrimitive(stream), work_(std::move(work)) {}
    ~BlockingRetirement() override {
      work_->started = true;
      work_->gate.wait();
      ++work_->destroyed;
    }
    void eval_cpu(const std::vector<array>& inputs, array& output) override {
      output.copy_shared_buffer(inputs[0]);
    }
    void eval_gpu(const std::vector<array>&, array&) override {
      throw std::runtime_error("CPU-only retirement test primitive");
    }
    DEFINE_NAME(BlockingRetirement);
   private:
    std::shared_ptr<Work> work_;
  };
  OwnedScope scope;
  Gate retirement;
  auto stream = new_stream(Device::cpu);
  auto output = array(Shape{}, float32,
      std::make_shared<BlockingRetirement>(stream, retirement.work), {array(1.0f)});
  auto completion = async_eval_with_completion({output});
  std::promise<void> returned;
  auto observed_return = returned.get_future();
  std::atomic<bool> watchdog_released{false};
  std::thread watchdog([&] {
    if (observed_return.wait_for(std::chrono::milliseconds(500)) !=
        std::future_status::ready) {
      watchdog_released = true;
      retirement.work->open();
    }
  });
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  mlx_submission_status status{};
  do {
    status = scope.progress();
    std::this_thread::yield();
  } while (status.activity == MLX_SUBMISSION_ACTIVITY_PENDING &&
           std::chrono::steady_clock::now() < deadline);
  returned.set_value();
  watchdog.join();
  CHECK_FALSE(watchdog_released);
  CHECK(status.activity == MLX_SUBMISSION_ACTIVITY_TERMINAL);
  CHECK_FALSE(retirement.work->started);
  scope.drop();
  CHECK_FALSE(retirement.work->started);
  retirement.work->open();
  submission::retire_records();
  CHECK(retirement.work->started);
  CHECK(retirement.work->destroyed == 1);
}

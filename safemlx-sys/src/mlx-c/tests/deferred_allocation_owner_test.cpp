// Allocation-owner publication without eager evaluation, including native
// submission failures and attachment while a CPU worker still uses backing.
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>

#include "mlx/c/array.h"
#include "mlx/c/error.h"
#include "mlx/c/private/array.h"
#include "mlx/c/submission.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/submission.h"

using namespace mlx::core;

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

void release_counter(void* pointer) {
  static_cast<std::atomic<int>*>(pointer)->fetch_add(1);
}

struct Handle {
  mlx_array value;
  explicit Handle(const array& input) : value(mlx_array_new_(input)) {}
  ~Handle() { mlx_array_free(value); }
};

void attach(const array& value, std::atomic<int>& releases) {
  Handle handle(value);
  bool attached = false;
  require(mlx_array_retain_deferred_allocation_owner(
              &attached, handle.value, &releases, release_counter) == 0,
          "deferred attachment failed");
  require(attached, "deferred attachment did not consume the payload");
}

void unrecognized_backing_preserves_caller_and_descriptor_authority() {
  std::atomic<int> rejected_releases{0};
  {
    const auto source = array(3.0f);
    auto weak = array::unsafe_weak_copy(source);
    Handle handle(weak);
    bool attached = true;
    require(mlx_array_retain_deferred_allocation_owner(
                &attached, handle.value, &rejected_releases, release_counter) != 0,
            "borrowed weak backing accepted an owner");
    require(!attached && rejected_releases == 0,
            "failed borrowed attachment consumed its payload");
  }
  require(rejected_releases == 0, "failed payload was armed during teardown");

  std::atomic<int> deferred_releases{0};
  int freed = 0;
  {
    auto future = array(Shape{1}, float32, nullptr, {});
    attach(future, deferred_releases);
    require(!future.data_shared_ptr(), "attachment materialized the descriptor");
    bool rejected = false;
    try {
      future.set_data(allocator::malloc(sizeof(float)), [&](auto buffer) {
        ++freed;
        allocator::free(buffer);
      });
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected && freed == 1, "uncertified future backing was not released");
    require(!future.data_shared_ptr(), "failed materialization installed backing");
    require(deferred_releases == 0, "failed materialization released authority");
  }
  require(deferred_releases == 1, "failed descriptor did not retire authority");
}

void shared_buffer_donation_and_late_attachment_preserve_owners() {
  std::atomic<int> releases{0};
  std::optional<array> source(array(7.0f));
  {
    auto destination = array(Shape{1}, float32, nullptr, {});
    attach(destination, releases);
    destination.copy_shared_buffer(*source);
    require(destination.data_shared_ptr() == source->data_shared_ptr(),
            "shared-buffer publication copied instead of sharing backing");
    require(destination.data_shared_ptr()->allocation_generation ==
                source->data_shared_ptr()->allocation_generation,
            "publication changed the shared allocation generation");
    attach(destination, releases);
    require(releases == 0, "live aliases lost attached authority");
  }
  require(releases == 0, "source alias did not preserve destination authority");
  source.reset();
  require(releases == 2, "shared backing did not retire both owners exactly once");
}

struct Work {
  std::promise<void> release;
  std::shared_future<void> gate{release.get_future().share()};
  std::atomic<bool> opened{false};
  std::atomic<bool> started{false};
  std::atomic<bool> finished{false};
  std::atomic<float> observed{0};
  void open() {
    if (!opened.exchange(true)) release.set_value();
  }
};

struct Gate {
  std::shared_ptr<Work> work{std::make_shared<Work>()};
  ~Gate() { work->open(); }
};

struct Scope {
  mlx_submission_scope value{nullptr};
  Scope() {
    require(mlx_submission_scope_new(&value) == 0, "scope creation failed");
  }
  ~Scope() { mlx_submission_scope_free(value); }
  mlx_submission_status progress() const {
    mlx_submission_status result{};
    require(mlx_submission_scope_progress(&result, value) == 0,
            "scope progress failed");
    return result;
  }
};

class GatedPrimitive : public Primitive {
 public:
  GatedPrimitive(Stream stream, std::shared_ptr<Work> work, bool fail)
      : Primitive(stream), work_(std::move(work)), fail_(fail) {}
  void eval_cpu(const ArrayVector&, ArrayVector& outputs) override {
    outputs[0].set_data(allocator::malloc(sizeof(float)));
    *outputs[0].data<float>() = 23.0f;
    scheduler::enqueue(stream(), [work = work_, pointer = outputs[0].data<float>()] {
      work->started.store(true, std::memory_order_release);
      work->gate.wait();
      work->observed.store(*pointer, std::memory_order_release);
      work->finished.store(true, std::memory_order_release);
    });
    if (fail_) throw std::runtime_error("failure after accepting native work");
  }
  void eval_gpu(const ArrayVector&, ArrayVector&) override {
    throw std::runtime_error("CPU-only deferred-owner test primitive");
  }
  DEFINE_NAME(GatedPrimitive);

 private:
  std::shared_ptr<Work> work_;
  bool fail_;
};

void await_started(const Gate& gate) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!gate.work->started.load(std::memory_order_acquire)) {
    require(std::chrono::steady_clock::now() < deadline, "worker did not start");
    std::this_thread::yield();
  }
}

void await_retired(const Scope& scope, const Gate& gate, std::atomic<int>& releases) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (scope.progress().activity == MLX_SUBMISSION_ACTIVITY_PENDING || releases == 0) {
    submission::retire_records();
    require(std::chrono::steady_clock::now() < deadline, "authority did not retire");
    std::this_thread::yield();
  }
  require(gate.work->finished && gate.work->observed == 23.0f,
          "accepted worker lost its physical backing");
  require(releases == 1, "native authority did not retire exactly once");
}

void failed_submission_retains_materialized_authority() {
  std::atomic<int> releases{0};
  Scope scope;
  Gate gate;
  auto stream = new_stream(Device::cpu);
  {
    auto output = array(Shape{1}, float32,
                        std::make_shared<GatedPrimitive>(stream, gate.work, true), {});
    attach(output, releases);
    bool failed = false;
    try {
      async_eval_with_completion({output});
    } catch (const std::runtime_error&) {
      failed = true;
    }
    require(failed, "native test failure did not propagate");
  }
  require(mlx_submission_scope_seal(scope.value) == 0, "scope sealing failed");
  await_started(gate);
  const auto status = scope.progress();
  require(status.failed && status.activity == MLX_SUBMISSION_ACTIVITY_PENDING,
          "failed accepted work was not retained as pending");
  require(releases == 0, "failed submission released authority before its worker");
  gate.work->open();
  await_retired(scope, gate, releases);
}

void attachment_during_pending_work_does_not_wait_or_race_backing_retirement() {
  std::atomic<int> releases{0};
  Scope scope;
  Gate gate;
  auto stream = new_stream(Device::cpu);
  {
    auto output = array(Shape{1}, float32,
                        std::make_shared<GatedPrimitive>(stream, gate.work, false), {});
    auto completion = async_eval_with_completion({output});
    await_started(gate);
    const auto status = output.status();
    const auto generation = output.data_shared_ptr()->allocation_generation;
    attach(output, releases);
    require(!gate.work->finished && output.status() == status,
            "pending attachment waited or changed completion status");
    require(output.data_shared_ptr()->allocation_generation == generation,
            "pending attachment replaced physical backing");
  }
  require(mlx_submission_scope_seal(scope.value) == 0, "scope sealing failed");
  require(releases == 0, "pending worker lost its newly attached authority");
  gate.work->open();
  await_retired(scope, gate, releases);
}
} // namespace

int main() {
  mlx_set_error_handler([](const char*, void*) {}, nullptr, nullptr);
  unrecognized_backing_preserves_caller_and_descriptor_authority();
  shared_buffer_donation_and_late_attachment_preserve_owners();
  failed_submission_retains_materialized_authority();
  attachment_during_pending_work_does_not_wait_or_race_backing_retirement();
}

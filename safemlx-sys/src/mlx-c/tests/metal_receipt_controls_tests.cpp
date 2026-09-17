#include "doctest/doctest.h"
#include "mlx/backend/metal/device.h"
#include "mlx/c/event.h"
#include "mlx/c/submission.h"
#include "mlx/c/private/array.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/submission.h"
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

using namespace mlx::core;
using namespace mlx::core::submission;
namespace {
struct FailureCustody {};
FailureCarrierRef make_failure() {
  auto owner = std::make_unique<FailureCustody>();
  auto failure = FailureCarrier::create(owner.get(), [](void* raw) {
    delete static_cast<FailureCustody*>(raw);
  });
  REQUIRE(failure);
  owner.release();
  return failure;
}
struct GraphRelease {
  void operator()(GraphQuota* value) const noexcept { value->release(); }
};
struct RecordRelease {
  void operator()(RecordQuota* value) const noexcept { value->release(); }
};
struct ScopeRelease {
  void operator()(Scope* value) const noexcept { value->seal(); value->release(); }
};
struct Owner {
  FailureCarrierRef failure{make_failure()};
  std::unique_ptr<GraphQuota, GraphRelease> graph{
      GraphQuota::create(4 << 20, nullptr, nullptr)};
  std::unique_ptr<RecordQuota, RecordRelease> records{
      RecordQuota::create(4 << 20, nullptr, nullptr)};
  std::unique_ptr<Scope, ScopeRelease> scope{
      new Scope(nullptr, nullptr, records.get(), graph.get())};
  Owner() {
    REQUIRE(scope->enable_scoped_observation());
    REQUIRE(scope->require_original_controls() == NativeControlFailure::none);
    REQUIRE(scope->bind_failure(failure));
    REQUIRE(scope->enable_original_controls() == NativeControlFailure::none);
  }
};
struct TestRecord final : Record {
  explicit TestRecord(Allocation allocation) : Record(allocation) {}
  bool scoped_observation_supported() const noexcept override { return true; }
};
TestRecord* entered() {
  auto record = Record::create<TestRecord>();
  record->enter();
  return record.release();
}
Stream prepared_stream() {
  auto stream = new_stream(Device::gpu);
  mlx_submission_runtime_baseline facts{};
  REQUIRE(mlx_submission_prepare_runtime(&facts, mlx_stream{&stream}, mlx_stream{&stream}) == 0);
  return stream;
}
template<class F> void await(F&& condition) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!condition()) {
    REQUIRE(std::chrono::steady_clock::now() < end);
    std::this_thread::yield();
  }
}
void settle(Owner& owner) {
  await([&] {
    REQUIRE(owner.scope->progress_scoped() == ScopedProgress::observed);
    REQUIRE_FALSE(owner.scope->query().failed);
    return owner.scope->query().activity != Activity::pending;
  });
  REQUIRE(owner.scope->retire_scoped_records() == ScopedProgress::observed);
}
struct ProbeAdd : Add {
  long& observed;
  ProbeAdd(Stream stream, long& observed) : Add(stream), observed(observed) {}
  void eval_gpu(const ArrayVector& inputs, array& output) override {
    observed = inputs.front().data_shared_ptr().use_count();
    Add::eval_gpu(inputs, output);
  }
};
using Counter = std::shared_ptr<std::atomic<unsigned>>;
Counter counter() { return std::make_shared<std::atomic<unsigned>>(0); }
struct SignalOnDrop {
  Counter count;
  explicit SignalOnDrop(Counter count) : count(std::move(count)) {}
  SignalOnDrop(SignalOnDrop&&) noexcept = default;
  ~SignalOnDrop() { if (count) ++*count; }
  void operator()(MTL::CommandBuffer*) noexcept {}
};
struct DelayedState {
  std::atomic<bool> started{false}, release{false};
  std::atomic<unsigned> destroyed{0}, tail{0};
};
}

// Source-local peer: inject the terminal callback at a chosen host-lifetime
// point using a real completed Metal buffer. This is not a device-OOM claim or
// a new production hook. Actual registered callbacks are covered separately.
namespace mlx::core::metal {
struct ReceiptTestAccess {
  static void terminal_before_host_return(CommandEncoder& encoder, MTL::CommandBuffer* buffer,
                                         std::atomic<unsigned>& destroyed) {
    auto host_alias = encoder.receipt_;
    auto* raw = host_alias.get();
    raw->callback_owner = host_alias;
    encoder.complete_receipt(raw, buffer);
    CHECK(destroyed == 1);
    CHECK(!raw->callback_owner);
    CHECK(encoder.submission_idle());
    CHECK(host_alias.use_count() == 2); // encoder + actual host temporary
    encoder.receipt_.reset();
    CHECK(host_alias.use_count() == 1);
  }
  static void ambiguous_commit_retains(CommandEncoder& encoder) {
    auto held = encoder.receipt_;
    held->callback_owner = held;
    encoder.commit_attempted_ = true;
    CHECK_THROWS(encoder.commit());
    CHECK(encoder.receipt_ == held);
    CHECK(held->callback_owner == held);
    CHECK_FALSE(encoder.submission_idle());
    // Undo only this injected pre-native ambiguous state, then let the actual
    // commit/callback complete the same retained prefix below.
    encoder.commit_attempted_ = false;
    held->callback_owner.reset();
  }
};
}

TEST_CASE("Metal receipt controls preserve nonzero multistream thresholds and exact window retirement") {
  auto first = prepared_stream();
  auto second = prepared_stream();
  array input({2.0f, -3.0f, 7.0f});
  eval(add(input, input, first));
  eval(add(input, input, second));
  Owner owner;
  const auto graph_before = owner.graph->occupied_bytes();
  {
    array output = input;
    // Exceeds default command-buffer thresholds and crosses real stream fences.
    for (unsigned i = 0; i != 128; ++i)
      output = add(output, input, (i / 64) ? second : first);
    mlx_submission_observer observer{};
    REQUIRE(mlx_submission_observer_current(&observer) == 0);
    mlx_operation_event completion{};
    REQUIRE(mlx_operation_event_new_exact(&completion, observer, mlx_stream{&second}, 1) == 0);
    REQUIRE(mlx_operation_event_append(completion, mlx_array{&output}) == 0);
    REQUIRE(mlx_operation_event_submit_on_stream(completion, mlx_stream{&second}) == 0);
    REQUIRE(mlx_operation_event_wait(completion) == 0);
    settle(owner);
    REQUIRE(mlx_submission_observer_validate_array(observer, mlx_array{&output}) == 0);
    CHECK(output.data<float>()[0] == 258.0f);
    CHECK(output.data<float>()[1] == -387.0f);
    CHECK(output.data<float>()[2] == 903.0f);
    mlx_operation_event_free(completion);
    mlx_submission_observer_release(observer);
  }
  settle(owner);
  CHECK(owner.graph->occupied_bytes() == graph_before);
  CHECK(owner.records->occupied_bytes() == 0);
}

TEST_CASE("Metal receipt destination preparation does not pin Data before the primitive donation decision") {
  auto stream = prepared_stream();
  array input({2.0f, -3.0f, 7.0f});
  eval(add(input, input, stream));
  Owner owner;
  {
    long observed = -1;
    auto output = add(input, input, stream);
    output.primitive_ptr() = make_graph_primitive<ProbeAdd>(stream, observed);
    auto* record = entered();
    {
      RecordDispatchGuard dispatch(*record);
      record->prepare_stream(stream);
      const auto before = input.data_shared_ptr().use_count();
      gpu::eval(output); // actual selected shared worker and actual Add kernel
      CHECK(observed == before);
      gpu::finalize(stream);
    }
    record->finish(false);
    settle(owner);
    CHECK(output.data<float>()[0] == 4.0f);
    CHECK(output.data<float>()[1] == -6.0f);
    CHECK(output.data<float>()[2] == 14.0f);
  }
  settle(owner);
  CHECK(owner.graph->occupied_bytes() == 0);
}

TEST_CASE("Metal receipt mixed ordinary and independent original roles keep their actual allocation owners") {
  auto stream = prepared_stream();
  auto encoder = metal::get_command_encoder_owner(stream);
  auto ordinary = counter(), first_done = counter(), second_done = counter();
  encoder->record_submission();
  encoder->add_completed_handler(SignalOnDrop(ordinary));
  Owner first;
  auto* a = entered();
  {
    RecordDispatchGuard dispatch(*a);
    a->prepare_stream(stream);
    encoder->record_submission();
    encoder->add_completed_handler(SignalOnDrop(first_done));
  }
  a->finish(false);
  CHECK(first.graph->occupied_bytes() > 0);
  first.scope->seal();
  {
    Owner second;
    auto* b = entered();
    {
      RecordDispatchGuard dispatch(*b);
      b->prepare_stream(stream);
      encoder->record_submission();
      encoder->add_completed_handler(SignalOnDrop(second_done));
      CHECK(*first_done == 0);
      CHECK(second.graph->occupied_bytes() > 0);
      encoder->commit();
    }
    b->finish(false);
    settle(second);
    CHECK(second.graph->occupied_bytes() == 0);
  }
  settle(first);
  CHECK(*ordinary == 1);
  CHECK(*first_done == 1);
  CHECK(*second_done == 1);
  CHECK(first.graph->occupied_bytes() == 0);
  // The next ordinary receipt is created at actual reuse, then still completes.
  encoder->record_submission();
  encoder->add_completed_handler(SignalOnDrop(ordinary));
  encoder->synchronize();
  CHECK(*ordinary == 2);
}

TEST_CASE("Metal receipt delayed callback keeps exact payloads and clears them before its frontier") {
  auto stream = prepared_stream();
  auto encoder = metal::get_command_encoder_owner(stream);
  auto state = std::make_shared<DelayedState>(); // cold, before original role
  Owner owner;
  struct ReleaseOnExit {
    std::shared_ptr<DelayedState> state;
    ~ReleaseOnExit() { state->release = true; }
  } cleanup{state};
  auto* record = entered();
  {
    RecordDispatchGuard dispatch(*record);
    record->prepare_stream(stream);
    encoder->record_submission();
    auto blocking = [state, payload = SignalOnDrop(Counter(state, &state->destroyed))]
        (MTL::CommandBuffer*) {
      state->started = true;
      while (!state->release.load()) std::this_thread::yield();
    };
    encoder->add_completed_handler(std::move(blocking));
    // The actual callback/Record retains this physical encoder until its
    // frontier finishes; do not borrow the local shared_ptr variable.
    encoder->add_completed_handler([state, physical = encoder.get()](MTL::CommandBuffer*) {
      CHECK(state->destroyed == 1);
      CHECK_FALSE(physical->submission_idle());
      ++state->tail;
    });
    encoder->commit();
  }
  record->finish(false);
  await([&] { return state->started.load(); });
  CHECK(state->destroyed == 0);
  CHECK(owner.graph->occupied_bytes() > 0);
  CHECK_FALSE(encoder->submission_idle());
  state->release = true;
  settle(owner);
  CHECK(state->destroyed == 1);
  CHECK(state->tail == 1);
  CHECK(owner.graph->occupied_bytes() == 0);
}

TEST_CASE("Metal receipt exact-owner and finite-capacity refusal precede acceptance and retain an existing prefix") {
  auto stream = prepared_stream();
  auto foreign = prepared_stream();
  auto encoder = metal::get_command_encoder_owner(stream);
  auto other = metal::get_command_encoder_owner(foreign);
  auto destroyed = counter();
  Owner owner;
  auto* record = entered();
  {
    RecordDispatchGuard dispatch(*record);
    record->prepare_stream(stream);
    const auto start = encoder->submission_progress().accepted;
    const auto graph_start = owner.graph->occupied_bytes();
    CHECK_THROWS_AS(other->prepare_completed_handler([](MTL::CommandBuffer*) {}), NativeControlError);
    CHECK(owner.graph->occupied_bytes() == graph_start);
    CHECK(encoder->submission_progress().accepted == start);
    std::array<unsigned char, 2048> capture{};
    auto refuse_large = [&](size_t leave) {
      auto bytes = owner.graph->capacity() - owner.graph->occupied_bytes() - leave;
      auto* occupied = owner.graph->try_allocate(bytes, 1);
      REQUIRE(occupied != nullptr);
      const auto accepted = encoder->submission_progress().accepted;
      CHECK_THROWS_AS(encoder->prepare_completed_handler(
          [capture](MTL::CommandBuffer*) { (void)capture; }), GraphQuotaError);
      CHECK(encoder->submission_progress().accepted == accepted);
      owner.graph->deallocate(occupied, bytes, 1);
    };
    refuse_large(256);
    encoder->record_submission();
    encoder->add_completed_handler(SignalOnDrop(destroyed));
    refuse_large(256);
    CHECK(*destroyed == 0);
    CHECK_FALSE(encoder->submission_idle());
    metal::ReceiptTestAccess::ambiguous_commit_retains(*encoder);
    CHECK(*destroyed == 0);
    encoder->commit();
  }
  record->finish(false);
  settle(owner);
  CHECK(*destroyed == 1);
  CHECK(owner.graph->occupied_bytes() == 0);
}

TEST_CASE("Metal receipt callback before host return drops payload but preserves actual host header custody") {
  auto stream = prepared_stream();
  auto encoder = metal::get_command_encoder_owner(stream);
  auto completed = NS::RetainPtr(encoder->get_command_buffer());
  encoder->synchronize(); // actual completed native command buffer for the injection
  auto destroyed = counter();
  Owner owner;
  auto* record = entered();
  {
    RecordDispatchGuard dispatch(*record);
    record->prepare_stream(stream);
    encoder->record_submission();
    encoder->add_completed_handler(SignalOnDrop(destroyed));
    metal::ReceiptTestAccess::terminal_before_host_return(*encoder, completed.get(), *destroyed);
    CHECK(owner.graph->occupied_bytes() == 0);
  }
  record->finish(false);
  settle(owner);
  CHECK(*destroyed == 1);
}

TEST_CASE("Metal receipt counted callback and exception race decrement the real scheduler exactly once") {
  auto stream = prepared_stream();
  auto* scheduler = scheduler::prepared_scheduler();
  REQUIRE(scheduler != nullptr);
  const auto before = scheduler->n_active_tasks();
  Owner owner;
  {
    auto counted = std::allocate_shared<metal::CountedCompletion>(
        GraphAllocator<metal::CountedCompletion>(owner.graph.get()), scheduler);
    scheduler->notify_new_task(stream);
    counted->counted = true;
    std::atomic<bool> start{false};
    std::thread callback([&] { while (!start.load()) {} counted->complete(); });
    std::thread exception([&] { while (!start.load()) {} counted->complete(); });
    start = true;
    callback.join(); exception.join();
    CHECK(scheduler->n_active_tasks() == before);
    CHECK_FALSE(counted->counted.load());
    counted->complete();
    CHECK(scheduler->n_active_tasks() == before);
  }
  CHECK(owner.graph->occupied_bytes() == 0);
  metal::ReceiptControlLayout receipt{};
  metal::BufferRetentionControlLayout buffers{};
  REQUIRE(metal::CommandEncoder::receipt_control_layout(receipt));
  REQUIRE(metal::buffer_retention_control_layout(buffers));
  CHECK(receipt.receipt.graph_extent >= receipt.receipt.bytes);
  CHECK(receipt.event_node.graph_extent >= receipt.event_node.bytes);
  CHECK(receipt.cleanup_node.graph_extent >= receipt.cleanup_node.bytes);
  CHECK(buffers.handler.graph_extent >= buffers.handler.bytes);
  CHECK(buffers.counted.graph_extent >= buffers.counted.bytes);
  CHECK(receipt.fixed_block_capture_bytes == 2 * sizeof(void*));
}

// Included by operation_observer_tests.cpp; uses its exact configured Owner.
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include <array>

namespace {
void prepare_evaluator_streams(std::array<Stream, 4>& streams) {
  for (auto& stream : streams) {
    mlx_submission_runtime_baseline runtime{};
    REQUIRE(mlx_submission_prepare_runtime(
        &runtime, mlx_stream{&stream}, mlx_stream{&stream}) == 0);
  }
  for (size_t i = 0; i != streams.size(); ++i)
    for (size_t j = i + 1; j != streams.size(); ++j)
      REQUIRE(streams[i] != streams[j]);
}

void original_evaluator_streams(std::array<Stream, 4> streams) {
  prepare_evaluator_streams(streams);
  array input({2.0f, -3.0f, 7.0f});
  eval(input);
  Owner owner;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  const auto graph_start = owner.graph->occupied_bytes();
  const auto record_start = owner.records->occupied_bytes();
  {
    auto left = add(input, input, streams[0]);
    auto right = multiply(input, input, streams[1]);
    auto joined = add(left, right, streams[2]);
    auto output = multiply(joined, input, streams[3]);
    auto selected = streams[3];
    mlx_operation_event event{};
    REQUIRE(mlx_operation_event_new_exact(
        &event, observer, mlx_stream{&selected}, 1) == 0);
    REQUIRE(mlx_operation_event_append(event, mlx_array{&output}) == 0);
    REQUIRE(mlx_operation_event_submit_on_stream(event, mlx_stream{&selected}) == 0);
    REQUIRE(mlx_operation_event_wait(event) == 0);
    await_original_terminal(owner);
    REQUIRE(mlx_submission_observer_validate_array(observer, mlx_array{&output}) == 0);
    const float expected[] = {16.0f, -9.0f, 441.0f};
    for (unsigned i = 0; i != 3; ++i) CHECK(output.data<float>()[i] == expected[i]);
    mlx_operation_event_free(event);
  }
  await_original_terminal(owner);
  REQUIRE(mlx_submission_observer_retire(observer) == 0);
  CHECK(owner.graph->occupied_bytes() == graph_start);
  CHECK(owner.records->occupied_bytes() == record_start);
  CHECK(owner.scope->active_on_owner_thread());
  mlx_submission_observer_release(observer);
}

// An actual evaluated view: no new data allocation or Record capture occurs
// inside this primitive. Its invocation counter distinguishes pre-dispatch
// quota refusal from a later stream-table allocation failure. CPU evaluation
// still submits its real cleanup task and all cross-stream Fence dependencies.
class EvaluatorViewProbe final : public UnaryPrimitive {
 public:
  EvaluatorViewProbe(Stream stream, std::atomic<unsigned>& visits)
      : UnaryPrimitive(stream), visits_(visits) {}
  void eval_cpu(const ArrayVector& inputs, array& output) override {
    ++visits_;
    output.copy_shared_buffer(inputs[0]);
  }
  void eval_gpu(const ArrayVector&, array&) override {
    throw std::logic_error("CPU-only evaluator view probe");
  }
  const char* name() const override { return "EvaluatorViewProbe"; }
 private:
  std::atomic<unsigned>& visits_;
};

void retire_evaluator_attempt(Owner& owner) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  for (;;) {
    REQUIRE(owner.scope->progress_scoped() == ScopedProgress::observed);
    if (owner.scope->query().activity != Activity::pending) break;
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
  REQUIRE(owner.scope->retire_scoped_records() == ScopedProgress::observed);
}
}

TEST_CASE("original evaluator preflights CPU stream table and retires nonzero dependencies") {
  original_evaluator_streams({new_stream(Device::cpu), new_stream(Device::cpu),
      new_stream(Device::cpu), new_stream(Device::cpu)});
}

TEST_CASE("original evaluator Record quota refuses before the first primitive across stream capacities") {
  std::array<Stream, 4> streams{new_stream(Device::cpu), new_stream(Device::cpu),
      new_stream(Device::cpu), new_stream(Device::cpu)};
  prepare_evaluator_streams(streams);
  array input({19.0f});
  eval(input);
  unsigned refused = 0;
  bool completed = false;
  // Actual arena limits, not guessed per-object sizes. The successful endpoint
  // and at least one exhausted attempt are both required. Every quota refusal
  // must occur before *any* of the eight actual primitive invocations.
  for (size_t capacity = 1024; capacity <= 32768 && !completed; capacity += 128) {
    CAPTURE(capacity);
    Owner owner(capacity);
    std::atomic<unsigned> visits{0};
    bool quota_refused = false;
    {
      auto root = input;
      for (unsigned i = 0; i != 8; ++i) {
        root = array(Shape{1}, float32,
            make_graph_primitive<EvaluatorViewProbe>(streams[i % streams.size()], visits),
            ArrayVector{root});
      }
      Completion completion;
      try {
        const auto outcome = async_eval_with_completion_scoped_on_stream(
            ArrayVector{root}, *owner.scope, completion, streams.back());
        REQUIRE(outcome == ScopedProgress::observed);
        REQUIRE(wait_scoped_completion(*owner.scope, completion) == ScopedEvaluation::complete);
        await_original_terminal(owner);
        REQUIRE(validate_scoped_array(root, *owner.scope) == ScopedEvaluation::complete);
        CHECK(root.data<float>()[0] == 19.0f);
        CHECK(visits.load() == 8);
        completed = true;
      } catch (const RecordQuotaError& error) {
        CHECK(error.cause() == QuotaFailure::exhausted);
        CHECK(visits.load() == 0);
        ++refused;
        quota_refused = true;
      }
    }
    retire_evaluator_attempt(owner);
    CHECK(owner.graph->occupied_bytes() == 0);
    CHECK(owner.records->occupied_bytes() == 0);
    CHECK(owner.scope->active_on_owner_thread());
    // A regression can block a worker after partial submission. Preserve the
    // failed assertion without reusing that poisoned worker for another attempt.
    if (quota_refused && visits.load() != 0) return;
  }
  CHECK(refused > 0);
  CHECK(completed);
}

#ifdef MLX_C_PATCH_TEST_METAL
TEST_CASE("original evaluator preflights Metal stream table and retires nonzero dependencies") {
  original_evaluator_streams({new_stream(Device::gpu), new_stream(Device::gpu),
      new_stream(Device::gpu), new_stream(Device::gpu)});
}

TEST_CASE("original evaluator preserves mixed device stream identity and nonzero Fence coherence") {
  // Stream IDs increase in creation order; alternating device order makes ID
  // and device ordering oppose each other. Full identity must remain ordered.
  original_evaluator_streams({new_stream(Device::gpu), new_stream(Device::cpu),
      new_stream(Device::gpu), new_stream(Device::cpu)});
}
#endif

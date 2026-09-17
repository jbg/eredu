#include "mlx/transforms_impl.h"
// Included after eval_record_layout_tests.cpp; all C headers precede the parent
// using-directive. These execute the prepared consumer, not a layout surrogate.
namespace eval_traversal_tests {
using namespace eval_record_facts;
using Limits = mlx_operation_eval_traversal_limits;
Limits pointwise() { return {1, 5, 3, 5, 3, 1, 8}; }
unsigned submit(Operation& operation, Stream stream, const Limits& limits) {
  return mlx_operation_event_submit_on_stream_prepared(
      operation.value, mlx_stream{&stream}, &limits);
}
void complete(Role& role, Operation& operation, array& value) {
  REQUIRE(mlx_operation_event_wait(operation.value) == 0);
  REQUIRE(validate_scoped_array(value, *role.scope) == ScopedEvaluation::complete);
  settle(role);
}
void round_trip(Device device, bool siblings) {
  const auto first = new_stream(device), second = siblings ? new_stream(device) : first;
  prepare(first, second);
  array left({13, 17, 19}), right({5, 7, 8});
  Role role;
  Operation operation;
  REQUIRE(mlx_operation_event_validate_traversal_leaf(operation.observer, mlx_array{&left}) == 0);
  REQUIRE(mlx_operation_event_validate_traversal_leaf(operation.observer, mlx_array{&right}) == 0);
  if (siblings) {
    auto both = divmod(left, right, first);
    REQUIRE(both.size() == 2);
    REQUIRE(both[0].siblings().size() == 1);
    auto value = add(both[0], both[1], second);
    operation.append(value);
    // Six descriptors (two leaves, two sibling outputs, Add, Synchronizer),
    // three tape entries, five actual visited edges, four output pins.
    REQUIRE(submit(operation, second, Limits{1, 6, 3, 5, 4, 2, 8}) == 0);
    complete(role, operation, value);
    CHECK(value.data<int>()[0] == 5);
    CHECK(value.data<int>()[1] == 5);
    CHECK(value.data<int>()[2] == 5);
  } else {
    auto value = multiply(add(left, right, first), left, first);
    operation.append(value);
    REQUIRE(submit(operation, first, pointwise()) == 0);
    complete(role, operation, value);
    CHECK(value.data<int>()[0] == 234);
    CHECK(value.data<int>()[1] == 408);
    CHECK(value.data<int>()[2] == 513);
  }
  CHECK(role.records->occupied_bytes() == 0);
}
struct FixedCapturePrefix : Primitive {
  std::shared_ptr<PrefixState> state;
  FixedCapturePrefix(Stream stream, std::shared_ptr<PrefixState> value)
      : Primitive(stream), state(std::move(value)) {}
  const char* name() const override { return "Prepared Eval capture limit"; }
  void eval_cpu(const ArrayVector&, ArrayVector& outputs) override {
    outputs[0].allocate_data(sizeof(float));
    auto temporary = array::with_allocated_buffer(sizeof(float), Shape{1}, float32);
    state->temporary = temporary.data_shared_ptr();
    state->output = outputs[0].data_shared_ptr();
    auto* source = temporary.data<float>();
    auto* destination = outputs[0].data<float>();
    cpu::get_command_encoder(stream()).dispatch([state = state, source, destination] {
      while (!state->release.load(std::memory_order_acquire)) std::this_thread::yield();
      *source = 37.f;
      *destination = *source + 5.f;
      state->completed.store(true, std::memory_order_release);
    });
    // Actual entered-Record capacity refusal. No injected exception and no
    // arena drain: the prepaid eight slots cannot grow into available space.
    for (size_t i = 0; i != 8; ++i) {
      auto extra = array::with_allocated_buffer(sizeof(float), Shape{1}, float32);
      (void)extra;
    }
    FAIL("fixed capture destination must refuse before a ninth publication");
  }
  void eval_gpu(const ArrayVector&, ArrayVector&) override { std::abort(); }
};
} // namespace eval_traversal_tests

TEST_CASE("prepared Eval traversal query binds eleven requests and rejects invalid limits intact"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace eval_traversal_tests;
  static_assert(sizeof(Limits) == 7 * sizeof(size_t));
  static_assert(sizeof(mlx_operation_eval_traversal_layout) == 32 * sizeof(size_t));
  static_assert(alignof(mlx_operation_eval_traversal_layout) == alignof(size_t));
  mlx_operation_eval_traversal_layout layout{};
  const auto limits = pointwise();
  REQUIRE(mlx_operation_event_eval_traversal_layout(&layout, &limits));
  CHECK(layout.record_allocations == 11);
  size_t sum = 0;
  for (size_t i = 0; i != 11; ++i) {
    CHECK(layout.request_bytes[i] > 0);
    CHECK(layout.request_alignments[i] > 0);
    sum += layout.request_bytes[i];
  }
  CHECK(sum == layout.record_requested_bytes);
  CHECK(layout.named_control_bytes > 0);
  const auto before = layout;
  for (const Limits invalid : {Limits{1, 0, 1, 1, 1, 1, 8},
      {1, 1, 2, 1, 2, 1, 8}, {1, 1, 1, 0, 1, 1, 8},
      {1, 1, 1, 1, 0, 1, 8}, {1, 1, 1, 1, 1, 0, 8},
      {1, 1, 1, 1, 1, 1, 7}, {1, SIZE_MAX, 1, 1, 1, 1, 8}}) {
    CHECK_FALSE(mlx_operation_event_eval_traversal_layout(&layout, &invalid));
    CHECK(std::memcmp(&layout, &before, sizeof(layout)) == 0);
  }
  CHECK_FALSE(mlx_operation_event_eval_traversal_layout(nullptr, &limits));
  CHECK_FALSE(mlx_operation_event_eval_traversal_layout(&layout, nullptr));
}

TEST_CASE("prepared Eval eleven real constructor requests refuse prefixes before dispatch"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace eval_traversal_tests;
  const auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  array left({2.f, 3.f, 5.f}), right({7.f, 11.f, 13.f});
  const auto limits = pointwise();
  mlx_operation_eval_traversal_layout layout{};
  REQUIRE(mlx_operation_event_eval_traversal_layout(&layout, &limits));
  for (size_t prefix = 0; prefix <= 11; ++prefix) {
    CAPTURE(prefix);
    Role role;
    auto value = multiply(add(left, right, stream), left, stream);
    Operation operation;
    operation.append(value);
    std::vector<Block> blocks;
    blocks.reserve(role.records->capacity() / alignof(std::max_align_t) + 8);
    ArenaBlocks occupied(role.records.get(), blocks);
    for (size_t i = 0; i != 11; ++i) {
      occupied.add(layout.request_bytes[i], layout.request_alignments[i]);
      occupied.add(1, 1);
    }
    occupied.fill();
    for (size_t i = 0; i != prefix; ++i) occupied.release(2 * i);
    const auto held = role.records->occupied_bytes();
    scheduler::CpuStreamToken worker;
    REQUIRE(scheduler::prepared_cpu_stream(stream, worker) == submission::NativeControlFailure::none);
    const auto accepted = scheduler::cpu_stream_progress(worker).accepted;
    const auto status = submit(operation, stream, limits);
    if (prefix < 11) {
      CHECK(status == static_cast<unsigned>(ScopedEvaluation::failed));
      REQUIRE(role.error.get()->borrow());
      CHECK(role.error.get()->borrow()->exception_type == &typeid(submission::RecordQuotaError));
      CHECK(scheduler::cpu_stream_progress(worker).accepted == accepted);
      // Constructor ownership never escaped into the registry.
      CHECK(role.records->occupied_bytes() == held);
      CHECK(submit(operation, stream, limits) != 0); // once-only attempted event
    } else {
      REQUIRE(status == 0);
      complete(role, operation, value);
      CHECK(value.data<float>()[0] == 18.f);
      CHECK(value.data<float>()[1] == 42.f);
      CHECK(value.data<float>()[2] == 90.f);
    }
    settle(role);
    CHECK(role.records->occupied_bytes() == held);
  }
}

TEST_CASE("prepared Eval traversal undercounts refuse actual insertion without dispatch or fallback"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace eval_traversal_tests;
  const auto first = new_stream(Device::cpu), second = new_stream(Device::cpu);
  prepare(first, second);
  array left({13, 17, 19}), right({5, 7, 8});
  // Independently stress DFS/cache, tape, edge, stream and sibling output bounds.
  for (const Limits limits : {Limits{1, 3, 3, 5, 4, 2, 8},
      {1, 6, 2, 5, 4, 2, 8}, {1, 6, 3, 4, 4, 2, 8},
      {1, 6, 3, 5, 4, 1, 8}, {1, 6, 3, 5, 3, 2, 8}}) {
    CAPTURE(limits.array_nodes); CAPTURE(limits.tape_entries);
    CAPTURE(limits.input_edges); CAPTURE(limits.stream_count); CAPTURE(limits.output_slots);
    mlx_operation_eval_traversal_layout layout{};
    REQUIRE(mlx_operation_event_eval_traversal_layout(&layout, &limits));
    Role role;
    auto both = divmod(left, right, first);
    auto value = add(both[0], both[1], second);
    Operation operation;
    operation.append(value);
    scheduler::CpuStreamToken a, b;
    REQUIRE(scheduler::prepared_cpu_stream(first, a) == submission::NativeControlFailure::none);
    REQUIRE(scheduler::prepared_cpu_stream(second, b) == submission::NativeControlFailure::none);
    const auto aa = scheduler::cpu_stream_progress(a).accepted;
    const auto bb = scheduler::cpu_stream_progress(b).accepted;
    CHECK(submit(operation, second, limits) == static_cast<unsigned>(ScopedEvaluation::failed));
    REQUIRE(role.error.get()->borrow());
    CHECK(role.error.get()->borrow()->exception_type == &typeid(submission::RecordQuotaError));
    CHECK(scheduler::cpu_stream_progress(a).accepted == aa);
    CHECK(scheduler::cpu_stream_progress(b).accepted == bb);
    CHECK(role.records->occupied_bytes() > 0); // entered prefix survives refusal
    settle(role);
    CHECK(role.records->occupied_bytes() == 0);
  }
}

TEST_CASE("prepared Eval CPU pointwise and cross stream siblings preserve values and retirement"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  eval_traversal_tests::round_trip(Device::cpu, false);
  eval_traversal_tests::round_trip(Device::cpu, true);
}
#ifdef MLX_C_PATCH_TEST_METAL
TEST_CASE("prepared Eval Metal pointwise and cross stream siblings preserve values and retirement"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  eval_traversal_tests::round_trip(Device::gpu, false);
  eval_traversal_tests::round_trip(Device::gpu, true);
}
#endif

TEST_CASE("prepared Eval completed leaf proof refuses lazy work and observes exact prior frontier"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace eval_traversal_tests;
  const auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  array input({2.f, 3.f, 5.f});
  Role role;
  Operation operation;
  CHECK(mlx_operation_event_validate_traversal_leaf(operation.observer, mlx_array{&input}) == 0);
  {
    detail::InTracing tracing(true);
    CHECK(mlx_operation_event_validate_traversal_context(operation.observer) != 0);
    CHECK(mlx_operation_event_validate_traversal_leaf(operation.observer, mlx_array{&input}) != 0);
  }
  CHECK(mlx_operation_event_validate_traversal_context(operation.observer) == 0);
  auto value = multiply(input, input, stream);
  scheduler::CpuStreamToken worker;
  REQUIRE(scheduler::prepared_cpu_stream(stream, worker) == submission::NativeControlFailure::none);
  const auto accepted = scheduler::cpu_stream_progress(worker).accepted;
  CHECK(mlx_operation_event_validate_traversal_leaf(operation.observer, mlx_array{&value}) != 0);
  CHECK(value.status() == array::Status::unscheduled);
  CHECK(value.has_primitive());
  CHECK(scheduler::cpu_stream_progress(worker).accepted == accepted);
  operation.append(value);
  REQUIRE(submit(operation, stream, Limits{1, 3, 2, 3, 2, 1, 8}) == 0);
  REQUIRE(mlx_operation_event_wait(operation.value) == 0);
  CHECK(mlx_operation_event_validate_traversal_leaf(operation.observer, mlx_array{&value}) == 0);
  CHECK(value.status() == array::Status::available);
  CHECK_FALSE(value.has_primitive());
  CHECK_FALSE(value.event().valid());
  settle(role);
  CHECK(value.data<float>()[2] == 25.f);
  // A real foreign current scope cannot use the ancestor observer as authority.
  {
    std::unique_ptr<submission::Scope, wait_record_facts::ReleaseScope> foreign{
        new submission::Scope()}; // inherit the exact parent arenas
    CHECK(submission::current_scope() == foreign.get());
    CHECK(mlx_operation_event_validate_traversal_leaf(operation.observer, mlx_array{&value}) != 0);
  }
}

TEST_CASE("prepared Eval selected empty frontier still executes its one Synchronizer"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace eval_traversal_tests;
  const auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  Role role;
  Operation operation;
  scheduler::CpuStreamToken worker;
  REQUIRE(scheduler::prepared_cpu_stream(stream, worker) == submission::NativeControlFailure::none);
  const auto accepted = scheduler::cpu_stream_progress(worker).accepted;
  REQUIRE(submit(operation, stream, Limits{0, 1, 1, 0, 1, 1, 8}) == 0);
  REQUIRE(mlx_operation_event_wait(operation.value) == 0);
  CHECK(scheduler::cpu_stream_progress(worker).accepted > accepted);
  settle(role);
  CHECK(role.records->occupied_bytes() == 0);
}

TEST_CASE("prepared Eval fixed capture refusal retains the accepted task and real Data prefix"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace eval_traversal_tests;
  const auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  auto state = std::make_shared<PrefixState>();
  Role role;
  ReleasePrefix release{state}; // release before role teardown on assertion unwind
  std::optional<array> output;
  output.emplace(Shape{1}, float32,
      submission::make_graph_primitive<FixedCapturePrefix>(stream, state), ArrayVector{});
  Operation operation;
  operation.append(*output);
  scheduler::CpuStreamToken worker;
  REQUIRE(scheduler::prepared_cpu_stream(stream, worker) == submission::NativeControlFailure::none);
  const auto accepted = scheduler::cpu_stream_progress(worker).accepted;
  CHECK(submit(operation, stream, Limits{1, 2, 2, 1, 2, 1, 8}) ==
      static_cast<unsigned>(ScopedEvaluation::failed));
  REQUIRE(role.error.get()->borrow());
  CHECK(role.error.get()->borrow()->exception_type == &typeid(submission::RecordQuotaError));
  CHECK(scheduler::cpu_stream_progress(worker).accepted > accepted);
  operation.reset();
  output.reset();
  CHECK_FALSE(state->temporary.expired());
  CHECK_FALSE(state->output.expired());
  CHECK(role.scope->query_records().pending > 0);
  state->release.store(true, std::memory_order_release);
  settle(role);
  CHECK(state->completed.load(std::memory_order_acquire));
  CHECK(state->temporary.expired());
  CHECK(state->output.expired());
  CHECK(role.records->occupied_bytes() == 0);
  state->temporary.reset();
  state->output.reset();
}

#if !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION != 210106 || __cplusplus != 202002L
TEST_CASE("prepared Eval unqualified traversal query leaves the actual output unchanged") {
  using wait_record_facts::check_unknown_unchanged;
  const auto limits = eval_traversal_tests::pointwise();
  check_unknown_unchanged<mlx_operation_eval_traversal_layout>([&](auto& out) {
    return mlx_operation_event_eval_traversal_layout(&out, &limits);
  });
}
#endif

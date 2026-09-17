#ifdef MLX_C_PATCH_TEST_METAL
#include "mlx/backend/metal/binary.h"
#endif
// Included after the exact Eval/Graph fixtures. All positive preparation and
// execution below uses the actual private fixed Eval type through the C entry.
// Existing cold stream/kernel setup remains an explicit component prerequisite.
namespace gpu_eval_prologue_tests {
using namespace eval_synchronizer_graph_tests;
#if defined(MLX_C_PATCH_TEST_METAL)
constexpr bool qualified = wait_record_facts::layout_qualified;
#else
constexpr bool qualified = false;
#endif
struct PretenderRecord : submission::Record {
  mutable size_t queries{0};
  explicit PretenderRecord(Allocation allocation) : Record(allocation) {}
  bool scoped_observation_supported() const noexcept override { return true; }
  // These are deliberately NOT overrides. Exact native type recognition must
  // never call extension methods or accept this record's declaration.
  bool fixed_gpu_eval_prologue() const noexcept { ++queries; return true; }
  bool gpu_eval_current_array(const array&) const noexcept { ++queries; return true; }
};
}

TEST_CASE("fixed GPU prologue query is qualified or leaves the full output unchanged") {
  using namespace gpu_eval_prologue_tests;
  static_assert(sizeof(mlx_gpu_eval_prologue_layout) == 26 * sizeof(size_t));
  static_assert(std::is_same_v<decltype(&mlx_operation_event_gpu_eval_prologue_layout),
      bool (*)(mlx_gpu_eval_prologue_layout*, size_t, size_t)>);
  static_assert(std::is_same_v<decltype(&mlx_operation_event_gpu_eval_prologue_layout_with_tracing),
      bool (*)(mlx_gpu_eval_prologue_layout*, size_t, size_t, bool)>);
  if (std::getenv("EREDU_REQUIRE_GPU_EVAL_PROLOGUE_QUALIFICATION")) REQUIRE(qualified);
  mlx_gpu_eval_prologue_layout value;
  std::memset(&value, 0xa5, sizeof(value));
  const auto before = value;
  if (!qualified) {
    CHECK_FALSE(mlx_operation_event_gpu_eval_prologue_layout(&value, 2, 0));
    CHECK(std::memcmp(&value, &before, sizeof(value)) == 0);
    CHECK_FALSE(mlx_operation_event_gpu_eval_prologue_layout_with_tracing(&value, 2, 0, true));
    CHECK(std::memcmp(&value, &before, sizeof(value)) == 0);
    return;
  }
  REQUIRE(mlx_operation_event_gpu_eval_prologue_layout(&value, 2, 0));
  CHECK(value.blocks == 4);
  mlx_pointwise_graph_layout host{};
  mlx_operation_eval_record_layout eval{};
  REQUIRE(mlx_operation_event_pointwise_graph_layout(&host, 0, 0));
  REQUIRE(mlx_operation_event_eval_record_layout(&eval, 1, 1, 1));
  CHECK(value.header_bytes == host.header_bytes);
  CHECK(value.header_bytes == eval.host_graph_header_bytes);
  CHECK(eval.object_bytes >= sizeof(submission::Record));
  mlx_operation_eval_traversal_layout traversal{};
  const mlx_operation_eval_traversal_limits limits{0, 1, 1, 0, 1, 1, 8};
  REQUIRE(mlx_operation_event_eval_traversal_layout(&traversal, &limits));
  CHECK(traversal.request_bytes[0] >= sizeof(submission::Record));
  CHECK(value.request_bytes[2] == 2 * sizeof(std::shared_ptr<array::Data>));
  CHECK(value.requested_bytes == value.header_bytes + value.slots_bytes +
      value.request_bytes[0] + value.request_bytes[1] + value.request_bytes[2] + value.request_bytes[3]);
  CHECK(value.request_bytes[3] == sizeof(array));
  CHECK(value.request_bytes[4] == 0);
  REQUIRE(mlx_operation_event_gpu_eval_prologue_layout_with_tracing(&value, 2, 0, true));
  CHECK(value.blocks == 5);
  CHECK(value.tracer_inputs == 2);
  CHECK(value.request_bytes[4] == 2 * sizeof(array));
  const auto saved = value;
  CHECK_FALSE(mlx_operation_event_gpu_eval_prologue_layout(&value, SIZE_MAX, 1));
  CHECK(std::memcmp(&value, &saved, sizeof(value)) == 0);
  REQUIRE(mlx_operation_event_gpu_eval_prologue_layout(&value, 0, 0));
  CHECK(value.blocks == 3);
  CHECK(value.request_bytes[2] == 0);
  CHECK(value.request_bytes[3] == sizeof(array));
  CHECK(value.request_bytes[4] == 0);
}

TEST_CASE("custom and ordinary Records cannot certify a fixed GPU prologue"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace gpu_eval_prologue_tests;
  array value({2.f, 3.f});
  Role role;
  auto owner = submission::Record::create<PretenderRecord>();
  owner->enter(); auto* record = owner.release(); FinishRecord finish{record};
  {
    submission::RecordDispatchGuard dispatch(*record);
    record->begin_primitive();
    const auto occupied = role.graph->occupied_bytes();
    CHECK_FALSE(submission::fixed_gpu_eval_record(*record));
    CHECK_FALSE(submission::fixed_gpu_eval_record(*record, &value));
    CHECK_FALSE(submission::fixed_gpu_eval_prologue_required());
    CHECK_THROWS_AS(submission::GpuEvalPrologueGuard{value}, submission::NativeControlError);
    CHECK(record->queries == 0);
    CHECK(role.graph->occupied_bytes() == occupied);
    record->end_primitive();
  }
  finish.success(); settle(role);
}

TEST_CASE("shared output filling reuses exact storage preserves sibling order and refuses intact"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace gpu_eval_prologue_tests;
  const auto stream = new_stream(Device::cpu); prepare(stream, stream);
  array left({7, 11}), right({3, 4});
  Role role;
  auto both = divmod(left, right, stream);
  REQUIRE(both.size() == 2);
  auto ordinary = both[1].outputs();
  REQUIRE(ordinary.size() == 2);
  ArrayVector supplied; supplied.reserve(2);
  ArrayVector short_storage; short_storage.reserve(1);
  auto* pointer = supplied.data();
  GraphHoles holes(role.graph.get()); holes.fill();
  const auto occupied = role.graph->occupied_bytes();
  both[1].fill_outputs(supplied);
  CHECK(supplied.data() == pointer);
  REQUIRE(supplied.size() == 2);
  for (size_t i = 0; i < 2; ++i) CHECK(supplied[i].id() == ordinary[i].id());
  CHECK_THROWS_AS(both[1].fill_outputs(supplied), submission::GraphQuotaError);
  CHECK_THROWS_AS(both[1].fill_outputs(short_storage), submission::GraphQuotaError);
  CHECK(short_storage.empty());
  CHECK(supplied.data() == pointer);
  CHECK(role.graph->occupied_bytes() == occupied);
}

#ifdef MLX_C_PATCH_TEST_METAL
TEST_CASE("actual fixed GPU prologue refuses retention and invocation vector prefixes before submission"
    * doctest::skip(!gpu_eval_prologue_tests::qualified)) {
  using namespace gpu_eval_prologue_tests;
  auto extent = [](size_t bytes, size_t alignment) {
    size_t value = 0;
    REQUIRE(submission::GraphQuota::minimum_allocation_extent(bytes, alignment, value));
    return value;
  };
  for (bool traced : {false, true}) {
    CAPTURE(traced);
    submission::EvalGraphLayout sync;
    submission::GpuEvalPrologueLayout prologue;
    REQUIRE(submission::eval_graph_layout(sync));
    REQUIRE(submission::gpu_eval_prologue_layout(2, 0, traced, prologue));
    REQUIRE(sync.blocks == 4);
    REQUIRE(prologue.blocks == (traced ? 5 : 4));
    REQUIRE(sync.header_bytes == prologue.header_bytes);
    REQUIRE(sync.header_alignment == prologue.header_alignment);
    REQUIRE(sync.slots_alignment == prologue.slots_alignment);
    REQUIRE(sync.slots_bytes / sync.blocks == prologue.slots_bytes / prologue.blocks);
    REQUIRE(prologue.request_bytes[2] == 2 * sizeof(std::shared_ptr<array::Data>));
    REQUIRE(prologue.request_bytes[3] == sizeof(array));
    REQUIRE(prologue.request_bytes[4] == (traced ? 2 * sizeof(array) : 0));
    const auto slot_bytes = std::max(sync.slots_bytes, prologue.slots_bytes);
    const auto slot_extent = extent(slot_bytes, sync.slots_alignment);
    const auto minimum = extent(1, 1);
    // Both real slot requests absorb this same isolated block. Neither can
    // leave enough bytes for even the smallest usable Graph allocation.
    REQUIRE(slot_extent - extent(sync.slots_bytes, sync.slots_alignment) < minimum);
    REQUIRE(slot_extent - extent(prologue.slots_bytes, prologue.slots_alignment) < minimum);
    // Four nontraced refusals cover retention/output. Only the new final
    // tracer destination is repeated under a real RetainGraph invocation.
    for (size_t prefix = traced ? 4 : 0; prefix < prologue.blocks; ++prefix) {
      CAPTURE(prefix);
      const auto stream = new_stream(Device::gpu); prepare(stream, stream);
      auto token = gpu::prepare_submission_stream(stream);
      array left({2.f, 3.f}), right({7.f, 11.f});
      Role role;
      Operation operation;
      std::optional<array> sum, value;
      auto limits = eval_traversal_tests::pointwise();
      REQUIRE(left.status() == array::Status::available);
      REQUIRE(right.status() == array::Status::available);
      REQUIRE_FALSE(left.has_primitive());
      REQUIRE_FALSE(right.has_primitive());
      if (!traced) {
        sum.emplace(add(left, right, stream));
        value.emplace(multiply(*sum, left, stream));
        // Actual first reverse-tape primitive: Add with detached leaf inputs.
        REQUIRE(typeid(sum->primitive()) == typeid(Add));
        REQUIRE(typeid(value->primitive()) == typeid(Multiply));
        REQUIRE(sum->primitive().stream() == stream);
        REQUIRE(value->primitive().stream() == stream);
        REQUIRE(sum->inputs().size() == 2);
        REQUIRE(value->inputs().size() == 2);
        REQUIRE(sum->inputs()[0].id() == left.id());
        REQUIRE(sum->inputs()[1].id() == right.id());
        REQUIRE(value->inputs()[0].id() == sum->id());
        REQUIRE(value->inputs()[1].id() == left.id());
        REQUIRE(sum->siblings().empty());
        REQUIRE(value->siblings().empty());
        REQUIRE_FALSE(sum->is_tracer());
        operation.append(*value);
      } else {
        // Available roots require only the genuine Synchronizer (two inputs).
        // RetainGraph does not bypass the existing async unscheduled-tracer
        // rejection: neither root is unscheduled or has a primitive.
        operation.append(left); operation.append(right);
        limits = {2, 3, 1, 2, 1, 1, 8};
      }
      std::optional<detail::RetainGraph> tracing;
      if (traced) { tracing.emplace(); REQUIRE(left.is_tracer()); }
      const auto accepted = gpu::submission_progress(token).accepted;
      const auto left_data = left.data_shared_ptr().get();
      const auto right_data = right.data_shared_ptr().get();
      {
        GraphHoles holes(role.graph.get());
        auto add_hole = [&](size_t bytes, size_t alignment) {
          const auto before = role.graph->occupied_bytes();
          holes.add(bytes, alignment);
          REQUIRE(role.graph->occupied_bytes() - before == extent(bytes, alignment));
          holes.add(1, 1);
        };
        add_hole(sync.header_bytes, sync.header_alignment);
        add_hole(slot_bytes, sync.slots_alignment);
        for (size_t i = 0; i < 4; ++i)
          add_hole(sync.request_bytes[i], sync.reserved_alignment);
        for (size_t i = 0; i < prefix; ++i)
          add_hole(prologue.request_bytes[i], prologue.reserved_alignment);
        holes.fill();
        for (size_t i = 0; i < 6 + prefix; ++i) holes.release(2 * i);
        const auto occupied = role.graph->occupied_bytes();
        REQUIRE(eval_traversal_tests::submit(operation, stream, limits) ==
            static_cast<unsigned>(ScopedEvaluation::failed));
        REQUIRE(role.error.get()->borrow());
        CHECK(role.error.get()->borrow()->exception_type == &typeid(submission::GraphQuotaError));
        CHECK(gpu::submission_progress(token).accepted == accepted);
        CHECK(left.data_shared_ptr().get() == left_data);
        CHECK(right.data_shared_ptr().get() == right_data);
        CHECK(left.data<float>()[0] == 2.f);
        CHECK(right.data<float>()[1] == 11.f);
        if (value) {
          CHECK(sum->status() == array::Status::unscheduled);
          CHECK(value->status() == array::Status::unscheduled);
        }
        size_t sync_payload = 0;
        for (size_t i = 0; i < 4; ++i)
          sync_payload += extent(sync.request_bytes[i], sync.reserved_alignment);
        CHECK(role.graph->occupied_bytes() == occupied + sync_payload);
        CHECK(role.records->occupied_bytes() > 0);
      }
      settle(role); operation.reset();
      CHECK(role.records->occupied_bytes() == 0);
    }
  }
}

TEST_CASE("fixed GPU prologue completes actual Add Multiply and retains final alias"
    * doctest::skip(!gpu_eval_prologue_tests::qualified)) {
  // Genuine private PreparedEvalSubmissionRecord; controls precede each actual
  // primitive and retire through the real completion path, not a probe class.
  eval_synchronizer_graph_tests::numerical(Device::gpu);
}
TEST_CASE("fixed GPU prologue zero roots has no Data slot vector and completes"
    * doctest::skip(!gpu_eval_prologue_tests::qualified)) {
  using namespace gpu_eval_prologue_tests;
  const auto stream = new_stream(Device::gpu); prepare(stream, stream);
  Role role; Operation operation;
  const mlx_operation_eval_traversal_limits limits{0, 1, 1, 0, 1, 1, 8};
  REQUIRE(eval_traversal_tests::submit(operation, stream, limits) == 0);
  REQUIRE(mlx_operation_event_wait(operation.value) == 0);
  settle(role); operation.reset();
  CHECK(role.records->occupied_bytes() == 0);
  CHECK(role.graph->occupied_bytes() == 0);
}
TEST_CASE("ordinary GPU Add Multiply remains independent of fixed prologue ownership") {
  const auto stream = new_stream(Device::gpu);
  auto left = array({2.f, 3.f, 5.f}), right = array({7.f, 11.f, 13.f});
  auto value = multiply(add(left, right, stream), left, stream);
  CHECK_FALSE(submission::fixed_gpu_eval_prologue_required());
  eval(value);
  CHECK(value.data<float>()[0] == 18.f);
  CHECK(value.data<float>()[1] == 42.f);
  CHECK(value.data<float>()[2] == 90.f);
}

TEST_CASE("accepted CPU worker keeps independent Graph storage during fixed GPU Eval"
    * doctest::skip(!gpu_eval_prologue_tests::qualified)) {
  using namespace gpu_eval_prologue_tests;
  struct State {
    std::atomic<bool> started{false}, stop{false}, failed{false};
    std::atomic<size_t> attempts{0};
  };
  struct Release { std::shared_ptr<State> state; ~Release() { state->stop.store(true); } };
  const auto cpu_stream = new_stream(Device::cpu), gpu_stream = new_stream(Device::gpu);
  prepare(cpu_stream, gpu_stream);
  auto state = std::make_shared<State>();
  array input({2.f, 3.f});
  Role role;
  Release release{state};
  auto owned = submission::Record::create<HostRecord>();
  owned->enter(); auto* worker_record = owned.release(); FinishRecord finish{worker_record};
  worker_record->reserve_streams(1); worker_record->prepare_stream(cpu_stream);
  {
    submission::RecordDispatchGuard dispatch(*worker_record);
    scheduler::enqueue(cpu_stream, [state, graph = submission::GraphQuotaRef(role.graph.get())] {
      state->started.store(true, std::memory_order_release);
      while (!state->stop.load(std::memory_order_acquire)) {
        try {
          submission::GraphAllocator<unsigned char> allocator(graph.get());
          auto* block = allocator.allocate(17);
          block[16] = 37;
          allocator.deallocate(block, 17);
        } catch (...) { state->failed.store(true); }
        state->attempts.fetch_add(1);
        std::this_thread::yield();
      }
    });
  }
  finish.success();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!state->started.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  REQUIRE(state->started.load());
  auto value = input;
  constexpr size_t steps = 128;
  for (size_t i = 0; i < steps; ++i) value = add(value, input, gpu_stream);
  Operation operation; operation.append(value);
  // Actual chain: one detached leaf, 128 binary outputs and Synchronizer.
  const mlx_operation_eval_traversal_limits limits{1, steps + 2, steps + 1,
      2 * steps + 1, steps + 1, 1, 8};
  const auto result = eval_traversal_tests::submit(operation, gpu_stream, limits);
  state->stop.store(true, std::memory_order_release);
  REQUIRE(result == 0);
  eval_traversal_tests::complete(role, operation, value);
  CHECK(state->attempts.load() > 0);
  CHECK_FALSE(state->failed.load());
  CHECK(value.data<float>()[0] == 258.f);
  CHECK(value.data<float>()[1] == 387.f);
}
TEST_CASE("binary zero output borrows its range with no remaining Graph allocation"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace gpu_eval_prologue_tests;
  const auto stream = new_stream(Device::gpu); prepare(stream, stream);
  std::vector<float> empty;
  array left(empty.begin(), Shape{0}, float32), right(empty.begin(), Shape{0}, float32);
  Role role;
  auto out = add(left, right, stream);
  ArrayVector inputs{left, right};
  GraphHoles holes(role.graph.get()); holes.fill();
  const auto occupied = role.graph->occupied_bytes();
  CHECK_NOTHROW(binary_op_gpu_inplace(inputs, out, "Add", stream));
  CHECK(role.graph->occupied_bytes() == occupied);
  CHECK(out.status() == array::Status::unscheduled);
}

TEST_CASE("fixed invocation empty destinations preserve unique intermediate donation"
    * doctest::skip(!gpu_eval_prologue_tests::qualified)) {
  using namespace gpu_eval_prologue_tests;
  for (bool fixed : {false, true}) {
    CAPTURE(fixed);
    const auto stream = new_stream(Device::gpu); prepare(stream, stream);
    array left({2.f, 3.f, 5.f}), scale({2.f, 3.f, 5.f});
    std::optional<array> right(std::in_place, std::initializer_list<float>{7.f, 11.f, 13.f});
    const auto* expected = right->data_shared_ptr().get(); // borrowed identity only
    std::optional<Role> role;
    if (fixed) role.emplace();
    // No external sum Array: after Add detaches, Multiply's first input is the
    // sole descriptor owner. The retained right input also dies before Eval.
    auto value = multiply(add(left, *right, stream), scale, stream);
    right.reset();
    REQUIRE(value.inputs()[0].inputs()[1].is_donatable());
    if (fixed) {
      Operation operation; operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1, 6, 3, 5, 3, 1, 8};
      REQUIRE(eval_traversal_tests::submit(operation, stream, limits) == 0);
      eval_traversal_tests::complete(*role, operation, value);
      operation.reset();
    } else eval(value);
    CHECK(value.data_shared_ptr().get() == expected);
    CHECK(value.data<float>()[0] == 18.f);
    CHECK(value.data<float>()[1] == 42.f);
    CHECK(value.data<float>()[2] == 90.f);
  }
}

TEST_CASE("fixed RetainGraph Synchronizer fills tracer destinations and retires actual backing"
    * doctest::skip(!gpu_eval_prologue_tests::qualified)) {
  using namespace gpu_eval_prologue_tests;
  const auto stream = new_stream(Device::gpu); prepare(stream, stream);
  array left({2.f, 3.f}), right({7.f, 11.f});
  const auto* left_data = left.data_shared_ptr().get();
  const auto* right_data = right.data_shared_ptr().get();
  Role role;
  {
    Operation operation; operation.append(left); operation.append(right);
    detail::RetainGraph tracing;
    REQUIRE(left.is_tracer());
    const mlx_operation_eval_traversal_limits limits{2, 3, 1, 2, 1, 1, 8};
    REQUIRE(eval_traversal_tests::submit(operation, stream, limits) == 0);
    REQUIRE(mlx_operation_event_wait(operation.value) == 0);
    settle(role); operation.reset();
  }
  CHECK(role.records->occupied_bytes() == 0);
  CHECK(role.graph->occupied_bytes() == 0);
  CHECK(left.data_shared_ptr().get() == left_data);
  CHECK(right.data_shared_ptr().get() == right_data);
  CHECK(left.is_donatable());
  CHECK(right.is_donatable());
  CHECK(left.data<float>()[1] == 3.f);
  CHECK(right.data<float>()[1] == 11.f);
}

#endif

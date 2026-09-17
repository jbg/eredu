#include "mlx/backend/cpu/retention.h"
// Included after Graph construction fixtures; same genuine Scope/Record owners.
namespace eval_synchronizer_graph_tests {
using namespace eval_record_facts;
using pointwise_graph_tests::GraphHoles;
struct Bank {
  submission::GraphConstruction* value{nullptr};
  ~Bank() { reset(); }
  void reset() { if (value) value->destroy(); value = nullptr; }
};
struct HostRecord : submission::Record {
  explicit HostRecord(Allocation allocation) : Record(allocation) {}
  bool scoped_observation_supported() const noexcept override { return true; }
};
struct FinishRecord {
  submission::Record* record;
  ~FinishRecord() { if (record) record->finish(true); }
  void success() { record->finish(false); record = nullptr; }
};
void numerical(Device device) {
  const auto stream = new_stream(device);
  prepare(stream, stream);
  array left({2.f, 3.f, 5.f}), right({7.f, 11.f, 13.f});
  Role role;
  std::optional<array> alias;
  {
    auto value = multiply(add(left, right, stream), left, stream);
    alias.emplace(value);
    Operation operation;
    operation.append(value);
    REQUIRE(eval_traversal_tests::submit(operation, stream, eval_traversal_tests::pointwise()) == 0);
    eval_traversal_tests::complete(role, operation, value);
    CHECK(value.data<float>()[0] == 18.f);
    CHECK(value.data<float>()[1] == 42.f);
    CHECK(value.data<float>()[2] == 90.f);
    CHECK(alias->data_shared_ptr() == value.data_shared_ptr());
    operation.reset();
    CHECK(role.records->occupied_bytes() == 0);
  }
  CHECK(role.graph->occupied_bytes() > 0); // actual alias owns the final Data/descriptor
  alias.reset();
  CHECK(role.graph->occupied_bytes() == 0);
}
}

TEST_CASE("fixed Eval host Graph recipe includes four owners header slots and platform population"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace eval_synchronizer_graph_tests;
  mlx_operation_eval_record_layout c{};
  CompletionEvalRecordLayout native{};
  REQUIRE(completion_eval_record_layout(3, 1, 3, native));
  REQUIRE(mlx_operation_event_eval_record_layout(&c, 3, 1, 3));
  CHECK(c.host_graph_blocks == 4);
  CHECK(native.host_graph.blocks == 4);
  CHECK(c.host_graph_header_bytes > 0);
  CHECK(c.host_graph_slots_bytes > 0);
  size_t sum = c.host_graph_header_bytes + c.host_graph_slots_bytes;
  for (size_t i = 0; i < 4; ++i) {
    CHECK(c.host_graph_request_bytes[i] == native.host_graph.request_bytes[i]);
    CHECK(c.host_graph_request_alignments[i] == native.host_graph.request_alignments[i]);
    CHECK(c.host_graph_request_bytes[i] > 0);
    CHECK(native.host_graph.request_counts[i] == 1);
    sum += c.host_graph_request_bytes[i];
  }
  CHECK(sum == c.host_graph_requested_bytes);
  CHECK(c.host_graph_allocation_extents > sum);
  EventControlLayout event{};
  REQUIRE(event_control_layout(event));
  CHECK(c.host_graph_request_bytes[3] == event.shared_bytes);
  CHECK(c.host_graph_event_controls == event.construction_controls);
  CHECK(c.host_graph_platform_events == event.platform_events);
  CHECK(c.named_control_bytes > c.host_graph_event_controls);
  CHECK(c.record_allocations == 6); // Graph requests are not duplicated Record charges
  const auto saved = c;
  CHECK_FALSE(mlx_operation_event_eval_record_layout(&c, 0, 1, 1));
  CHECK(std::memcmp(&c, &saved, sizeof(c)) == 0);
}

TEST_CASE("fixed Eval host Graph reservation refuses each real prefix before root mutation or dispatch"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace eval_synchronizer_graph_tests;
  const auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  array left({2.f, 3.f}), right({7.f, 11.f});
  submission::EvalGraphLayout layout;
  REQUIRE(submission::eval_graph_layout(layout));
  for (size_t prefix = 0; prefix < 6; ++prefix) {
    CAPTURE(prefix);
    Role role;
    auto value = multiply(add(left, right, stream), left, stream);
    const auto id = value.id();
    const auto primitive = value.primitive_ptr();
    Operation operation;
    const auto before_roots = role.graph->occupied_bytes();
    operation.append(value);
    const auto with_roots = role.graph->occupied_bytes();
    REQUIRE(with_roots > before_roots);
    const auto root_vector_extent = with_roots - before_roots;
    GraphHoles holes(role.graph.get());
    holes.add(layout.header_bytes, layout.header_alignment); holes.add(1, 1);
    holes.add(layout.slots_bytes, layout.slots_alignment); holes.add(1, 1);
    for (size_t i = 0; i < 4; ++i) {
      holes.add(layout.request_bytes[i], layout.reserved_alignment); holes.add(1, 1);
    }
    holes.fill();
    for (size_t i = 0; i < prefix; ++i) holes.release(2*i);
    const auto occupied = role.graph->occupied_bytes();
    scheduler::CpuStreamToken worker;
    REQUIRE(scheduler::prepared_cpu_stream(stream, worker) == submission::NativeControlFailure::none);
    const auto accepted = scheduler::cpu_stream_progress(worker).accepted;
    CHECK(eval_traversal_tests::submit(operation, stream, eval_traversal_tests::pointwise()) ==
        static_cast<unsigned>(ScopedEvaluation::failed));
    REQUIRE(role.error.get()->borrow());
    CHECK(role.error.get()->borrow()->exception_type == &typeid(submission::GraphQuotaError));
    CHECK(scheduler::cpu_stream_progress(worker).accepted == accepted);
    // The submitted ArrayVector moves into Eval's by-value outputs and retires
    // on refusal before Synchronizer adoption. Every reserved prefix also retires.
    REQUIRE(occupied >= root_vector_extent);
    CHECK(role.graph->occupied_bytes() == occupied - root_vector_extent);
    CHECK(value.id() == id);
    CHECK(value.primitive_ptr() == primitive);
    CHECK(value.status() == array::Status::unscheduled);
    CHECK_FALSE(value.event().valid());
    CHECK(role.records->occupied_bytes() > 0); // actual entered failure retained
    settle(role);
    CHECK(role.records->occupied_bytes() == 0);
  }
}

TEST_CASE("fixed Eval host bank authenticates exact entered dispatch without nested fallback"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace eval_synchronizer_graph_tests;
  Role role;
  auto a_owner = submission::Record::create<HostRecord>();
  a_owner->enter(); auto* a = a_owner.release(); FinishRecord finish_a{a};
  auto b_owner = submission::Record::create<HostRecord>();
  b_owner->enter(); auto* b = b_owner.release(); FinishRecord finish_b{b};
  const auto baseline = role.graph->occupied_bytes();
  {
    submission::RecordDispatchGuard dispatch(*a);
    {
      auto child_failure = wait_record_facts::failure();
      std::unique_ptr<submission::Scope, wait_record_facts::ReleaseScope> child(new submission::Scope());
      REQUIRE(child->bind_failure(std::move(child_failure)));
      REQUIRE(child->enable_original_controls() == submission::NativeControlFailure::none);
      REQUIRE(child->valid_original_observer());
      REQUIRE(child->active_on_owner_thread());
      REQUIRE(child->graph_quota() == role.graph.get());
      Bank child_bank;
      CHECK(submission::GraphConstruction::create_eval(*role.graph, child->identity(), *a, child_bank.value)
          == submission::GraphFailure::foreign_parent);
      CHECK_FALSE(child_bank.value); // parent guard cannot authorize the child
    }
    Bank foreign;
    CHECK(submission::GraphConstruction::create_eval(*role.graph, role.scope->identity(), *b, foreign.value)
        == submission::GraphFailure::foreign_parent);
    CHECK_FALSE(foreign.value);
    Bank bank;
    REQUIRE(submission::GraphConstruction::create_eval(*role.graph, role.scope->identity(), *a, bank.value)
        == static_cast<submission::GraphFailure>(0));
    CHECK(bank.value->remaining() == 4);
    const auto held = role.graph->occupied_bytes();
    Bank nested;
    CHECK(submission::GraphConstruction::create_eval(*role.graph, role.scope->identity(), *a, nested.value)
        == submission::GraphFailure::construction_busy);
    {
      submission::RecordDispatchGuard other(*b);
      CHECK_THROWS_AS(role.graph->allocate(1, 1), submission::GraphQuotaError);
      try { role.graph->allocate(1, 1); FAIL("foreign dispatch must refuse"); }
      catch (const submission::GraphQuotaError& error) {
        CHECK(error.cause() == submission::GraphFailure::foreign_parent);
      }
    }
    CHECK(bank.value->remaining() == 4);
    CHECK(role.graph->occupied_bytes() == held);
    finish_a.success(); // guard still borrows A, but A is now frozen
    void* unexpected = nullptr;
    bool refused = false;
    try { unexpected = role.graph->allocate(1, 1); }
    catch (const submission::GraphQuotaError& error) {
      CHECK(error.cause() == submission::GraphFailure::foreign_parent);
      refused = true;
    }
    if (unexpected) role.graph->deallocate(unexpected, 1, 1);
    CHECK(refused);
    CHECK(bank.value->remaining() == 4);
    CHECK(role.graph->occupied_bytes() == held);
  }
  CHECK(role.graph->occupied_bytes() == baseline);
  finish_b.success(); settle(role);
}

TEST_CASE("fixed Eval host Graph completes nonzero CPU work and retires after final alias"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  eval_synchronizer_graph_tests::numerical(Device::cpu);
}
#ifdef MLX_C_PATCH_TEST_METAL
TEST_CASE("fixed Eval host Graph completes nonzero Metal work and retires after final alias"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  eval_synchronizer_graph_tests::numerical(Device::gpu);
}
#endif

TEST_CASE("fixed Eval selected zero roots still constructs Synchronizer and retires it"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace eval_synchronizer_graph_tests;
  const auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  Role role;
  Operation operation;
  const mlx_operation_eval_traversal_limits limits{0, 1, 1, 0, 1, 1, 8};
  REQUIRE(eval_traversal_tests::submit(operation, stream, limits) == 0);
  REQUIRE(mlx_operation_event_wait(operation.value) == 0);
  settle(role);
  CHECK(role.records->occupied_bytes() == 0);
  operation.reset();
  CHECK(role.graph->occupied_bytes() == 0);
}

TEST_CASE("fixed Eval dispatch bank leaves previously accepted CPU worker allocations independent"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace eval_synchronizer_graph_tests;
  struct State { std::atomic<bool> release{false}, done{false}, success{false}; };
  struct Release {
    std::shared_ptr<State> state;
    ~Release() { state->release.store(true, std::memory_order_release); }
  };
  const auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  auto state = std::make_shared<State>();
  Role role;
  Release release{state};
  auto worker_owner = submission::Record::create<HostRecord>();
  worker_owner->enter(); auto* worker = worker_owner.release(); FinishRecord finish_worker{worker};
  worker->reserve_streams(1);
  worker->prepare_stream(stream);
  {
    submission::RecordDispatchGuard dispatch(*worker);
    scheduler::enqueue(stream, [state, graph = submission::GraphQuotaRef(role.graph.get())] {
      while (!state->release.load(std::memory_order_acquire)) std::this_thread::yield();
      try {
        submission::GraphAllocator<unsigned char> allocator(graph.get());
        auto* block = allocator.allocate(64);
        block[63] = 37;
        state->success.store(block[63] == 37, std::memory_order_relaxed);
        allocator.deallocate(block, 64);
      } catch (...) { state->success.store(false, std::memory_order_relaxed); }
      state->done.store(true, std::memory_order_release);
    });
  }
  finish_worker.success();
  auto host_owner = submission::Record::create<HostRecord>();
  host_owner->enter(); auto* host = host_owner.release(); FinishRecord finish_host{host};
  {
    submission::RecordDispatchGuard dispatch(*host);
    Bank bank;
    REQUIRE(submission::GraphConstruction::create_eval(*role.graph, role.scope->identity(), *host, bank.value)
        == static_cast<submission::GraphFailure>(0));
    REQUIRE(bank.value->remaining() == 4);
    state->release.store(true, std::memory_order_release);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!state->done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    REQUIRE(state->done.load(std::memory_order_acquire));
    CHECK(state->success.load(std::memory_order_relaxed));
    CHECK(bank.value->remaining() == 4); // worker used its retained Graph, not host destinations
  }
  finish_host.success();
  settle(role);
  CHECK(role.graph->occupied_bytes() == 0);
}

TEST_CASE("fixed CPU cleanup reserves exact node and Data prefixes before primitive dispatch"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace eval_synchronizer_graph_tests;
  static_assert(sizeof(mlx_cpu_eval_cleanup_layout) == 17 * sizeof(size_t));
  static_assert(std::is_same_v<decltype(&mlx_operation_event_cpu_eval_cleanup_layout),
      bool (*)(mlx_cpu_eval_cleanup_layout*, size_t, size_t)>);
  mlx_cpu_eval_cleanup_layout exported{};
  REQUIRE(mlx_operation_event_cpu_eval_cleanup_layout(&exported, 2, 0));
  REQUIRE(exported.blocks == 2);
  CHECK(exported.request_bytes[0] == sizeof(scheduler::TaskNode<cpu::detail::Cleanup>));
  CHECK(exported.request_bytes[1] == 2 * sizeof(cpu::detail::DataRetention::Value));
  CHECK(exported.requested_bytes == exported.header_bytes + exported.slots_bytes +
      exported.request_bytes[0] + exported.request_bytes[1]);
  const auto saved = exported;
  CHECK_FALSE(mlx_operation_event_cpu_eval_cleanup_layout(&exported, SIZE_MAX, 1));
  CHECK(std::memcmp(&exported, &saved, sizeof(exported)) == 0);
  REQUIRE(mlx_operation_event_cpu_eval_cleanup_layout(&exported, 0, 0));
  CHECK(exported.blocks == 1);
  CHECK(exported.request_bytes[1] == 0);

  auto extent = [](size_t bytes, size_t alignment) {
    size_t value = 0;
    REQUIRE(submission::GraphQuota::minimum_allocation_extent(bytes, alignment, value));
    return value;
  };
  submission::EvalGraphLayout sync;
  submission::CpuEvalCleanupLayout cleanup;
  REQUIRE(submission::eval_graph_layout(sync));
  REQUIRE(submission::cpu_eval_cleanup_layout(2, 0, cleanup));
  REQUIRE(sync.blocks == 4);
  REQUIRE(cleanup.blocks == 2);
  REQUIRE(sync.header_bytes == cleanup.header_bytes);
  REQUIRE(sync.header_alignment == cleanup.header_alignment);
  REQUIRE(sync.slots_alignment == cleanup.slots_alignment);
  const auto slot_bytes = std::max(sync.slots_bytes, cleanup.slots_bytes);
  const auto slot_extent = extent(slot_bytes, sync.slots_alignment);
  const auto minimum = extent(1, 1);
  REQUIRE(slot_extent - extent(sync.slots_bytes, sync.slots_alignment) < minimum);
  REQUIRE(slot_extent - extent(cleanup.slots_bytes, cleanup.slots_alignment) < minimum);
  for (size_t prefix : {size_t{0}, size_t{1}}) {
    CAPTURE(prefix);
    const auto stream = new_stream(Device::cpu); prepare(stream, stream);
    const auto token = scheduler::prepare_cpu_stream(stream);
    array left({2.f, 3.f}), right({7.f, 11.f});
    Role role;
    auto sum = add(left, right, stream);
    auto value = multiply(sum, left, stream);
    REQUIRE(typeid(sum.primitive()) == typeid(Add));
    REQUIRE(typeid(value.primitive()) == typeid(Multiply));
    REQUIRE(sum.primitive().stream() == stream);
    REQUIRE(sum.inputs().size() == 2);
    REQUIRE(sum.siblings().empty());
    REQUIRE(sum.inputs()[0].id() == left.id());
    REQUIRE(sum.inputs()[1].id() == right.id());
    REQUIRE(value.inputs()[0].id() == sum.id());
    REQUIRE(left.status() == array::Status::available);
    REQUIRE(right.status() == array::Status::available);
    REQUIRE_FALSE(left.has_primitive());
    REQUIRE_FALSE(right.has_primitive());
    Operation operation; operation.append(value);
    const auto accepted = scheduler::cpu_stream_progress(token).accepted;
    const auto* left_data = left.data_shared_ptr().get();
    const auto* right_data = right.data_shared_ptr().get();
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
      for (size_t i = 0; i < sync.blocks; ++i)
        add_hole(sync.request_bytes[i], sync.reserved_alignment);
      for (size_t i = 0; i < prefix; ++i)
        add_hole(cleanup.request_bytes[i], cleanup.reserved_alignment);
      holes.fill();
      for (size_t i = 0; i < 6 + prefix; ++i) holes.release(2 * i);
      const auto occupied = role.graph->occupied_bytes();
      REQUIRE(eval_traversal_tests::submit(operation, stream, eval_traversal_tests::pointwise()) ==
          static_cast<unsigned>(ScopedEvaluation::failed));
      REQUIRE(role.error.get()->borrow());
      CHECK(role.error.get()->borrow()->exception_type == &typeid(submission::GraphQuotaError));
      CHECK(scheduler::cpu_stream_progress(token).accepted == accepted);
      CHECK(sum.status() == array::Status::unscheduled);
      CHECK(value.status() == array::Status::unscheduled);
      CHECK(left.data_shared_ptr().get() == left_data);
      CHECK(right.data_shared_ptr().get() == right_data);
      CHECK(left.data<float>()[0] == 2.f);
      CHECK(right.data<float>()[1] == 11.f);
      size_t sync_payload = 0;
      for (size_t i = 0; i < sync.blocks; ++i)
        sync_payload += extent(sync.request_bytes[i], sync.reserved_alignment);
      CHECK(role.graph->occupied_bytes() == occupied + sync_payload);
    }
    settle(role); operation.reset();
    CHECK(role.records->occupied_bytes() == 0);
  }
  // The same actual Fixed Eval path completes nonzero Add/Multiply and keeps
  // final aliases through real retirement. Cold component prerequisites remain.
  numerical(Device::cpu);
  const auto stream = new_stream(Device::cpu); prepare(stream, stream);
  Role role; Operation empty;
  const mlx_operation_eval_traversal_limits limits{0, 1, 1, 0, 1, 1, 8};
  REQUIRE(eval_traversal_tests::submit(empty, stream, limits) == 0);
  REQUIRE(mlx_operation_event_wait(empty.value) == 0);
  settle(role); empty.reset();
  CHECK(role.records->occupied_bytes() == 0);
  CHECK(role.graph->occupied_bytes() == 0);
}

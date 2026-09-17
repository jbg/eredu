// Included after wait_record_layout_tests.cpp; reuse its genuine cold streams
// and retained original Role. C declarations were included before using mlx.
#include "mlx/backend/cpu/encoder.h"
#include "mlx/primitives.h"
#include <cstring>
#include <cstdlib>
#include <type_traits>

size_t native_recovery_without_allocations(void (*operation)(void*), void* context);

namespace eval_record_facts {
using wait_record_facts::Role;
using wait_record_facts::prepare;
using wait_record_facts::settle;

struct Operation {
  mlx_submission_observer observer{nullptr};
  mlx_operation_event value{nullptr};
  Operation() {
    REQUIRE(mlx_submission_observer_current(&observer) == 0);
    if (mlx_operation_event_new(&value, observer) != 0) {
      reset();
      FAIL("actual original operation construction");
    }
  }
  void reset() noexcept {
    if (value.ctx) mlx_operation_event_free(value);
    value = {nullptr};
    if (observer.ctx) mlx_submission_observer_release(observer);
    observer = {nullptr};
  }
  ~Operation() { reset(); }
  void append(array& input) {
    REQUIRE(mlx_operation_event_append(value, mlx_array{&input}) == 0);
  }
  unsigned submit(Stream stream) {
    return mlx_operation_event_submit_on_stream(value, mlx_stream{&stream});
  }
};

struct Block { void* pointer; size_t bytes, alignment; };
struct ArenaBlocks {
  submission::RecordQuota* quota;
  std::vector<Block>& blocks;
  explicit ArenaBlocks(submission::RecordQuota* quota, std::vector<Block>& blocks)
      : quota(quota), blocks(blocks) {
    REQUIRE(blocks.empty());
    REQUIRE(blocks.capacity() >= quota->capacity() / alignof(std::max_align_t) + 8);
  }
  void add(size_t bytes, size_t alignment) {
    // Cold bookkeeping is pre-reserved; no growth can lose an allocated block.
    void* pointer = quota->allocate(bytes, alignment);
    blocks.push_back({pointer, bytes, alignment});
  }
  void fill() {
    try {
      for (;;) add(1, 1);
    } catch (const submission::RecordQuotaError& error) {
      REQUIRE(error.cause() == submission::QuotaFailure::exhausted);
    }
  }
  void release(size_t index) {
    auto& b = blocks[index];
    quota->deallocate(b.pointer, b.bytes, b.alignment);
    b.pointer = nullptr;
  }
  ~ArenaBlocks() {
    for (const auto& b : blocks)
      if (b.pointer) quota->deallocate(b.pointer, b.bytes, b.alignment);
    blocks.clear();
  }
};

void sibling_round_trip(Device device) {
  const auto first = new_stream(device), second = new_stream(device);
  prepare(first, second);
  array input({13, 17, 19}), denominator({5, 7, 8});
  Role role;
  auto both = divmod(input, denominator, first);
  REQUIRE(both.size() == 2);
  REQUIRE(both[0].primitive_ptr() == both[1].primitive_ptr());
  REQUIRE(both[0].siblings().size() == 1);
  REQUIRE(both[0].inputs().size() == 2);
  CHECK(both[0].inputs()[0].id() == input.id());
  CHECK(both[0].inputs()[1].id() == denominator.id());
  auto value = add(both[0], both[1], second);
  REQUIRE(value.inputs().size() == 2);
  CHECK(value.inputs()[0].id() == both[0].id());
  CHECK(value.inputs()[1].id() == both[1].id());
  // Visible graph: DivMod sibling group, Add, selected Synchronizer. This
  // fixture shape is not promoted into a general cold graph population bound.
  mlx_operation_eval_record_layout layout{};
  REQUIRE(mlx_operation_event_eval_record_layout(&layout, 3, 2, 4));
  CHECK(layout.record_allocations == 6);
  Operation operation;
  operation.append(value);
  REQUIRE(operation.submit(second) == 0);
  REQUIRE(mlx_operation_event_wait(operation.value) == 0);
  REQUIRE(validate_scoped_array(value, *role.scope) == ScopedEvaluation::complete);
  CHECK(value.data<int>()[0] == 5);
  CHECK(value.data<int>()[1] == 5);
  CHECK(value.data<int>()[2] == 5);
  CHECK(role.records->occupied_bytes() > 0); // observation is not retirement
  settle(role);
  CHECK(role.records->occupied_bytes() == 0);
  operation.reset();
  role.records.reset();
  CHECK(role.record_retired->load() == 0);
  role.scope.reset();
  CHECK(role.record_retired->load() == 1);
}

struct PrefixState {
  std::atomic<bool> release{false}, completed{false};
  std::weak_ptr<array::Data> temporary, output;
  std::vector<Block> scratch;
  PrefixState() { scratch.reserve((256 << 10) / alignof(std::max_align_t) + 8); }
};
struct ReleasePrefix {
  std::shared_ptr<PrefixState> state;
  ~ReleasePrefix() { state->release.store(true, std::memory_order_release); }
};
struct RefuseCaptureAfterDispatch : Primitive {
  std::shared_ptr<PrefixState> state;
  RefuseCaptureAfterDispatch(Stream stream, std::shared_ptr<PrefixState> state)
      : Primitive(stream), state(std::move(state)) {}
  const char* name() const override { return "Eval retained capture prefix"; }
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
    ArenaBlocks occupied(submission::current_record_quota(), state->scratch);
    occupied.fill();
    submission::RecordFixedStorageLayout initial{};
    if (!submission::Record::fixed_storage_layout(1, initial)) std::abort();
    // Force genuine captures_ growth refusal in the actual entered Record,
    // after an accepted task. This is not a synthetic thrown failure or a
    // claim that one of the four earlier frontier reserves failed.
    for (size_t i = 0; i != initial.capture_slots; ++i) {
      auto extra = array::with_allocated_buffer(sizeof(float), Shape{1}, float32);
      (void)extra;
    }
    FAIL("actual exhausted Record capture backing must refuse");
  }
  void eval_gpu(const ArrayVector&, ArrayVector&) override { std::abort(); }
};
} // namespace eval_record_facts

TEST_CASE("Eval Record facts expose six actual requests and reject invalid shapes intact"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using CQuery = bool (*)(mlx_operation_eval_record_layout*, size_t, size_t, size_t);
  using NativeQuery = bool (*)(size_t, size_t, size_t, CompletionEvalRecordLayout&) noexcept;
  static_assert(std::is_same_v<decltype(&mlx_operation_event_eval_record_layout), CQuery>);
  static_assert(std::is_same_v<decltype(&completion_eval_record_layout), NativeQuery>);
  static_assert(sizeof(mlx_operation_eval_record_layout) == 37 * sizeof(size_t));
  static_assert(alignof(mlx_operation_eval_record_layout) == alignof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, tape_entries) == 0 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, stream_count) == 1 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, output_slots) == 2 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, object_bytes) == 3 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, object_alignment) == 4 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, capture_slots) == 5 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, capture_bytes) == 6 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, capture_alignment) == 7 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, stream_state_bytes) == 8 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, stream_state_alignment) == 9 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, stream_receipt_bytes) == 10 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, stream_receipt_alignment) == 11 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, primitive_owner_bytes) == 12 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, primitive_owner_alignment) == 13 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, output_pin_bytes) == 14 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, output_pin_alignment) == 15 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, record_allocations) == 16 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, record_requested_bytes) == 17 * sizeof(size_t));
  static_assert(offsetof(mlx_operation_eval_record_layout, named_control_bytes) == 18 * sizeof(size_t));
  CompletionEvalRecordLayout native{};
  REQUIRE(completion_eval_record_layout(3, 2, 4, native));
  submission::RecordFrontierStorageLayout base{};
  REQUIRE(submission::Record::frontier_storage_layout(2, 4, base));
  CHECK(native.capture_slots == 8);
  CHECK(native.capture_bytes == 8 * sizeof(std::shared_ptr<array::Data>));
  CHECK(native.stream_receipt_bytes == base.fixed.stream_bytes);
  CHECK(native.output_pin_bytes == base.output_bytes);
  CHECK(native.primitive_owner_bytes == 3 * sizeof(std::shared_ptr<Primitive>));
  CHECK(native.record_allocations == 6);
  CHECK(native.record_requested_bytes == native.object_bytes + native.capture_bytes +
      native.stream_state_bytes + native.stream_receipt_bytes +
      native.primitive_owner_bytes + native.output_pin_bytes);
  CHECK(native.object_bytes == submission::minimum_record_object_bytes());
  mlx_operation_eval_record_layout result{};
  REQUIRE(mlx_operation_event_eval_record_layout(&result, 3, 2, 4));
  CHECK(result.record_requested_bytes == native.record_requested_bytes);
  const auto saved = result;
  for (const auto shape : {std::array<size_t, 3>{0, 0, 0}, {1, 0, 1}, {1, 2, 2},
                           {2, 1, 1}, {SIZE_MAX, 1, SIZE_MAX}, {1, 1, SIZE_MAX}}) {
    CHECK_FALSE(mlx_operation_event_eval_record_layout(&result, shape[0], shape[1], shape[2]));
    CHECK(std::memcmp(&result, &saved, sizeof(result)) == 0);
  }
  CHECK_FALSE(mlx_operation_event_eval_record_layout(nullptr, 1, 1, 1));
  REQUIRE(submission::Record::frontier_storage_layout(0, 0, base));
  CHECK(base.fixed.stream_bytes == 0);
  CHECK(base.output_bytes == 0);
  CHECK(base.fixed.capture_bytes != 0); // base construction is still nonzero
}

TEST_CASE("Eval Record query needs no runtime owner or ordinary allocation"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  struct Query { mlx_operation_eval_record_layout layout{}; bool valid{false}; } query;
  auto before = submission::current_scope();
  CHECK(native_recovery_without_allocations([](void* raw) {
    auto& value = *static_cast<Query*>(raw);
    value.valid = mlx_operation_event_eval_record_layout(&value.layout, 3, 2, 4);
  }, &query) == 0);
  REQUIRE(query.valid);
  CHECK(query.layout.named_control_bytes > 0);
  CHECK(submission::current_scope() == before);
}

TEST_CASE("Eval selected empty Synchronizer differs from unselected eventless fast return") {
  using namespace eval_record_facts;
  const auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  array input({23.f, 29.f});
  {
    Role role;
    std::vector<Block> blocks;
    blocks.reserve(role.records->capacity() / alignof(std::max_align_t) + 8);
    Operation selected;
    ArenaBlocks occupied(role.records.get(), blocks);
    occupied.fill();
    const auto before = role.records->occupied_bytes();
    ArrayVector roots{submission::GraphAllocator<array>(role.graph.get())};
    roots.push_back(input);
    Completion skipped;
    CHECK(async_eval_with_completion_scoped(std::move(roots), *role.scope, skipped) ==
        submission::ScopedProgress::observed);
    CHECK(skipped.observe() == EventObservation::ready);
    CHECK(role.records->occupied_bytes() == before);
    CHECK(selected.submit(stream) == static_cast<unsigned>(ScopedEvaluation::failed));
    REQUIRE(role.error.get()->borrow());
    CHECK(role.error.get()->borrow()->exception_type == &typeid(submission::RecordQuotaError));
    settle(role);
    CHECK(role.records->occupied_bytes() == before);
  }
  {
    Role role;
    Operation selected;
    scheduler::CpuStreamToken worker;
    REQUIRE(scheduler::prepared_cpu_stream(stream, worker) == submission::NativeControlFailure::none);
    const auto before = scheduler::cpu_stream_progress(worker).accepted;
    REQUIRE(selected.submit(stream) == 0);
    REQUIRE(mlx_operation_event_wait(selected.value) == 0);
    CHECK(scheduler::cpu_stream_progress(worker).accepted > before);
    CHECK(role.records->occupied_bytes() > 0);
    settle(role);
    CHECK(role.records->occupied_bytes() == 0);
  }
}

TEST_CASE("Eval Record constructor and entered traversal refusals retain their real prefixes"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace eval_record_facts;
  const auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  mlx_operation_eval_record_layout layout{};
  REQUIRE(mlx_operation_event_eval_record_layout(&layout, 1, 1, 1));
  for (size_t prefix = 0; prefix != 3; ++prefix) {
    CAPTURE(prefix);
    Role role;
    Operation operation;
    std::vector<Block> blocks;
    blocks.reserve(role.records->capacity() / alignof(std::max_align_t) + 8);
    ArenaBlocks occupied(role.records.get(), blocks);
    occupied.add(layout.object_bytes, layout.object_alignment);
    occupied.add(1, 1);
    occupied.add(layout.capture_bytes, layout.capture_alignment);
    occupied.add(1, 1);
    occupied.fill();
    if (prefix > 0) occupied.release(0);
    if (prefix > 1) occupied.release(2);
    const auto before = role.records->occupied_bytes();
    scheduler::CpuStreamToken worker;
    REQUIRE(scheduler::prepared_cpu_stream(stream, worker) == submission::NativeControlFailure::none);
    const auto accepted = scheduler::cpu_stream_progress(worker).accepted;
    CHECK(operation.submit(stream) == static_cast<unsigned>(ScopedEvaluation::failed));
    REQUIRE(role.error.get()->borrow());
    CHECK(role.error.get()->borrow()->exception_type == &typeid(submission::RecordQuotaError));
    CHECK(scheduler::cpu_stream_progress(worker).accepted == accepted);
    if (prefix < 2) CHECK(role.records->occupied_bytes() == before);
    else CHECK(role.records->occupied_bytes() > before);
    settle(role);
    CHECK(role.records->occupied_bytes() == before);
  }
}

TEST_CASE("Eval Record CPU sibling frontier retains real stream and output owners"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  eval_record_facts::sibling_round_trip(Device::cpu);
}
#ifdef MLX_C_PATCH_TEST_METAL
TEST_CASE("Eval Record Metal sibling frontier retains real stream and output owners"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  eval_record_facts::sibling_round_trip(Device::gpu);
}
#endif

TEST_CASE("Eval failed accepted prefix retains Data until actual terminal Record retirement"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace eval_record_facts;
  const auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  auto state = std::make_shared<PrefixState>();
  ReleasePrefix release{state}; // installed before any callback can be accepted
  Role role;
  std::optional<array> output;
  output.emplace(Shape{1}, float32,
      submission::make_graph_primitive<RefuseCaptureAfterDispatch>(stream, state), ArrayVector{});
  Operation operation;
  operation.append(*output);
  scheduler::CpuStreamToken worker;
  REQUIRE(scheduler::prepared_cpu_stream(stream, worker) == submission::NativeControlFailure::none);
  const auto before = scheduler::cpu_stream_progress(worker).accepted;
  CHECK(operation.submit(stream) == static_cast<unsigned>(ScopedEvaluation::failed));
  REQUIRE(role.error.get()->borrow());
  CHECK(role.error.get()->borrow()->exception_type == &typeid(submission::RecordQuotaError));
  CHECK(scheduler::cpu_stream_progress(worker).accepted > before);
  operation.reset();
  output.reset();
  CHECK_FALSE(state->temporary.expired());
  CHECK_FALSE(state->output.expired());
  CHECK(role.records->occupied_bytes() > 0);
  CHECK(role.scope->query_records().pending > 0);
  state->release.store(true, std::memory_order_release);
  settle(role);
  CHECK(state->completed.load(std::memory_order_acquire));
  CHECK(state->temporary.expired());
  CHECK(state->output.expired());
  CHECK(role.records->occupied_bytes() == 0);
  state->temporary.reset(); // release actual Graph-owned weak controls
  state->output.reset();
  role.records.reset();
  CHECK(role.record_retired->load() == 0);
  role.scope.reset();
  CHECK(role.record_retired->load() == 1);
}

#if !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION != 210106 || __cplusplus != 202002L
TEST_CASE("Eval Record unqualified producer reports unknown without changing output") {
  using wait_record_facts::check_unknown_unchanged;
  for (const auto shape : {std::array<size_t, 3>{1, 1, 1}, {3, 2, 4},
                           {0, 0, 0}, {SIZE_MAX, 1, SIZE_MAX}}) {
    CAPTURE(shape[0]);
    CAPTURE(shape[1]);
    CAPTURE(shape[2]);
    check_unknown_unchanged<submission::RecordFrontierStorageLayout>([&](auto& out) {
      return submission::Record::frontier_storage_layout(shape[1], shape[2], out);
    });
    check_unknown_unchanged<CompletionEvalRecordLayout>([&](auto& out) {
      return completion_eval_record_layout(shape[0], shape[1], shape[2], out);
    });
    check_unknown_unchanged<mlx_operation_eval_record_layout>([&](auto& out) {
      return mlx_operation_event_eval_record_layout(&out, shape[0], shape[1], shape[2]);
    });
  }
  CHECK_FALSE(mlx_operation_event_eval_record_layout(nullptr, 1, 1, 1));
}
#endif

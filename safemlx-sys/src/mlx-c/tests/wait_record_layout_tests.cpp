// Included by submission_record_quota_tests.cpp; actual same-role wait producer.
#include "mlx/c/event.h"
#include "mlx/c/private/stream.h"
#include "mlx/scoped_evaluation.h"
#include "mlx/completion.h"
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <type_traits>

namespace wait_record_facts {
// Match the producer's compile-time qualification, independently of its result.
// A false query on the qualified build must fail the positive cases, not skip them.
#if defined(_LIBCPP_VERSION) && _LIBCPP_VERSION == 210106 && __cplusplus == 202002L
constexpr bool layout_qualified = true;
#else
constexpr bool layout_qualified = false;
template <typename Layout, typename Query>
void check_unknown_unchanged(Query query) {
  static_assert(std::is_trivially_copyable_v<Layout>);
  Layout value;
  std::memset(&value, 0xa5, sizeof(value));
  std::array<unsigned char, sizeof(Layout)> before;
  std::memcpy(before.data(), &value, sizeof(value));
  CHECK_FALSE(query(value));
  CHECK(std::memcmp(&value, before.data(), sizeof(value)) == 0);
}
#endif
struct ReleaseScope {
  void operator()(submission::Scope* value) const noexcept {
    value->seal();
    value->release();
  }
};
struct ReleaseRecords {
  void operator()(submission::RecordQuota* value) const noexcept { value->release(); }
};
struct ReleaseGraph {
  void operator()(submission::GraphQuota* value) const noexcept { value->release(); }
};
void release_token(void* value) { delete static_cast<int*>(value); }
FailureCarrierRef failure() {
  auto token = std::make_unique<int>(0);
  auto owner = FailureCarrier::create(token.get(), release_token);
  if (owner) token.release();
  return owner;
}
void retired_records(void* raw) {
  std::unique_ptr<std::shared_ptr<std::atomic<size_t>>> owner(
      static_cast<std::shared_ptr<std::atomic<size_t>>*>(raw));
  ++**owner;
}
std::unique_ptr<submission::RecordQuota, ReleaseRecords> record_arena(
    const std::shared_ptr<std::atomic<size_t>>& count) {
  auto owner = std::make_unique<std::shared_ptr<std::atomic<size_t>>>(count);
  auto* quota = submission::RecordQuota::create(256 << 10, owner.get(), retired_records);
  owner.release();
  return std::unique_ptr<submission::RecordQuota, ReleaseRecords>(quota);
}
struct Role {
  std::shared_ptr<std::atomic<size_t>> record_retired{std::make_shared<std::atomic<size_t>>(0)};
  std::unique_ptr<submission::RecordQuota, ReleaseRecords> records{record_arena(record_retired)};
  std::unique_ptr<submission::GraphQuota, ReleaseGraph> graph{
      submission::GraphQuota::create(2 << 20, nullptr, nullptr)};
  FailureCarrierRef error{failure()};
  std::unique_ptr<submission::Scope, ReleaseScope> scope;
  explicit Role(mlx_pipeline_cache pipeline = {}) {
    // Install while Graph has its sole constructor reference, before Scope
    // retains it. The default fixture keeps the existing no-pipeline behavior.
    if (pipeline.ctx) REQUIRE(mlx_pipeline_cache_install(pipeline, {graph.get()}) == 0);
    scope.reset(new submission::Scope(nullptr, nullptr, records.get(), graph.get()));
    REQUIRE(error);
    REQUIRE(scope->enable_scoped_observation());
    REQUIRE(scope->require_original_controls() == submission::NativeControlFailure::none);
    REQUIRE(scope->bind_failure(error));
    REQUIRE(scope->enable_original_controls() == submission::NativeControlFailure::none);
  }
};
void prepare(Stream first, Stream second) {
  // Real cold native setup, before any original role, as ordinary model workers.
  auto input = array({2.f, 3.f, 5.f});
  eval(add(negative(input, first), input, second));
  mlx_submission_runtime_baseline baseline{};
  REQUIRE(mlx_submission_prepare_runtime(&baseline, mlx_stream{&first}, mlx_stream{&second}) == 0);
}
void settle(Role& role) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (role.scope->query_records().pending) {
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    REQUIRE(role.scope->progress_scoped() == submission::ScopedProgress::observed);
    std::this_thread::yield();
  }
  for (;;) {
    const auto result = role.scope->retire_scoped_records();
    if (result == submission::ScopedProgress::observed) break;
    REQUIRE(result == submission::ScopedProgress::busy);
    REQUIRE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
}
Completion submit(Role& role, array& value, Stream stream) {
  Completion result;
  ArrayVector roots{submission::GraphAllocator<array>(role.graph.get())};
  roots.push_back(value);
  REQUIRE(async_eval_with_completion_scoped_on_stream(
      std::move(roots), *role.scope, result, stream) == submission::ScopedProgress::observed);
  return result;
}
void ready(Role& role, Completion& completion, array& value) {
  REQUIRE(wait_scoped_completion(*role.scope, completion) == ScopedEvaluation::complete);
  REQUIRE(validate_scoped_array(value, *role.scope) == ScopedEvaluation::complete);
  settle(role);
}
struct Holes {
  struct Block { void* value; size_t bytes, alignment; };
  submission::RecordQuota* quota;
  std::vector<Block> blocks;
  std::array<size_t, 3> requests{};
  explicit Holes(submission::RecordQuota* quota, const CompletionWaitRecordLayout& layout)
      : quota(quota) {
    // Test bookkeeping is ordinary cold storage; every physical arena block
    // consumes at least max_align_t alignment, so push_back cannot grow here.
    blocks.reserve(quota->capacity() / alignof(std::max_align_t) + 8);
    const std::array<std::pair<size_t, size_t>, 3> sizes{{
        {layout.object_bytes, layout.object_alignment},
        {layout.capture_bytes, layout.capture_alignment},
        {layout.stream_bytes, layout.stream_alignment}}};
    for (size_t i = 0; i != sizes.size(); ++i) {
      const auto [bytes, alignment] = sizes[i];
      requests[i] = blocks.size();
      blocks.push_back({quota->allocate(bytes, alignment), bytes, alignment});
      // Live separators prevent adjacent holes coalescing and changing sites.
      blocks.push_back({quota->allocate(1, 1), 1, 1});
    }
    try {
      for (;;) blocks.push_back({quota->allocate(1, 1), 1, 1});
    } catch (const submission::RecordQuotaError& error) {
      REQUIRE(error.cause() == submission::QuotaFailure::exhausted);
    }
  }
  void allow_prefix(size_t count) {
    for (size_t i = 0; i != count; ++i) {
      auto& block = blocks[requests[i]];
      quota->deallocate(block.value, block.bytes, block.alignment);
      block.value = nullptr;
    }
  }
  ~Holes() {
    for (const auto& block : blocks)
      if (block.value) quota->deallocate(block.value, block.bytes, block.alignment);
  }
};
void round_trip(Device device) {
  const auto producer = new_stream(device), consumer = new_stream(device);
  prepare(producer, consumer);
  auto left = array({2.f, 3.f, 5.f}), right = array({7.f, 11.f, 13.f});
  Role role;
  auto value = multiply(left, right, producer);
  auto completion = submit(role, value, producer);
  ready(role, completion, value);
  CHECK(role.records->occupied_bytes() == 0);
  // Completion is already signaled, but each actual consumer wait still owns
  // its distinct Record. Repeated host observation creates no such records.
  for (int i = 0; i != 3; ++i) {
    REQUIRE(wait_scoped_completion(*role.scope, completion) == ScopedEvaluation::complete);
    CHECK(role.records->occupied_bytes() == 0);
  }
  REQUIRE(completion.wait_scoped(*role.scope, consumer) == ScopedEvaluation::complete);
  const auto first = role.records->occupied_bytes();
  CHECK(first > 0);
  REQUIRE(completion.wait_scoped(*role.scope, consumer) == ScopedEvaluation::complete);
  CHECK(role.records->occupied_bytes() > first);
  // A selected consumer frontier commits any Metal waits through the shared evaluator.
  auto consumer_value = add(value, left, consumer);
  auto consumed = submit(role, consumer_value, consumer);
  ready(role, consumed, consumer_value);
  CHECK(consumer_value.data<float>()[0] == 16.f);
  CHECK(consumer_value.data<float>()[1] == 36.f);
  CHECK(consumer_value.data<float>()[2] == 70.f);
  CHECK(role.records->occupied_bytes() == 0);
  CHECK(role.graph->occupied_bytes() > 0); // actual result still owns Data/descriptor
  CHECK(role.record_retired->load() == 0);
  role.records.reset();
  CHECK(role.record_retired->load() == 0); // Scope still owns the physical arena
  role.scope.reset();
  CHECK(role.record_retired->load() == 1); // blocks, then arena, then custody
}
} // namespace wait_record_facts

TEST_CASE("wait Record facts expose all three qualified requests and checked populations"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  CompletionWaitRecordLayout native{};
  REQUIRE(completion_wait_record_layout(native));
  CHECK(native.capture_slots == 8);
  CHECK(native.capture_bytes == 8 * sizeof(std::shared_ptr<array::Data>));
  CHECK(native.stream_receipts == 1);
  CHECK(native.record_allocations == 3);
  CHECK(native.record_requested_bytes == native.object_bytes + native.capture_bytes + native.stream_bytes);
  CHECK(native.named_control_bytes > 0);
  mlx_operation_wait_record_layout result{};
  REQUIRE(mlx_operation_event_wait_record_layout(&result, 3));
  CHECK(result.total_record_allocations == 9);
  CHECK(result.total_record_requested_bytes == 3 * native.record_requested_bytes);
  const auto before = result;
  CHECK_FALSE(mlx_operation_event_wait_record_layout(&result, SIZE_MAX));
  CHECK(result.wait_count == before.wait_count);
  CHECK(result.total_record_requested_bytes == before.total_record_requested_bytes);
  CHECK_FALSE(mlx_operation_event_wait_record_layout(nullptr, 1));
  REQUIRE(mlx_operation_event_wait_record_layout(&result, 0));
  CHECK(result.total_record_allocations == 0);
  CHECK(result.total_record_requested_bytes == 0);
  CHECK(result.requested_bytes_per_wait == native.record_requested_bytes);
  static_assert(sizeof(mlx_operation_wait_record_layout) == 14 * sizeof(size_t));
  static_assert(alignof(mlx_operation_wait_record_layout) == alignof(size_t));
}

TEST_CASE("wait Record refuses each actual allocation before consumer dispatch and retires its prefix"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace wait_record_facts;
  CompletionWaitRecordLayout layout{};
  REQUIRE(completion_wait_record_layout(layout));
  const auto producer = new_stream(Device::cpu), consumer = new_stream(Device::cpu);
  prepare(producer, consumer);
  auto input = array({13.f, 17.f, 19.f});
  for (size_t allowed = 0; allowed != 4; ++allowed) {
    CAPTURE(allowed);
    Role role;
    auto completion = submit(role, input, producer);
    ready(role, completion, input);
    REQUIRE(role.records->occupied_bytes() == 0);
    scheduler::CpuStreamToken worker;
    REQUIRE(scheduler::prepared_cpu_stream(consumer, worker) == submission::NativeControlFailure::none);
    const auto accepted = scheduler::cpu_stream_progress(worker).accepted;
    {
      Holes holes(role.records.get(), layout);
      holes.allow_prefix(allowed);
      const auto before = role.records->occupied_bytes();
      // An eventless token performs no allocation even in this full arena.
      Completion empty;
      CHECK(empty.wait_scoped(*role.scope, consumer) == ScopedEvaluation::complete);
      CHECK(role.records->occupied_bytes() == before);
      const auto status = completion.wait_scoped(*role.scope, consumer);
      if (allowed < 3) {
        CHECK(status == ScopedEvaluation::failed);
        REQUIRE(role.error.get()->borrow());
        CHECK(role.error.get()->borrow()->exception_type == &typeid(submission::RecordQuotaError));
        CHECK(scheduler::cpu_stream_progress(worker).accepted == accepted);
      } else {
        CHECK(status == ScopedEvaluation::complete);
        CHECK_FALSE(role.error.get()->borrow());
        CHECK(scheduler::cpu_stream_progress(worker).accepted == accepted + 1);
      }
      if (allowed < 2) CHECK(role.records->occupied_bytes() == before);
      else CHECK(role.records->occupied_bytes() > before); // entered object/capture retained
      settle(role);
      CHECK(role.records->occupied_bytes() == before);
    }
    CHECK(role.records->occupied_bytes() == 0);
    CHECK(input.data<float>()[1] == 17.f);
  }
}

TEST_CASE("wait Record validates scope device and missing worker before consumer work") {
  using namespace wait_record_facts;
  const auto producer = new_stream(Device::cpu), consumer = new_stream(Device::cpu);
  prepare(producer, consumer);
  // Cold stream creation alone does not prepare a Scheduler worker.
  const auto absent = new_stream(Device::cpu);
  auto input = array({23.f, 29.f});
  Role role;
  auto completion = submit(role, input, producer);
  ready(role, completion, input);
  const auto before = role.records->occupied_bytes();
  CHECK(completion.wait_scoped(*role.scope, Stream{consumer.index, Device{Device::cpu, 1000}}) == ScopedEvaluation::domain);
  CHECK(role.records->occupied_bytes() == before);
  {
    // Genuine child is current, but this completion belongs to its parent.
    std::unique_ptr<submission::Scope, ReleaseScope> child{new submission::Scope};
    CHECK(completion.wait_scoped(*role.scope, consumer) == ScopedEvaluation::domain);
    auto other = failure();
    REQUIRE(child->enable_scoped_observation());
    REQUIRE(child->require_original_controls() == submission::NativeControlFailure::none);
    REQUIRE(child->bind_failure(other));
    REQUIRE(child->enable_original_controls() == submission::NativeControlFailure::none);
    CHECK(completion.wait_scoped(*child, consumer) == ScopedEvaluation::domain);
    CHECK(role.records->occupied_bytes() == before);
  }
  CHECK(completion.wait_scoped(*role.scope, absent) == ScopedEvaluation::failed);
  REQUIRE(role.error.get()->borrow());
  CHECK(role.error.get()->borrow()->exception_type == &typeid(submission::NativeControlError));
  settle(role);
  CHECK(role.records->occupied_bytes() == 0);
}

TEST_CASE("wait Record CPU completed producer retains two real consumer receipts") {
  wait_record_facts::round_trip(Device::cpu);
}
#ifdef MLX_C_PATCH_TEST_METAL
TEST_CASE("wait Record Metal completed producer retains two real consumer receipts") {
  wait_record_facts::round_trip(Device::gpu);
}
#endif

#if !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION != 210106 || __cplusplus != 202002L
TEST_CASE("wait Record unqualified producer reports unknown without changing output") {
  using wait_record_facts::check_unknown_unchanged;
  check_unknown_unchanged<CompletionWaitRecordLayout>([](auto& out) {
    return completion_wait_record_layout(out);
  });
  for (size_t count : {size_t{0}, size_t{1}, size_t{7}, SIZE_MAX}) {
    CAPTURE(count);
    check_unknown_unchanged<submission::RecordFixedStorageLayout>([&](auto& out) {
      return submission::Record::fixed_storage_layout(count, out);
    });
    check_unknown_unchanged<mlx_operation_wait_record_layout>([&](auto& out) {
      return mlx_operation_event_wait_record_layout(&out, count);
    });
  }
  CHECK_FALSE(mlx_operation_event_wait_record_layout(nullptr, 1));
}
#endif

// Included by operation_observer_tests.cpp; genuine configured original Owner.
#include "mlx/c/owned_host_copy.h"
#include <array>
#include <cstring>

namespace {
struct CopySource {
  unsigned retired{0};
  GraphQuota* arena{nullptr};
  mlx_owned_host_copy_slot slot{};
  mlx_owned_host_copy_layout layout{};
  CopySource(mlx_prepared_input_runtime runtime, const int* shape, size_t rank, mlx_dtype dtype) {
    REQUIRE(mlx_owned_host_copy_layout_for(&layout, runtime, shape, rank, dtype) == 0);
    CHECK(layout.strategy == MLX_OWNED_HOST_COPY_STRATEGY);
    arena = GraphQuota::create(layout.metadata_bytes, &retired,
        [](void* p) { ++*static_cast<unsigned*>(p); });
    REQUIRE(arena != nullptr);
    REQUIRE(mlx_owned_host_copy_new(&slot, runtime, mlx_submission_graph_quota{arena}, shape, rank, dtype) == 0);
  }
  void release_slot() { mlx_owned_host_copy_free(slot); slot = {}; }
  void release_arena() { if (arena) arena->release(); arena = nullptr; }
  ~CopySource() { release_slot(); release_arena(); }
};
}
void original_owned_host_copy(Device::DeviceType device) {
  auto stream = default_stream(device);
  mlx_submission_runtime_baseline baseline{};
  REQUIRE(mlx_submission_prepare_runtime(&baseline, mlx_stream{&stream}, mlx_stream{&stream}) == 0);
  mlx_prepared_input_runtime runtime{};
  REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
  const int shape[] = {3};
  // Metadata/final handle precede the actual current host payload and role.
  CopySource source(runtime, shape, 1, MLX_FLOAT32);
  CHECK(source.layout.copy_bytes == 3 * sizeof(float));
  CHECK(source.layout.backing_bytes >= source.layout.copy_bytes);
  CHECK(mlx_owned_host_copy_state(source.slot) == 0);
  const float input[] = {2.0f, -3.0f, 7.0f};
  Owner owner;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  const auto graph = owner.graph->occupied_bytes();
  const auto records = owner.records->occupied_bytes();
  mlx_array result{}, refused{};
  CHECK(mlx_owned_host_copy_fill(&refused, source.slot, {}, input, sizeof(input)) == 6);
  CHECK(mlx_owned_host_copy_fill(&refused, source.slot, observer, input, sizeof(input) - 1) == 1);
  CHECK(mlx_owned_host_copy_state(source.slot) == 0);
  CHECK(refused.ctx == nullptr);
  REQUIRE(mlx_owned_host_copy_fill(&result, source.slot, observer, input, sizeof(input)) == 0);
  REQUIRE(result.prepared_owner != nullptr);
  CHECK(mlx_owned_host_copy_state(source.slot) == 1);
  CHECK(mlx_owned_host_copy_fill(&refused, source.slot, observer, input, sizeof(input)) == 7);
  CHECK(refused.ctx == nullptr);
  CHECK(mlx_array_get_(result).data<float>() != input);
  for (unsigned i = 0; i != 3; ++i) CHECK(mlx_array_get_(result).data<float>()[i] == input[i]);
  {
    auto doubled = add(mlx_array_get_(result), mlx_array_get_(result), stream);
    mlx_operation_event event{};
    REQUIRE(mlx_operation_event_new_exact(&event, observer, mlx_stream{&stream}, 1) == 0);
    REQUIRE(mlx_operation_event_append(event, mlx_array{&doubled}) == 0);
    REQUIRE(mlx_operation_event_submit_on_stream(event, mlx_stream{&stream}) == 0);
    REQUIRE(mlx_operation_event_wait(event) == 0);
    await_original_terminal(owner);
    REQUIRE(mlx_submission_observer_validate_array(observer, mlx_array{&doubled}) == 0);
    for (unsigned i = 0; i != 3; ++i) CHECK(doubled.data<float>()[i] == input[i] * 2);
    mlx_operation_event_free(event);
  }
  await_original_terminal(owner);
  REQUIRE(mlx_submission_observer_retire(observer) == 0);
  CHECK(owner.graph->occupied_bytes() == graph);
  CHECK(owner.records->occupied_bytes() == records);
  source.release_slot();
  source.release_arena();
  CHECK(source.retired == 0);
  // The prepared outer handle is now the last source owner: its actual block
  // must be returned before the source callback, not merely its Array value.
  REQUIRE(mlx_array_free(result) == 0);
  CHECK(source.retired == 1);
  CHECK(owner.scope->active_on_owner_thread());
  mlx_submission_observer_release(observer);
}
TEST_CASE("prepared owned host copy preserves CPU nonzero completion and final handle custody") {
  original_owned_host_copy(Device::cpu);
}
#ifdef MLX_C_PATCH_TEST_METAL
TEST_CASE("prepared owned host copy preserves Metal nonzero completion and final handle custody") {
  original_owned_host_copy(Device::gpu);
}
#endif
TEST_CASE("prepared owned host copy supports scalar empty rank and all non-bool storage dtypes") {
  mlx_prepared_input_runtime runtime{};
  REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
  alignas(16) const std::array<unsigned char, 16> bytes{1, 2, 3, 4, 5, 6, 7, 8};
  for (unsigned dtype = MLX_UINT8; dtype <= MLX_COMPLEX64; ++dtype) {
    CopySource source(runtime, nullptr, 0, static_cast<mlx_dtype>(dtype));
    Owner owner;
    mlx_submission_observer observer{};
    REQUIRE(mlx_submission_observer_current(&observer) == 0);
    mlx_array output{};
    REQUIRE(source.layout.copy_bytes <= bytes.size());
    REQUIRE(mlx_owned_host_copy_fill(&output, source.slot, observer, bytes.data(), source.layout.copy_bytes) == 0);
    CHECK(mlx_array_get_(output).ndim() == 0);
    CHECK(mlx_array_get_(output).size() == 1);
    CHECK(std::memcmp(mlx_array_get_(output).data<char>(), bytes.data(), source.layout.copy_bytes) == 0);
    // A cached/shared descriptor alias can outlive its first prepared wrapper.
    std::optional<array> cached{mlx_array_get_(output)};
    REQUIRE(mlx_array_free(output) == 0);
    source.release_slot(); source.release_arena();
    CHECK(source.retired == 0);
    cached.reset();
    CHECK(source.retired == 1);
    mlx_submission_observer_release(observer);
  }
  for (const auto& shape : {std::vector<int>{2, 0, 3}, std::vector<int>(12, 1)}) {
    CopySource source(runtime, shape.data(), shape.size(), MLX_FLOAT32);
    Owner owner;
    mlx_submission_observer observer{};
    REQUIRE(mlx_submission_observer_current(&observer) == 0);
    mlx_array output{};
    REQUIRE(mlx_owned_host_copy_fill(&output, source.slot, observer,
        source.layout.copy_bytes ? bytes.data() : nullptr, source.layout.copy_bytes) == 0);
    CHECK(mlx_array_get_(output).ndim() == shape.size());
    CHECK(mlx_array_get_(output).size() == (shape.size() == 12 ? 1 : 0));
    if (!source.layout.copy_bytes) CHECK(source.layout.backing_bytes == 0);
    REQUIRE(mlx_array_free(output) == 0);
    source.release_slot(); source.release_arena();
    CHECK(source.retired == 1);
    mlx_submission_observer_release(observer);
  }
  mlx_owned_host_copy_layout untouched{};
  CHECK(mlx_owned_host_copy_layout_for(&untouched, runtime, nullptr, 0, MLX_BOOL) == 1);
  const int invalid[] = {-1};
  CHECK(mlx_owned_host_copy_layout_for(&untouched, runtime, invalid, 1, MLX_FLOAT32) == 1);
  CHECK(untouched.metadata_bytes == 0);
}
TEST_CASE("prepared owned host copy metadata refusal preserves source owner and leaves no published prefix") {
  mlx_prepared_input_runtime runtime{};
  REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
  const int shape[] = {2};
  mlx_owned_host_copy_layout layout{};
  REQUIRE(mlx_owned_host_copy_layout_for(&layout, runtime, shape, 1, MLX_FLOAT32) == 0);
  unsigned retired = 0;
  auto* arena = GraphQuota::create(64, &retired, [](void* p) { ++*static_cast<unsigned*>(p); });
  REQUIRE(arena != nullptr);
  mlx_owned_host_copy_slot slot{};
  CHECK(mlx_owned_host_copy_new(&slot, runtime, mlx_submission_graph_quota{arena}, shape, 1, MLX_FLOAT32) == 4);
  CHECK(slot.ctx == nullptr);
  CHECK(arena->occupied_bytes() == 0);
  CHECK(retired == 0);
  arena->release();
  CHECK(retired == 1);
}

namespace {
// Test-only injection at the real allocator interface. Actual facts/backing and
// release delegate to the installed allocator; this is not platform exhaustion.
struct RefusingCopyAllocator final : allocator::Allocator {
  allocator::Allocator& actual;
  bool after_backing;
  unsigned calls{0};
  unsigned allocated{0};
  unsigned freed{0};
  RefusingCopyAllocator(allocator::Allocator& value, bool after)
      : actual(value), after_backing(after) {}
  allocator::Buffer malloc(size_t n) override { return actual.malloc(n); }
  void free(allocator::Buffer b) override { actual.free(b); }
  size_t size(allocator::Buffer b) const override { return actual.size(b); }
  allocator::PreparedInputCause allocate_prepared_input(
      size_t n, allocator::PreparedInputAllocation& output) noexcept override {
    ++calls;
    if (!after_backing) return allocator::PreparedInputCause::busy;
    const auto result = actual.allocate_prepared_input(n, output);
    if (result != allocator::PreparedInputCause::success) return result;
    ++allocated;
    output.allocator = this;
    return allocator::PreparedInputCause::capacity;
  }
  void free_prepared_input(allocator::PreparedInputAllocation& input) noexcept override {
    ++freed;
    actual.free_prepared_input(input);
  }
};
}
TEST_CASE("prepared owned host copy retains spent allocator-refusal prefix without second attempt") {
  mlx_prepared_input_runtime runtime{};
  REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
  const int shape[] = {2};
  const float input[] = {13.0f, -7.0f};
  for (bool after_backing : {false, true}) {
    RefusingCopyAllocator injected(*static_cast<allocator::Allocator*>(runtime.allocator), after_backing);
    auto selected = runtime;
    selected.allocator = &injected;
    CopySource source(selected, shape, 1, MLX_FLOAT32);
    Owner owner;
    mlx_submission_observer observer{};
    REQUIRE(mlx_submission_observer_current(&observer) == 0);
    mlx_array output{};
    CHECK(mlx_owned_host_copy_fill(&output, source.slot, observer, input, sizeof(input)) == (after_backing ? 4 : 2));
    CHECK(output.ctx == nullptr);
    CHECK(output.prepared_owner == nullptr);
    CHECK(mlx_owned_host_copy_state(source.slot) == 1);
    CHECK(injected.calls == 1);
    CHECK(injected.allocated == (after_backing ? 1 : 0));
    CHECK(injected.freed == 0);
    CHECK(mlx_owned_host_copy_fill(&output, source.slot, observer, input, sizeof(input)) == 7);
    CHECK(injected.calls == 1);
    CHECK(source.retired == 0);
    source.release_slot();
    CHECK(injected.freed == injected.allocated);
    CHECK(source.retired == 0);
    source.release_arena();
    CHECK(source.retired == 1);
    CHECK(owner.scope->query().activity != Activity::pending);
    CHECK_FALSE(owner.scope->query().failed);
    mlx_submission_observer_release(observer);
  }
}

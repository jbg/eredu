#include "mlx/backend/cpu/gather_mm_storage.h"

namespace tiled_gather_mm_tests {
size_t captures(size_t actual) {
  mlx::core::submission::RecordFixedStorageLayout base;
  REQUIRE(mlx::core::submission::Record::fixed_storage_layout(1, base));
  return std::max(base.capture_slots, actual);
}
}

TEST_CASE("CPU selected GatherMM source checks physical matrices indices and immutable selection"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  const Stream selected(stream.index, stream.device, CpuMatmulKernel::Float32Tiles);
  std::array<float, 24> left{}, right{};
  array a(left.data(), {2, 3, 4}, float32), b(right.data(), {2, 4, 3}, float32);
  const uint32_t indices[]{1, 0}; array ids(indices, {2}, uint32);
  auto value = gather_mm(a, b, ids, ids, false, selected);
  auto ordinary = gather_mm(a, b, ids, ids, false, stream);
  CHECK_FALSE(value.primitive().is_equivalent(ordinary.primitive()));
  cpu::CopyEvalStorage cold, actual;
  REQUIRE(cpu::tiled_gather_mm_eval_layout(3, 3, 1, 3, 3, 4, 2, false, cold));
  REQUIRE(cpu::tiled_gather_mm_eval_storage(value, actual));
  CHECK(actual.inputs == 4); CHECK(actual.backing_births == 1);
  CHECK(actual.worker_graph_extents == 0);
  CHECK(actual.allocation_extents == cold.allocation_extents);
  CHECK(actual.named_control_bytes == cold.named_control_bytes);
  const auto saved = actual;
  CHECK_FALSE(cpu::tiled_gather_mm_eval_storage(ordinary, actual));
  CHECK_FALSE(cpu::tiled_gather_mm_eval_layout(1, 3, 1, 3, 3, 4, 2, false, actual));
  CHECK_FALSE(cpu::tiled_gather_mm_eval_layout(3, 3, 4, 3, 3, 4, 2, false, actual));
  CHECK_FALSE(cpu::tiled_gather_mm_eval_layout(3, 3, 0, 3, 3, 4, 2, false, actual));
  CHECK_FALSE(cpu::tiled_gather_mm_eval_layout(3, 3, 1, 0, 3, 4, 2, false, actual));
  CHECK_FALSE(cpu::tiled_gather_mm_eval_layout(3, 3, 1, INT_MAX, 3, 4, 2, false, actual));
  auto wrong_indices = array(indices, {2}, int32);
  auto wrong = array({2, 3, 3}, float32, std::make_shared<GatherMM>(selected),
      {a, b, wrong_indices, ids});
  CHECK_FALSE(cpu::tiled_gather_mm_eval_storage(wrong, actual));
  auto short_backing = array(a.shape(), float32, nullptr, {});
  short_backing.copy_shared_buffer(array(1.f));
  auto unreadable = gather_mm(short_backing, b, ids, ids, false, selected);
  CHECK_FALSE(cpu::tiled_gather_mm_eval_storage(unreadable, actual));
  CHECK(std::memcmp(&saved, &actual, sizeof(actual)) == 0);
  mlx_cpu_copy_eval_layout query{};
  REQUIRE(mlx_operation_event_cpu_tiled_gather_mm_eval_layout(&query, 3, 3, 1, 3, 3, 4, 2, false));
  CHECK(query.graph_extents == cold.allocation_extents);
  CHECK(query.backing_births == 1);
  const auto query_saved = query;
  CHECK_FALSE(mlx_operation_event_cpu_tiled_gather_mm_eval_layout(&query, 3, 3, 1, 3, 3, 0, 2, false));
  CHECK(std::memcmp(&query, &query_saved, sizeof(query)) == 0);
}

TEST_CASE("CPU selected GatherMM shares tiled tails transpose repeated indices and original custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  const Stream selected(stream.index, stream.device, CpuMatmulKernel::Float32Tiles);
  constexpr int groups = 3, batches = 4, m = 17, n = 19, k = 23;
  std::vector<float> left(groups * m * k), right(groups * n * k);
  for (size_t i = 0; i < left.size(); ++i) left[i] = (int(i % 29) - 14) * .03125f;
  for (size_t i = 0; i < right.size(); ++i) right[i] = (int(i % 17) - 8) * .0625f;
  array a(left.data(), {groups, m, k}, float32), bank(right.data(), {groups, n, k}, float32);
  auto b = swapaxes(bank, 1, 2, selected); eval(b);
  const uint32_t lhs[]{2, 0, 2, 1}, rhs[]{1, 2, 0, 1};
  array ai(lhs, {batches}, uint32), bi(rhs, {batches}, uint32);
  auto ordinary = gather_mm(a, b, ai, bi, false, selected);
  auto platform = gather_mm(a, b, ai, bi, false, stream); eval(ordinary, platform);
  for (int batch = 0; batch < batches; ++batch) for (int row = 0; row < m; ++row)
    for (int col = 0; col < n; ++col) {
      float expected = 0;
      for (int inner = 0; inner < k; ++inner)
        expected += left[(lhs[batch] * m + row) * k + inner] *
            right[(rhs[batch] * n + col) * k + inner];
      const size_t i = (batch * m + row) * n + col;
      CHECK(ordinary.data<float>()[i] == expected);
      CHECK(std::abs(platform.data<float>()[i] - expected) < 1e-5f);
    }
  mlx_prepared_input_runtime runtime{}; REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
  struct Budget { mlx_original_buffer_budget value{}; ~Budget() { mlx_original_buffer_budget_release(value); } } budget;
  unsigned retired = 0;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value, runtime, 1 << 20, &retired,
      [](void* p) { ++*static_cast<unsigned*>(p); }) == 0);
  std::optional<array> escaped;
  {
    Role role; REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()}, budget.value) == 0);
    Observer observer; Bank constructors;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&constructors.value, observer.value, 1, 0, 3) == 0);
    auto value = gather_mm(a, b, ai, bi, false, selected); constructors.reset();
    Operation operation; operation.append(value);
    // Four leaves, GatherMM and completion; one actual fresh Data capture.
    const mlx_operation_eval_traversal_limits limits{1, 6, 2, 5, 2, 1, tiled_gather_mm_tests::captures(1)};
    REQUIRE(eval_traversal_tests::submit(operation, selected, limits) == 0);
    eval_traversal_tests::complete(role, operation, value);
    for (size_t i = 0; i < value.size(); ++i) CHECK(value.data<float>()[i] == ordinary.data<float>()[i]);
    CHECK(role.records->occupied_bytes() == 0);
    mlx_original_buffer_info info{};
    REQUIRE(mlx_original_buffer_array_info(&info, {&value}, budget.value) == 0);
    REQUIRE(info.known); escaped.emplace(value);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value) > 0);
  mlx_original_buffer_budget_release(budget.value); budget.value = {};
  CHECK(retired == 0); escaped.reset(); CHECK(retired == 1);
}

TEST_CASE("CPU selected GatherMM original frontend uses actual U32 range and cast sources"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  const Stream selected(stream.index, stream.device, CpuMatmulKernel::Float32Tiles);
  constexpr int rows = 5, groups = 3, k = 7, n = 4;
  std::array<float, rows * k> left; std::array<float, groups * n * k> right;
  for (size_t i = 0; i < left.size(); ++i) left[i] = (int(i % 11) - 5) * .25f;
  for (size_t i = 0; i < right.size(); ++i) right[i] = (int(i % 13) - 6) * .125f;
  array a(left.data(), {rows, 1, k}, float32), weights(right.data(), {groups, n, k}, float32);
  auto b = swapaxes(weights, 1, 2, selected); eval(b);
  const int32_t indices[]{2, 0, 2, 1, 0}; array ids(indices, {rows}, int32);
  auto ordinary = gather_mm(a, b, std::nullopt, ids, false, selected); eval(ordinary);
  mlx_prepared_input_runtime runtime{}; REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
  struct Budget { mlx_original_buffer_budget value{}; ~Budget() { mlx_original_buffer_budget_release(value); } } budget;
  unsigned retired = 0;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value, runtime, 1 << 20, &retired,
      [](void* p) { ++*static_cast<unsigned*>(p); }) == 0);
  {
    Role role; REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()}, budget.value) == 0);
    Observer observer; Bank constructors;
    // Arange, I32-to-U32 Cast, GatherMM. Shape-preserving frontend calls alias.
    REQUIRE(mlx_operation_event_prepare_resident_graph(&constructors.value, observer.value, 3, 0, 3) == 0);
    auto value = gather_mm(a, b, std::nullopt, ids, false, selected);
    REQUIRE(value.inputs().size() == 4);
    CHECK(value.inputs()[2].dtype() == uint32); CHECK(value.inputs()[3].dtype() == uint32);
    constructors.reset(); Operation operation; operation.append(value);
    // Three source leaves plus the three operations and completion. Arange and
    // GatherMM each publish one Data; Cast also creates two weak Data captures.
    const mlx_operation_eval_traversal_limits limits{1, 7, 4, 6, 4, 1, tiled_gather_mm_tests::captures(5)};
    REQUIRE(eval_traversal_tests::submit(operation, selected, limits) == 0);
    eval_traversal_tests::complete(role, operation, value);
    for (size_t i = 0; i < value.size(); ++i) CHECK(value.data<float>()[i] == ordinary.data<float>()[i]);
  }
  CHECK(mlx_original_buffer_budget_occupied(budget.value) == 0);
}

TEST_CASE("CPU U32 coordinate source preserves exact integers and refuses unknown dtype"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  cpu::CopyEvalStorage cold, actual;
  REQUIRE(cpu::arange_int_eval_layout(uint32, 7, false, cold));
  for (int offset : {0, 16777217, INT_MAX - 7}) {
    auto value = arange(double(offset), double(offset) + 7.0, 1.0, uint32, stream);
    REQUIRE(cpu::arange_int_eval_storage(value, actual));
    CHECK(actual.named_control_bytes == cold.named_control_bytes);
    eval(value);
    for (uint32_t i = 0; i < 7; ++i) CHECK(value.data<uint32_t>()[i] == uint32_t(offset) + i);
  }
  const auto saved = actual;
  CHECK_FALSE(cpu::arange_int_eval_layout(float32, 7, false, actual));
  CHECK_FALSE(cpu::arange_int_eval_layout(uint64, 7, false, actual));
  CHECK(std::memcmp(&actual, &saved, sizeof(actual)) == 0);
}

TEST_CASE("CPU selected GatherMM checks queued index bounds and retains failed output custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  const Stream selected(stream.index, stream.device, CpuMatmulKernel::Float32Tiles);
  const float values[]{1, 2, 3, 4};
  array a(values, {1, 2, 2}, float32), b(values, {1, 2, 2}, float32);
  const uint32_t good[]{0}, bad[]{1};
  array ai(good, {1}, uint32), bi(bad, {1}, uint32);
  mlx_prepared_input_runtime runtime{}; REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
  struct Budget { mlx_original_buffer_budget value{}; ~Budget() { mlx_original_buffer_budget_release(value); } } budget;
  unsigned retired = 0;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value, runtime, 1 << 20, &retired,
      [](void* p) { ++*static_cast<unsigned*>(p); }) == 0);
  std::optional<array> failed_output;
  {
    Role role; REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()}, budget.value) == 0);
    Observer observer; Bank constructors;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&constructors.value, observer.value, 1, 0, 3) == 0);
    auto value = gather_mm(a, b, ai, bi, false, selected);
    cpu::CopyEvalStorage actual;
    // A source check validates metadata only. Actual index values belong to the
    // queued worker, after their own dependencies complete.
    REQUIRE(cpu::tiled_gather_mm_eval_storage(value, actual));
    constructors.reset(); Operation operation; operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1, 6, 2, 5, 2, 1, tiled_gather_mm_tests::captures(1)};
    const auto submitted = eval_traversal_tests::submit(operation, selected, limits);
    CHECK((submitted == 0 || submitted == static_cast<unsigned>(ScopedEvaluation::failed)));
    CHECK(mlx_operation_event_wait(operation.value) != 0);
    // A failed wait is not a completion proof. Settle observes the real pending
    // records before their retirement, preserving failure and buffer ownership.
    settle(role);
    REQUIRE(role.error.get()->borrow());
    CHECK(role.error.get()->borrow()->exception_type == &typeid(submission::GraphQuotaError));
    CHECK(role.records->occupied_bytes() == 0);
    CHECK(mlx_original_buffer_budget_occupied(budget.value) > 0);
    failed_output.emplace(value);
  }
  mlx_original_buffer_budget_release(budget.value); budget.value = {};
  CHECK(retired == 0); failed_output.reset(); CHECK(retired == 1);
}

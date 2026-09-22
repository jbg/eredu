// The grouped worker reduces a nonfinal routing axis. Retain that ordinary
// strided SIMD equation and price its existing one-axis plan directly.
TEST_CASE("CPU nonfinal F32 sum source checks compact axis geometry and refusal"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  std::array<float, 2 * 3 * 5> data;
  for (size_t i = 0; i < data.size(); ++i) data[i] = (int(i % 11) - 4) * .125f;
  array input(data.data(), Shape{2, 3, 5}, float32);
  auto value = sum(input, 1, true, stream);
  cpu::CopyEvalStorage cold, actual;
  REQUIRE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32StridedRows,
      3, 3, 10, false, cold));
  REQUIRE(cpu::reduction_eval_storage(value, actual));
  CHECK(actual.allocation_extents == cold.allocation_extents);
  CHECK(actual.named_control_bytes == cold.named_control_bytes);
  CHECK(actual.backing_births == 1); CHECK(actual.worker_graph_extents == 0);
  const auto saved = actual;
  for (size_t rank : {size_t(0), size_t(1), size_t(5)})
    CHECK_FALSE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32StridedRows,
        rank, 3, 10, false, actual));
  for (size_t width : {size_t(0), size_t(1), SIZE_MAX, size_t(INT_MAX)})
    CHECK_FALSE(cpu::reduction_eval_layout(cpu::ReductionEvalKind::Float32StridedRows,
        3, width, 10, false, actual));
  auto wrong_shape = array(Shape{1, 3, 5}, float32,
      std::make_shared<Reduce>(stream, Reduce::Sum, std::vector<int>{1}), {input});
  auto wrong_axes = array(Shape{2, 1, 5}, float32,
      std::make_shared<Reduce>(stream, Reduce::Sum, std::vector<int>{1, 1}), {input});
  auto sparse = slice(input, {0, 0, 0}, {2, 3, 5}, {1, 1, 2}, stream); eval(sparse);
  auto sparse_sum = sum(sparse, 1, true, stream);
  auto half = astype(input, float16, stream); eval(half);
  auto half_sum = sum(half, 1, true, stream);
  auto unreadable = array(Shape{2, 3, 5}, float32, nullptr, {});
  unreadable.copy_shared_buffer(array(1.f));
  auto unreadable_sum = sum(unreadable, 1, true, stream);
  for (const auto* invalid : {&wrong_shape, &wrong_axes, &sparse_sum, &half_sum, &unreadable_sum})
    CHECK_FALSE(cpu::reduction_eval_storage(*invalid, actual));
  CHECK(std::memcmp(&actual, &saved, sizeof(actual)) == 0);
}

TEST_CASE("CPU nonfinal F32 sums preserve numerical order and escaped custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  struct Geometry { Shape shape; int axis; };
  for (const auto& geometry : {Geometry{{3, 5}, 0}, Geometry{{2, 3, 17}, 1},
      Geometry{{2, 2, 3, 5}, 2}, Geometry{{2, 3, 1}, 1}}) {
    CAPTURE(geometry.shape); CAPTURE(geometry.axis);
    size_t outer = 1, inner = 1;
    for (int axis = 0; axis < geometry.axis; ++axis) outer *= geometry.shape[axis];
    for (size_t axis = geometry.axis + 1; axis < geometry.shape.size(); ++axis) inner *= geometry.shape[axis];
    const size_t width = geometry.shape[geometry.axis];
    std::vector<float> data(outer * width * inner);
    for (size_t i = 0; i < data.size(); ++i) data[i] = (int(i % 13) - 4) * .125f;
    array input(data.data(), geometry.shape, float32);
    auto ordinary = sum(input, geometry.axis, false, stream); eval(ordinary);
    mlx_prepared_input_runtime runtime{}; REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
    struct Budget { mlx_original_buffer_budget value{}; ~Budget() { mlx_original_buffer_budget_release(value); } } budget;
    unsigned retired = 0;
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value, runtime, 1 << 20, &retired,
        [](void* p) { ++*static_cast<unsigned*>(p); }, nullptr) == 0);
    std::optional<array> escaped;
    {
      Role role; REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()}, budget.value) == 0);
      Observer observer; Bank bank;
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value, observer.value, 2, 0, geometry.shape.size()) == 0);
      auto value = sum(input, geometry.axis, false, stream);
      bank.reset(); Operation operation; operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1, 5, 4, 4, 4, 1, 16};
      REQUIRE(eval_traversal_tests::submit(operation, stream, limits) == 0);
      eval_traversal_tests::complete(role, operation, value);
      for (size_t row = 0; row < outer; ++row) for (size_t column = 0; column < inner; ++column) {
        float expected = 0;
        for (size_t term = 0; term < width; ++term) expected += data[(row * width + term) * inner + column];
        const size_t index = row * inner + column;
        CHECK(value.data<float>()[index] == expected);
        CHECK(value.data<float>()[index] == ordinary.data<float>()[index]);
      }
      mlx_original_buffer_info info{};
      REQUIRE(mlx_original_buffer_array_info(&info, {&value}, budget.value) == 0);
      CHECK(info.known); escaped.emplace(value);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value) > 0);
    mlx_original_buffer_budget_release(budget.value); budget.value = {};
    CHECK(retired == 0); escaped.reset(); CHECK(retired == 1);
  }
}

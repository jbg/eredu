// Included after the original graph fixtures. GQA repeats physical KV storage
// before reshaping; its readable span can be smaller than its logical size.
TEST_CASE("CPU broadcast reshape uses exact physical span for alias and copy sources"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  for (int heads : {1, 2}) {
    CAPTURE(heads);
    constexpr int repeats = 3, positions = 2, width = 4;
    std::array<float, 2 * positions * width> data;
    for (size_t i = 0; i < data.size(); ++i) data[i] = (int(i % 13) - 6) * .125f;
    array source(data.data(), Shape{1, heads, positions, width}, float32);
    auto input = broadcast_to(reshape(source, {1, heads, 1, positions, width}, stream),
        {1, heads, repeats, positions, width}, stream);
    eval(input);
    REQUIRE(input.data_size() < input.size());
    const Shape shape{1, heads * repeats, positions, width};
    auto value = reshape(input, shape, stream);
    cpu::CopyEvalStorage cold, actual;
    REQUIRE(cpu::reshape_eval_layout(input.shape().data(), input.strides().data(), 5,
        shape.data(), 4, false, cold));
    REQUIRE(cpu::reshape_alias_eval_storage(value, actual));
    CHECK(actual.backing_births == size_t(heads != 1));
    CHECK(actual.backing_births == cold.backing_births);
    CHECK(actual.allocation_extents == cold.allocation_extents);
    CHECK(actual.named_control_bytes == cold.named_control_bytes);
    CHECK(actual.worker_graph_extents == cold.worker_graph_extents);
    const auto saved = actual;
    auto malformed = array(input.shape(), float32, nullptr, {});
    malformed.copy_shared_buffer(array(1.f));
    auto unreadable = reshape(malformed, shape, stream);
    CHECK_FALSE(cpu::reshape_alias_eval_storage(unreadable, actual));
    CHECK(std::memcmp(&actual, &saved, sizeof(actual)) == 0);
    const int bad_shape[]{1, heads * repeats, positions, width + 1};
    CHECK_FALSE(cpu::reshape_eval_layout(input.shape().data(), input.strides().data(), 5,
        bad_shape, 4, false, actual));
    CHECK(std::memcmp(&actual, &saved, sizeof(actual)) == 0);
  }
}

TEST_CASE("CPU GQA broadcast reshapes preserve ordinary values and original source custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  for (int heads : {1, 2}) {
    CAPTURE(heads);
    constexpr int repeats = 3, positions = 2, width = 4;
    std::array<float, 2 * positions * width * 2> data;
    for (size_t i = 0; i < data.size(); ++i) data[i] = (int(i % 17) - 8) * .125f;
    array base(data.data(), Shape{1, heads, positions, width * 2}, float32);
    auto strided = slice(base, {0, 0, 0, 0}, {1, heads, positions, width * 2}, {1, 1, 1, 2}, stream);
    eval(strided);
    mlx_prepared_input_runtime runtime{}; REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
    struct Budget { mlx_original_buffer_budget value{}; ~Budget() { mlx_original_buffer_budget_release(value); } } budget;
    unsigned retired = 0;
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value, runtime, 1 << 20, &retired,
        [](void* p) { ++*static_cast<unsigned*>(p); }, nullptr) == 0);
    std::optional<array> source, escaped;
    uint64_t identity = 0;
    {
      Role role; REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()}, budget.value) == 0);
      Observer observer; Bank bank;
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value, observer.value, 1, 0, 4) == 0);
      auto value = contiguous(strided, false, stream); bank.reset(); Operation operation; operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1, 3, 2, 2, 2, 1, 8};
      REQUIRE(eval_traversal_tests::submit(operation, stream, limits) == 0);
      eval_traversal_tests::complete(role, operation, value);
      mlx_original_buffer_info info{};
      REQUIRE(mlx_original_buffer_array_info(&info, {&value}, budget.value) == 0);
      REQUIRE(info.known); identity = info.identity; source.emplace(value);
    }
    {
      const Shape shape{1, heads * repeats, positions, width};
      auto input = broadcast_to(reshape(*source, {1, heads, 1, positions, width}, stream),
          {1, heads, repeats, positions, width}, stream);
      eval(input);
      auto ordinary = reshape(input, shape, stream); eval(ordinary);
      Role role; REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()}, budget.value) == 0);
      Observer observer; Bank bank;
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value, observer.value, 1, 0, 5) == 0);
      auto value = reshape(input, shape, stream); bank.reset(); Operation operation; operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1, 3, 2, 2, 2, 1, 8};
      REQUIRE(eval_traversal_tests::submit(operation, stream, limits) == 0);
      eval_traversal_tests::complete(role, operation, value);
      for (int h = 0; h < heads * repeats; ++h) for (int p = 0; p < positions; ++p) for (int d = 0; d < width; ++d) {
        const size_t actual = h * value.strides(1) + p * value.strides(2) + d * value.strides(3);
        const size_t reference = h * ordinary.strides(1) + p * ordinary.strides(2) + d * ordinary.strides(3);
        const size_t expected = ((h / repeats) * positions * width + p * width + d) * 2;
        CHECK(value.data<float>()[actual] == data[expected]);
        CHECK(value.data<float>()[actual] == ordinary.data<float>()[reference]);
      }
      mlx_original_buffer_info info{};
      REQUIRE(mlx_original_buffer_array_info(&info, {&value}, budget.value) == 0);
      REQUIRE(info.known); CHECK((info.identity != identity) == (heads != 1)); escaped.emplace(value);
    }
    source.reset(); CHECK(mlx_original_buffer_budget_occupied(budget.value) > 0);
    mlx_original_buffer_budget_release(budget.value); budget.value = {};
    CHECK(retired == 0); escaped.reset(); CHECK(retired == 1);
    CHECK(base.data<float>()[0] == data[0]);
  }
}

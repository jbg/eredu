// The same real alias workers handle empty shapes. They neither allocate Data
// nor discard an existing physical owner merely because its view has zero rows.
TEST_CASE("CPU empty reshape and squeeze use exact alias sources and ordinary completion"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  for (auto dtype : {float32, float16, bfloat16, int32}) {
    CAPTURE(dtype);
    auto input = full({0, 8}, array(0, dtype), dtype, stream);
    eval(input); input.eval();
    auto reshaped = reshape(input, {0, 1, 8}, stream);
    cpu::CopyEvalStorage cold, actual;
    REQUIRE(cpu::reshape_alias_eval_layout(2, 3, false, cold));
    REQUIRE(cpu::reshape_alias_eval_storage(reshaped, actual));
    CHECK(actual.named_control_bytes == cold.named_control_bytes);
    CHECK(actual.allocation_extents == cold.allocation_extents);
    CHECK(actual.backing_births == 0);
    CHECK(actual.request_counts[3] == 0);
    CHECK(actual.request_counts[6] == 0);
    eval(reshaped); reshaped.eval();
    auto squeezed = squeeze(reshaped, {1}, stream);
    REQUIRE(cpu::squeeze_eval_layout(3, false, cold));
    REQUIRE(cpu::greedy_eval_storage(squeezed, actual));
    CHECK(actual.named_control_bytes == cold.named_control_bytes);
    CHECK(actual.allocation_extents == cold.allocation_extents);
    CHECK(actual.backing_births == 0);
    CHECK(actual.request_counts[3] == 0);
    CHECK(actual.request_counts[6] == 0);
    eval(squeezed);
    CHECK(squeezed.shape() == input.shape());
    CHECK(squeezed.dtype() == dtype);
    CHECK(squeezed.size() == 0);
    CHECK(squeezed.data_size() == input.data_size());
    const auto saved = actual;
    auto false_reshape = array({1, 8}, dtype, std::make_shared<Reshape>(stream, Shape{1, 8}), {input});
    CHECK_FALSE(cpu::reshape_alias_eval_storage(false_reshape, actual));
    auto ambiguous = array({0, 0}, dtype, std::make_shared<Reshape>(stream, Shape{-1, 0}), {input});
    CHECK_FALSE(cpu::reshape_alias_eval_storage(ambiguous, actual));
    auto false_squeeze = array({1, 8}, dtype, std::make_shared<Squeeze>(stream, std::vector<int>{0}), {reshaped});
    CHECK_FALSE(cpu::greedy_eval_storage(false_squeeze, actual));
    CHECK(std::memcmp(&actual, &saved, sizeof(actual)) == 0);
    auto inferred = reshape(input, {-1, 8}, stream);
    REQUIRE(cpu::reshape_alias_eval_storage(inferred, actual));
    // Execute the same two empty aliases in a real Original role.
    mlx_prepared_input_runtime runtime{}; REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
    struct Budget { mlx_original_buffer_budget value{}; ~Budget() { mlx_original_buffer_budget_release(value); } } budget;
    unsigned retired = 0;
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value, runtime, 1 << 20, &retired,
        [](void* p) { ++*static_cast<unsigned*>(p); }) == 0);
    {
      Role role; REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()}, budget.value) == 0);
      Observer observer; Bank bank;
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value, observer.value, 2, 0, 3) == 0);
      auto value = squeeze(reshape(input, {0, 1, 8}, stream), {1}, stream);
      bank.reset(); Operation operation; operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1, 4, 3, 3, 3, 1, 8};
      REQUIRE(eval_traversal_tests::submit(operation, stream, limits) == 0);
      eval_traversal_tests::complete(role, operation, value);
      CHECK(value.shape() == input.shape()); CHECK(value.dtype() == dtype);
      CHECK(value.size() == 0); CHECK(value.data_size() == 0);
      CHECK(mlx_original_buffer_budget_occupied(budget.value) == 0);
    }
    mlx_original_buffer_budget_release(budget.value); budget.value = {};
    CHECK(retired == 1);
  }
}

TEST_CASE("CPU empty structural aliases retain the original nonempty physical backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  std::array<float, 16> data;
  for (size_t i = 0; i < data.size(); ++i) data[i] = float(i) - 7.25f;
  array base(data.data(), Shape{16}, float32);
  auto strided = slice(base, {0}, {16}, {2}, stream); eval(strided); strided.eval();
  mlx_prepared_input_runtime runtime{}; REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
  struct Budget { mlx_original_buffer_budget value{}; ~Budget() { mlx_original_buffer_budget_release(value); } } budget;
  unsigned retired = 0;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value, runtime, 1 << 20, &retired,
      [](void* p) { ++*static_cast<unsigned*>(p); }) == 0);
  std::optional<array> source, escaped;
  uint64_t identity = 0, capacity = 0;
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
    REQUIRE(info.known); identity = info.identity; capacity = mlx_original_buffer_budget_occupied(budget.value);
    REQUIRE(capacity > 0); source.emplace(value);
  }
  {
    // An ordinary zero-element AsStrided view still shares the real source Data.
    // Its logical data span is zero; allocation identity proves retained custody.
    auto input = as_strided(*source, {0, 8}, {0, 1}, 0, stream); eval(input); input.eval();
    REQUIRE(input.size() == 0); REQUIRE(input.data_size() == 0);
    mlx_original_buffer_info input_info{};
    REQUIRE(mlx_original_buffer_array_info(&input_info, {&input}, budget.value) == 0);
    REQUIRE(input_info.known); REQUIRE(input_info.identity == identity);
    Role role; REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()}, budget.value) == 0);
    Observer observer; Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value, observer.value, 2, 0, 3) == 0);
    auto value = squeeze(reshape(input, {0, 1, 8}, stream), {1}, stream);
    bank.reset(); Operation operation; operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1, 4, 3, 3, 3, 1, 8};
    REQUIRE(eval_traversal_tests::submit(operation, stream, limits) == 0);
    eval_traversal_tests::complete(role, operation, value);
    CHECK(value.size() == 0); CHECK(value.data_size() == input.data_size());
    mlx_original_buffer_info info{};
    REQUIRE(mlx_original_buffer_array_info(&info, {&value}, budget.value) == 0);
    REQUIRE(info.known); CHECK(info.identity == identity);
    CHECK(mlx_original_buffer_budget_occupied(budget.value) == capacity);
    escaped.emplace(value);
  }
  source.reset(); CHECK(mlx_original_buffer_budget_occupied(budget.value) == capacity);
  mlx_original_buffer_budget_release(budget.value); budget.value = {};
  CHECK(retired == 0); escaped.reset(); CHECK(retired == 1);
  CHECK(base.data<float>()[0] == data[0]);
}

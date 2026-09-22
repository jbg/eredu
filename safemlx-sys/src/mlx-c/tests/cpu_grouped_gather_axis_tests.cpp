// Exercise the existing GatherAxis worker at the routed sequential reduction
// geometry, including broadcast indices and completion-owned failure storage.
TEST_CASE("CPU grouped GatherAxis authenticates readable sources and broadcast index spans"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  std::array<float, 30> data{};
  const uint32_t ids[]{2, 0, 1, 1, 2, 0};
  array source(data.data(), {2, 3, 5}, float32), narrow(ids, {2, 3, 1}, uint32);
  auto indices = broadcast_to(narrow, source.shape(), stream); eval(indices);
  auto value = take_along_axis(source, indices, 1, stream);
  cpu::CopyEvalStorage cold, actual;
  REQUIRE(cpu::gather_axis_row_eval_layout(3, 30, false, cold));
  REQUIRE(cpu::gather_axis_row_eval_storage(value, actual));
  CHECK(actual.allocation_extents == cold.allocation_extents);
  CHECK(actual.named_control_bytes == cold.named_control_bytes);
  CHECK(actual.backing_births == 1);
  const auto prior = actual;
  auto wrong_shape = array({2, 1, 5}, float32,
      std::make_shared<GatherAxis>(stream, 1), {source, indices});
  auto wrong_axis = array(source.shape(), float32,
      std::make_shared<GatherAxis>(stream, 3), {source, indices});
  auto short_source = array(source.shape(), float32, nullptr, {});
  short_source.copy_shared_buffer(array(1.f));
  auto unreadable_source = take_along_axis(short_source, indices, 1, stream);
  auto short_index = array(source.shape(), uint32, nullptr, {});
  short_index.copy_shared_buffer(array(uint32_t(0)));
  auto unreadable_index = take_along_axis(source, short_index, 1, stream);
  // Reverse a nonbroadcast dimension, whose actual index stride is negative.
  auto negative_index = slice(indices, {0, 2, 0}, {2, -4, 5}, {1, -1, 1}, stream);
  eval(negative_index);
  auto negative = take_along_axis(source, negative_index, 1, stream);
  auto sparse = slice(source, {0, 0, 0}, {2, 3, 5}, {1, 1, 2}, stream); eval(sparse);
  auto sparse_indices = broadcast_to(narrow, sparse.shape(), stream); eval(sparse_indices);
  auto sparse_value = take_along_axis(sparse, sparse_indices, 1, stream);
  cpu::CopyEvalStorage sparse_cold,sparse_actual;
  REQUIRE(cpu::gather_axis_row_eval_layout(3,sparse.size(),false,sparse_cold));
  REQUIRE(cpu::gather_axis_row_eval_storage(sparse_value,sparse_actual));
  CHECK(sparse_actual.allocation_extents==sparse_cold.allocation_extents);
  CHECK(sparse_actual.backing_births==1);
  for (const auto* invalid : {&wrong_shape, &wrong_axis, &unreadable_source,
      &unreadable_index, &negative})
    CHECK_FALSE(cpu::gather_axis_row_eval_storage(*invalid, actual));
  CHECK(std::memcmp(&prior, &actual, sizeof(actual)) == 0);
}

TEST_CASE("CPU grouped GatherAxis preserves nonfinal ordering and escaped original custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  for (int width : {1, 17}) {
    constexpr int rows = 2, picks = 3;
    std::vector<float> data(rows * picks * width);
    for (size_t i = 0; i < data.size(); ++i) data[i] = (int(i % 19) - 7) * .125f;
    const uint32_t ids[]{2, 0, 1, 1, 1, 0};
    array source(data.data(), {rows, picks, width}, float32), narrow(ids, {rows, picks, 1}, uint32);
    auto indices = broadcast_to(narrow, source.shape(), stream); eval(indices);
    auto ordinary = take_along_axis(source, indices, 1, stream); eval(ordinary);
    mlx_prepared_input_runtime runtime{}; REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
    struct Budget { mlx_original_buffer_budget value{}; ~Budget() { mlx_original_buffer_budget_release(value); } } budget;
    unsigned retired = 0;
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value, runtime, 1 << 20, &retired,
        [](void* p) { ++*static_cast<unsigned*>(p); }, nullptr) == 0);
    std::optional<array> escaped;
    {
      Role role; REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()}, budget.value) == 0);
      Observer observer; Bank bank;
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value, observer.value, 1, 0, 3) == 0);
      auto value = take_along_axis(source, indices, 1, stream);
      bank.reset(); Operation operation; operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1, 4, 2, 3, 2, 1, 8};
      REQUIRE(eval_traversal_tests::submit(operation, stream, limits) == 0);
      eval_traversal_tests::complete(role, operation, value);
      for (int row = 0; row < rows; ++row) for (int pick = 0; pick < picks; ++pick)
        for (int column = 0; column < width; ++column) {
          const size_t destination = (row * picks + pick) * width + column;
          const float expected = data[(row * picks + ids[row * picks + pick]) * width + column];
          CHECK(value.data<float>()[destination] == expected);
          CHECK(value.data<float>()[destination] == ordinary.data<float>()[destination]);
        }
      mlx_original_buffer_info info{};
      REQUIRE(mlx_original_buffer_array_info(&info, {&value}, budget.value) == 0);
      CHECK(info.known); escaped.emplace(value);
    }
    mlx_original_buffer_budget_release(budget.value); budget.value = {};
    CHECK(retired == 0); escaped.reset(); CHECK(retired == 1);
  }
}

TEST_CASE("CPU grouped GatherAxis checks queued index values and retains failed output custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  const float data[]{1, 2, 3, 4}; const int32_t ids[]{0, 2, 1, 0};
  array source(data, {1, 2, 2}, float32), signed_indices(ids, {1, 2, 2}, int32);
  mlx_prepared_input_runtime runtime{}; REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
  struct Budget { mlx_original_buffer_budget value{}; ~Budget() { mlx_original_buffer_budget_release(value); } } budget;
  unsigned retired = 0;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value, runtime, 1 << 20, &retired,
      [](void* p) { ++*static_cast<unsigned*>(p); }, nullptr) == 0);
  std::optional<array> failed_output;
  {
    Role role; REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()}, budget.value) == 0);
    Observer observer; Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value, observer.value, 2, 0, 3) == 0);
    auto indices = astype(signed_indices, uint32, stream);
    auto value = take_along_axis(source, indices, 1, stream);
    bank.reset(); Operation operation; operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1, 5, 3, 4, 3, 1, 8};
    const auto submitted = eval_traversal_tests::submit(operation, stream, limits);
    CHECK((submitted == 0 || submitted == static_cast<unsigned>(ScopedEvaluation::failed)));
    CHECK(mlx_operation_event_wait(operation.value) != 0);
    // A wait failure alone cannot release the native submission authority.
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

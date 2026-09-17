// Included by operation_observer_tests.cpp: shares its actual configured Owner.
void original_prepared_clone_handles(Device::DeviceType device) {
  auto stream = default_stream(device);
  mlx_submission_runtime_baseline runtime{};
  REQUIRE(mlx_submission_prepare_runtime(&runtime, mlx_stream{&stream}, mlx_stream{&stream}) == 0);
  array input({2.0f, -3.0f, 7.0f});
  eval(input);
  void* storage = nullptr;
  void* unused = nullptr;
  REQUIRE(mlx_array_clone_storage_new(&storage) == 0);
  REQUIRE(mlx_array_clone_storage_new(&unused) == 0);
  auto* allocation = storage;
  Owner owner;
  mlx_submission_observer observer{};
  REQUIRE(mlx_submission_observer_current(&observer) == 0);
  const auto graph_start = owner.graph->occupied_bytes();
  const auto record_start = owner.records->occupied_bytes();
  {
    auto source = add(input, input, stream);
    const auto source_id = source.id();
    mlx_array result{}, refused{};
    CHECK(mlx_array_clone_storage_fill(&refused, &storage, mlx_array{}) == 4);
    CHECK(storage == allocation);
    CHECK(refused.ctx == nullptr);
    void* absent = nullptr;
    CHECK(mlx_array_clone_storage_fill(&refused, &absent, mlx_array{&source}) == 4);
    CHECK(refused.ctx == nullptr);
    REQUIRE(mlx_array_clone_storage_fill(&result, &storage, mlx_array{&source}) == 0);
    CHECK(storage == nullptr);
    CHECK(result.ctx == allocation);
    CHECK(result.prepared_owner == nullptr);
    CHECK(mlx_array_get_(result).id() == source_id);
    CHECK(mlx_array_clone_storage_fill(&refused, &storage, mlx_array{&source}) == 4);
    CHECK(refused.ctx == nullptr);
    CHECK(source.id() == source_id);
    mlx_operation_event event{};
    REQUIRE(mlx_operation_event_new_exact(&event, observer, mlx_stream{&stream}, 1) == 0);
    REQUIRE(mlx_operation_event_append(event, result) == 0);
    REQUIRE(mlx_operation_event_submit_on_stream(event, mlx_stream{&stream}) == 0);
    REQUIRE(mlx_operation_event_wait(event) == 0);
    await_original_terminal(owner);
    REQUIRE(mlx_submission_observer_validate_array(observer, result) == 0);
    const float expected[] = {4.0f, -6.0f, 14.0f};
    for (unsigned i = 0; i != 3; ++i) CHECK(mlx_array_get_(result).data<float>()[i] == expected[i]);
    // Exactly the ordinary destructor/free of the original cold allocation.
    REQUIRE(mlx_array_free(result) == 0);
    CHECK(source.id() == source_id);
    mlx_operation_event_free(event);
  }
  mlx_array_clone_storage_free(storage); // consumed null, no array destructor
  mlx_array_clone_storage_free(unused); // never filled, no descriptor exists
  await_original_terminal(owner);
  REQUIRE(mlx_submission_observer_retire(observer) == 0);
  CHECK(owner.graph->occupied_bytes() == graph_start);
  CHECK(owner.records->occupied_bytes() == record_start);
  CHECK(owner.scope->active_on_owner_thread());
  mlx_submission_observer_release(observer);
}

TEST_CASE("prepared clone handles preserve CPU source identity refusal and original completion") {
  original_prepared_clone_handles(Device::cpu);
}
#ifdef MLX_C_PATCH_TEST_METAL
TEST_CASE("prepared clone handles preserve Metal source identity refusal and original completion") {
  original_prepared_clone_handles(Device::gpu);
}
#endif

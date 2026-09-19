// Uses the actual unary producer and the shared original-scope fixtures.
TEST_CASE("CPU quantization unary sources retain task dtype and geometry checks"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  for (auto dtype : {float16, bfloat16, float32, float64, int8, int16, int32, int64}) {
    for (auto kind : {cpu::UnaryEvalKind::absolute, cpu::UnaryEvalKind::round}) {
      CAPTURE(dtype); CAPTURE(int(kind));
      const bool supported = kind == cpu::UnaryEvalKind::absolute || issubdtype(dtype, inexact);
      cpu::UnaryEvalStorage cold{};
      if (!supported) {
        CHECK_FALSE(cpu::unary_eval_layout(kind, dtype, 2, false, cold));
        continue;
      }
      auto input = astype(array({-2.5f, -0.5f, 0.5f, 3.5f}), dtype, stream);
      input = reshape(input, {2, 2}, stream);
      eval(input); input.eval();
      auto value = kind == cpu::UnaryEvalKind::absolute ? abs(input, stream) : round(input, stream);
      REQUIRE(cpu::unary_eval_layout(kind, dtype, 2, false, cold));
      cpu::UnaryEvalStorage actual{};
      REQUIRE(cpu::unary_eval_storage(value, actual));
      CHECK(actual.kind == kind);
      CHECK(actual.dtype == dtype);
      CHECK(actual.allocation_extents == cold.allocation_extents);
      CHECK(actual.worker_graph_extents == cold.worker_graph_extents);
      CHECK(actual.backing_births == 1);
      CHECK(actual.named_control_bytes == cold.named_control_bytes);
      CHECK(actual.named_control_bytes > 0);
      cpu::UnaryEvalStorage high{};
      REQUIRE(cpu::unary_eval_layout(kind, dtype, 14, true, high));
      CHECK(high.worker_graph_extents > cold.worker_graph_extents);
      const auto saved = high;
      CHECK_FALSE(cpu::unary_eval_layout(kind, uint32, 1, false, high));
      CHECK_FALSE(cpu::unary_eval_layout(kind, bool_, 1, false, high));
      CHECK_FALSE(cpu::unary_eval_layout(kind, dtype, SIZE_MAX, false, high));
      CHECK(std::memcmp(&saved, &high, sizeof(high)) == 0);
      std::shared_ptr<Primitive> primitive = kind == cpu::UnaryEvalKind::absolute
          ? std::static_pointer_cast<Primitive>(std::make_shared<Abs>(stream))
          : std::static_pointer_cast<Primitive>(std::make_shared<Round>(stream));
      auto wrong_shape = array(Shape{4}, dtype, primitive, {input});
      auto wrong_dtype = array(Shape{2,2}, uint32, primitive, {input});
      auto wrong_inputs = array(Shape{2,2}, dtype, primitive, {input, input});
      CHECK_FALSE(cpu::unary_eval_storage(wrong_shape, high));
      CHECK_FALSE(cpu::unary_eval_storage(wrong_dtype, high));
      CHECK_FALSE(cpu::unary_eval_storage(wrong_inputs, high));
      CHECK(std::memcmp(&saved, &high, sizeof(high)) == 0);
    }
  }
  for (auto kind : {cpu::UnaryEvalKind::absolute, cpu::UnaryEvalKind::round}) {
    for (auto dtype : {MLX_FLOAT16, MLX_BFLOAT16, MLX_FLOAT32}) {
      mlx_cpu_unary_eval_layout raw{};
      REQUIRE(mlx_operation_event_cpu_unary_eval_layout(&raw, uint32_t(kind), dtype, 3, false));
      CHECK(raw.graph_extents > 0);
      CHECK(raw.named_control_bytes > 0);
      CHECK(raw.backing_births == 1);
      const auto saved = raw;
      CHECK_FALSE(mlx_operation_event_cpu_unary_eval_layout(&raw, uint32_t(kind), MLX_UINT32, 3, false));
      CHECK_FALSE(mlx_operation_event_cpu_unary_eval_layout(&raw, uint32_t(kind), dtype, SIZE_MAX, false));
      CHECK(std::memcmp(&saved, &raw, sizeof(raw)) == 0);
    }
  }
}

TEST_CASE("CPU original quantization unary workers preserve values and escaped custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  mlx_prepared_input_runtime runtime{};
  REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
  for (auto dtype : {float16, bfloat16, float32}) {
    for (auto kind : {cpu::UnaryEvalKind::absolute, cpu::UnaryEvalKind::round}) {
      for (int rank : {1, 2, 14}) {
        CAPTURE(dtype); CAPTURE(int(kind)); CAPTURE(rank);
        // Rank one takes the contiguous SIMD path and its scalar tail. Higher
        // ranks take the noncontiguous iterator, including dynamic shape rows.
        const float sequence[] = {-3.5f,-2.5f,-1.5f,-0.5f,-0.0f,0.0f,0.5f,1.5f,2.5f,3.5f,-6.25f,7.75f,0.25f};
        std::vector<float> base_values;
        for (float x : sequence) {
          base_values.push_back(x);
          if (rank > 1) base_values.push_back(99.0f);
        }
        Shape shape(rank, 1); shape.back() = int(base_values.size());
        auto source = astype(array(base_values.begin(), shape), dtype, stream);
        if (rank > 1) {
          Shape starts(rank, 0), steps(rank, 1); steps.back() = 2;
          source = slice(source, starts, shape, steps, stream);
        }
        eval(source); source.eval();
        REQUIRE(source.flags().contiguous == (rank == 1));
        unsigned retired = 0;
        struct Budget {
          mlx_original_buffer_budget value{};
          ~Budget() { mlx_original_buffer_budget_release(value); }
        } budget;
        mlx_original_buffer_population_layout physical{};
        REQUIRE(mlx_original_buffer_request_layout_for(&physical, runtime,
            std::size(sequence) * dtype.size()) == 0);
        REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value, runtime, physical.capacity,
            &retired, [](void* p) { ++*static_cast<unsigned*>(p); }) == 0);
        std::optional<array> escaped;
        {
          Role role;
          REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()}, budget.value) == 0);
          Observer observer;
          Bank bank;
          REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value, observer.value, 1, 0, rank) == 0);
          auto value = kind == cpu::UnaryEvalKind::absolute ? abs(source, stream) : round(source, stream);
          bank.reset();
          Operation operation; operation.append(value);
          const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
          REQUIRE(eval_traversal_tests::submit(operation, stream, limits) == 0);
          eval_traversal_tests::complete(role, operation, value);
          CHECK(value.dtype() == dtype);
          CHECK(value.flags().row_contiguous);
          mlx_original_buffer_info info{};
          REQUIRE(mlx_original_buffer_array_info(&info, {&value}, budget.value) == 0);
          CHECK(info.known);
          CHECK(info.identity != 0);
          CHECK(info.charged_bytes >= std::size(sequence) * dtype.size());
          CHECK(info.charged_bytes <= physical.capacity);
          escaped.emplace(value);
        }
        REQUIRE(escaped.has_value());
        CHECK(mlx_original_buffer_budget_occupied(budget.value) > 0);
        CHECK(retired == 0);
        for (size_t i = 0; i < std::size(sequence); ++i) {
          const float expected = kind == cpu::UnaryEvalKind::absolute ? std::abs(sequence[i]) : std::nearbyint(sequence[i]);
          const float actual = dtype == float32 ? escaped->data<float>()[i] : dtype == float16
              ? float(escaped->data<float16_t>()[i]) : float(escaped->data<mlx::core::bfloat16_t>()[i]);
          CHECK(actual == expected);
          if (expected == 0) CHECK(std::signbit(actual) == std::signbit(expected));
        }
        mlx_original_buffer_budget_release(budget.value); budget.value = {};
        CHECK(retired == 0); // The escaped output retains the physical account.
        escaped.reset();
        CHECK(retired == 1);
      }
    }
  }
}

TEST_CASE("CPU quantization unary refuses a short physical budget and retires its prefix"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu);
  prepare(stream, stream);
  mlx_prepared_input_runtime runtime{};
  REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
  for (auto dtype : {float16, bfloat16, float32}) {
    for (auto kind : {cpu::UnaryEvalKind::absolute, cpu::UnaryEvalKind::round}) {
      CAPTURE(dtype); CAPTURE(int(kind));
      auto source = astype(array({-3.5f, -0.5f, 0.5f, 3.5f}), dtype, stream);
      eval(source); source.eval();
      mlx_original_buffer_population_layout physical{};
      REQUIRE(mlx_original_buffer_request_layout_for(&physical, runtime, source.nbytes()) == 0);
      REQUIRE(physical.capacity > 0);
      unsigned retired = 0;
      struct Budget {
        mlx_original_buffer_budget value{};
        ~Budget() { mlx_original_buffer_budget_release(value); }
      } budget;
      REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value, runtime, physical.capacity - 1,
          &retired, [](void* p) { ++*static_cast<unsigned*>(p); }) == 0);
      {
        Role role;
        REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()}, budget.value) == 0);
        Observer observer;
        Bank bank;
        REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value, observer.value, 1, 0, 1) == 0);
        auto value = kind == cpu::UnaryEvalKind::absolute ? abs(source, stream) : round(source, stream);
        bank.reset();
        Operation operation; operation.append(value);
        const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
        REQUIRE(eval_traversal_tests::submit(operation, stream, limits) ==
            static_cast<unsigned>(ScopedEvaluation::failed));
        settle(role);
        CHECK(role.scope->query_records().pending == 0);
        CHECK(role.records->occupied_bytes() == 0);
        REQUIRE(role.error.get()->borrow());
        try {
          std::rethrow_exception(role.error.get()->borrow()->exception);
          FAIL("physical refusal must retain its typed cause");
        } catch (const allocator::OriginalBufferError& error) {
          CHECK(error.cause() == allocator::OriginalBufferCause::capacity);
        }
        CHECK(mlx_original_buffer_budget_occupied(budget.value) == 0);
        CHECK(retired == 0);
      }
      mlx_original_buffer_budget_release(budget.value); budget.value = {};
      CHECK(retired == 1);
    }
  }
}

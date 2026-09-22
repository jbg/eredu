namespace quantization_source_tests {
using namespace pointwise_graph_tests;

// A single real producer must fit its exact physical request, and the escaped
// array must retain that account after its original scope and handle retire.
template <typename Make, typename Check>
void exercise(Stream stream, size_t bytes, size_t rank, Make make, Check check,
    bool short_budget = false) {
  mlx_prepared_input_runtime runtime{};
  REQUIRE(mlx_prepared_input_runtime_prepare(&runtime) == 0);
  mlx_original_buffer_population_layout physical{};
  REQUIRE(mlx_original_buffer_request_layout_for(&physical, runtime, bytes) == 0);
  REQUIRE(physical.capacity > 0);
  unsigned retired = 0;
  struct Budget {
    mlx_original_buffer_budget value{};
    ~Budget() { mlx_original_buffer_budget_release(value); }
  } budget;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value, runtime,
      physical.capacity - size_t(short_budget), &retired,
      [](void* p) { ++*static_cast<unsigned*>(p); }, nullptr) == 0);
  std::optional<array> escaped;
  {
    Role role;
    REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()}, budget.value) == 0);
    Observer observer;
    Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value, observer.value, 1, 0, rank) == 0);
    auto value = make();
    bank.reset();
    Operation operation; operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
    if (short_budget) {
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
    } else {
      REQUIRE(eval_traversal_tests::submit(operation, stream, limits) == 0);
      eval_traversal_tests::complete(role, operation, value);
      mlx_original_buffer_info info{};
      REQUIRE(mlx_original_buffer_array_info(&info, {&value}, budget.value) == 0);
      CHECK(info.known); CHECK(info.identity != 0);
      CHECK(info.charged_bytes >= bytes);
      CHECK(info.charged_bytes <= physical.capacity);
      escaped.emplace(value);
    }
  }
  CHECK(retired == 0);
  if (!short_budget) {
    REQUIRE(escaped.has_value());
    CHECK(mlx_original_buffer_budget_occupied(budget.value) > 0);
    check(*escaped);
  }
  mlx_original_buffer_budget_release(budget.value); budget.value = {};
  if (escaped) {
    CHECK(retired == 0);
    escaped.reset();
  }
  CHECK(retired == 1);
}
} // namespace quantization_source_tests

TEST_CASE("CPU quantization integer range checks integral steps and final increment"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  const std::array<std::array<double,3>,6> valid{{
      {{0,28,4}}, {{3,17,2}}, {{9,28,3}}, {{16777217,16777244,4}},
      {{double(INT_MAX)-28,double(INT_MAX),4}}, {{0,1,double(INT_MAX)}}}};
  for (auto dtype : {int32, uint32}) {
    for (auto endpoints : valid) {
      CAPTURE(dtype); CAPTURE(endpoints[0]); CAPTURE(endpoints[1]); CAPTURE(endpoints[2]);
      const size_t count = size_t(std::ceil((endpoints[1]-endpoints[0])/endpoints[2]));
      auto value = arange(endpoints[0], endpoints[1], endpoints[2], dtype, stream);
      cpu::CopyEvalStorage cold{}, actual{};
      REQUIRE(cpu::arange_int_eval_layout(dtype, count, false, cold));
      REQUIRE(cpu::arange_int_eval_storage(value, actual));
      CHECK(actual.allocation_extents == cold.allocation_extents);
      CHECK(actual.named_control_bytes == cold.named_control_bytes);
      CHECK(actual.inputs == 0); CHECK(actual.backing_births == 1);
      CHECK(actual.worker_graph_extents == 0);
      quantization_source_tests::exercise(stream, count*4, 1,
          [&] { return arange(endpoints[0], endpoints[1], endpoints[2], dtype, stream); },
          [&](const array& result) {
            CHECK(result.dtype() == dtype); CHECK(result.size() == count);
            for (size_t i=0; i<count; ++i) {
              const uint32_t expected = uint32_t(endpoints[0]) + uint32_t(i)*uint32_t(endpoints[2]);
              CHECK((dtype == int32 ? uint32_t(result.data<int32_t>()[i]) : result.data<uint32_t>()[i]) == expected);
            }
          });
    }
    cpu::CopyEvalStorage actual{};
    REQUIRE(cpu::arange_int_eval_layout(dtype, 7, false, actual));
    const auto saved = actual;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const std::array<std::array<double,3>,12> invalid{{
        {{-1,27,4}}, {{0.5,28.5,4}}, {{0,27.5,4}}, {{0,28,4.5}},
        {{0,28,0}}, {{28,0,-4}}, {{0,29,4}},
        {{double(INT_MAX)-27,double(INT_MAX),4}},
        {{nan,28,4}}, {{0,inf,4}}, {{0,28,nan}}, {{0,28,inf}}}};
    for (auto endpoints : invalid) {
      auto bad = array(Shape{7}, dtype,
          std::make_shared<Arange>(stream,endpoints[0],endpoints[1],endpoints[2]), {});
      CHECK_FALSE(cpu::arange_int_eval_storage(bad, actual));
      CHECK(std::memcmp(&saved, &actual, sizeof(actual)) == 0);
    }
    quantization_source_tests::exercise(stream, 7*4, 1,
        [&] { return arange(0.0,28.0,4.0,dtype,stream); }, [](const array&) {}, true);
  }
}

TEST_CASE("CPU quantization U32 row sum authenticates compact final-axis geometry"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  constexpr auto kind = cpu::ReductionEvalKind::Uint32Rows;
  for (size_t rank : {2,3,4}) {
    Shape shape(rank,1); shape[0]=3; shape.back()=7;
    std::vector<uint32_t> data(21,3);
    array input(data.data(), shape, uint32);
    auto value = sum(input, std::vector<int>{int(rank)-1}, true, stream);
    cpu::CopyEvalStorage cold{}, actual{};
    REQUIRE(cpu::reduction_eval_layout(kind,rank,7,3,false,cold));
    REQUIRE(cpu::reduction_eval_storage(value,actual));
    CHECK(actual.allocation_extents == cold.allocation_extents);
    CHECK(actual.worker_graph_extents == cold.worker_graph_extents);
    CHECK(actual.named_control_bytes == cold.named_control_bytes);
    CHECK(actual.backing_births == 1); CHECK(actual.inputs == 1);
    mlx_cpu_copy_eval_layout raw{};
    REQUIRE(mlx_operation_event_cpu_reduction_eval_layout(&raw, unsigned(kind),rank,7,3,false));
    CHECK(raw.graph_extents == cold.allocation_extents);
    CHECK(raw.named_control_bytes >= cold.named_control_bytes);
    const auto saved = actual;
    auto output_shape=shape; output_shape.back()=1;
    auto bad_axis=array(output_shape,uint32,std::make_shared<Reduce>(stream,Reduce::Sum,std::vector<int>{0}),{input});
    auto bad_shape=array(shape,uint32,std::make_shared<Reduce>(stream,Reduce::Sum,std::vector<int>{int(rank)-1}),{input});
    auto bad_dtype=array(output_shape,int32,std::make_shared<Reduce>(stream,Reduce::Sum,std::vector<int>{int(rank)-1}),{input});
    auto bad_axes=array(output_shape,uint32,std::make_shared<Reduce>(stream,Reduce::Sum,std::vector<int>{int(rank)-1,int(rank)-1}),{input});
    Shape starts(rank,0),steps(rank,1); steps.back()=2;
    auto sparse=slice(input,starts,shape,steps,stream); eval(sparse); sparse.eval();
    auto sparse_sum=sum(sparse,std::vector<int>{int(rank)-1},true,stream);
    for (const array* malformed : {&bad_axis,&bad_shape,&bad_dtype,&bad_axes,&sparse_sum}) {
      CHECK_FALSE(cpu::reduction_eval_storage(*malformed,actual));
      CHECK(std::memcmp(&saved,&actual,sizeof(actual)) == 0);
    }
    for (size_t width : {size_t(0),size_t(1),SIZE_MAX})
      CHECK_FALSE(cpu::reduction_eval_layout(kind,rank,width,3,false,actual));
    CHECK_FALSE(cpu::reduction_eval_layout(kind,rank,7,0,false,actual));
    CHECK(std::memcmp(&saved,&actual,sizeof(actual)) == 0);
  }
  for (size_t rank : {size_t(0),size_t(1),size_t(5),SIZE_MAX}) {
    cpu::CopyEvalStorage cold{};
    CHECK_FALSE(cpu::reduction_eval_layout(kind,rank,7,3,false,cold));
  }
}

TEST_CASE("CPU quantization U32 row sum preserves SIMD tails packed bits and escaped custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream = new_stream(Device::cpu); prepare(stream, stream);
  for (int rank : {2,3,4}) {
    for (int width : {2,7,8,33,1031}) {
      for (int rows : {1,6}) {
        CAPTURE(rank); CAPTURE(width); CAPTURE(rows);
        Shape shape(rank,1); shape[0]=rows; shape.back()=width;
        if (rank>2 && rows>1) { shape[0]=2; shape[1]=3; }
        std::vector<uint32_t> data(size_t(rows)*width);
        std::vector<uint32_t> expected(rows,0);
        for (int row=0; row<rows; ++row) {
          for (int i=0; i<width; ++i) {
            // Independent unsigned arithmetic checks the bit pattern, including
            // the high bit and wraparound, across SIMD blocks and scalar tails.
            const uint32_t x=uint32_t((i+row*3)%16) << ((i%8)*4);
            data[size_t(row)*width+i]=x; expected[row]+=x;
          }
        }
        array input(data.data(),shape,uint32);
        auto make=[&] { return sum(input,std::vector<int>{rank-1},true,stream); };
        quantization_source_tests::exercise(stream,size_t(rows)*4,rank,make,
            [&](const array& result) {
              CHECK(result.dtype() == uint32); CHECK(result.size() == size_t(rows));
              auto output_shape=shape; output_shape.back()=1;
              CHECK(result.shape() == output_shape);
              for (int row=0; row<rows; ++row) CHECK(result.data<uint32_t>()[row] == expected[row]);
            });
        if (width==8 && rows==6)
          quantization_source_tests::exercise(stream,size_t(rows)*4,rank,make,[](const array&) {},true);
      }
    }
  }
}

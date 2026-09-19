TEST_CASE("CPU MXFP4 payload inventory separates requests before allocator rounding") {
  using namespace pointwise_graph_tests;
  for(auto dtype:{float16,bfloat16,float32})for(size_t rows:{1u,2u,17u})for(size_t columns:{32u,64u,256u}) {
    const auto scalar=dtype==float16?MLX_FLOAT16:dtype==bfloat16?MLX_BFLOAT16:MLX_FLOAT32;
    submission::CpuMxFp4QuantizePayloadLayout native{};
    mlx_cpu_mxfp4_quantize_payload_layout raw{};
    REQUIRE(submission::cpu_mxfp4_quantize_payload_layout(dtype,rows,columns,native));
    REQUIRE(mlx_operation_event_cpu_mxfp4_quantize_payload_layout(&raw,scalar,rows,columns));
    CHECK(raw.request_count==(dtype==float32?28:29));CHECK(raw.named_control_bytes>=native.named_control_bytes);
    const size_t width=dtype.size(),row=columns*(35*width+8)+(columns/32)*(8*width+5);
    const size_t fixed=136+4*width+(width==2?16*width:0);
    CHECK(raw.temporary_row_bytes==row);CHECK(raw.temporary_fixed_bytes==fixed);
    CHECK(raw.output_bytes[0]==rows*columns/2);CHECK(raw.output_bytes[1]==rows*columns/32);
    size_t sum=0;for(size_t i=0;i<raw.request_count;++i){CHECK(raw.request_bytes[i]==native.request_bytes[i]);sum+=raw.request_bytes[i];}
    CHECK(sum==rows*row+fixed+rows*columns/2+rows*columns/32);
    const auto saved=raw;
    for(size_t bad:{size_t(0),size_t(31),size_t(33),SIZE_MAX})
      CHECK_FALSE(mlx_operation_event_cpu_mxfp4_quantize_payload_layout(&raw,scalar,rows,bad));
    CHECK_FALSE(mlx_operation_event_cpu_mxfp4_quantize_payload_layout(&raw,scalar,0,columns));
    CHECK_FALSE(mlx_operation_event_cpu_mxfp4_quantize_payload_layout(&raw,scalar,SIZE_MAX,columns));
    CHECK_FALSE(mlx_operation_event_cpu_mxfp4_quantize_payload_layout(&raw,MLX_INT32,rows,columns));
    CHECK_FALSE(mlx_operation_event_cpu_mxfp4_quantize_payload_layout(&raw,MLX_FLOAT64,rows,columns));
    CHECK(std::memcmp(&raw,&saved,sizeof(raw))==0);
  }
}

TEST_CASE("CPU MXFP4 payload capacities fund row and distance-table rounding") {
  for(auto dtype:{mlx::core::float16,mlx::core::bfloat16,mlx::core::float32})
    for(int variant:{0,1,2}) {
      mxfp4_composed_tests::exercise(dtype,2,1,32,variant);
      mxfp4_composed_tests::exercise(dtype,3,17,256,variant);
    }
}

TEST_CASE("CPU MXFP4 payload refusal retains a typed composed evaluation failure"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace affine_construction_tests;
  for(auto dtype:{float16,bfloat16,float32}) {
    auto stream=new_stream(Device::cpu);prepare(stream,stream);
    std::vector<float> data(17*256,1.f);
    auto input=astype(array(data.begin(),Shape{17,256}),dtype,stream);eval(input);input.eval();
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    const auto scalar=dtype==float16?MLX_FLOAT16:dtype==bfloat16?MLX_BFLOAT16:MLX_FLOAT32;
    mlx_cpu_mxfp4_quantize_payload_layout payload{};
    mlx_cpu_mxfp4_quantize_construction_layout construction{};
    REQUIRE(mlx_operation_event_cpu_mxfp4_quantize_payload_layout(&payload,scalar,17,256));
    REQUIRE(mlx_operation_event_cpu_mxfp4_quantize_construction_layout(&construction,scalar,2,17,256));
    size_t capacity=0;
    for(size_t i=0;i<6;++i) {
      mlx_original_buffer_population_layout physical{};
      REQUIRE(mlx_original_buffer_request_layout_for(&physical,runtime,payload.request_bytes[i])==0);
      capacity+=physical.capacity;
    }
    mlx_original_buffer_population_layout distance{};
    REQUIRE(mlx_original_buffer_request_layout_for(&distance,runtime,17*256*16*dtype.size())==0);
    REQUIRE(distance.capacity>capacity);
    unsigned retired=0;
    struct Budget {mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}} budget;
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,capacity,&retired,
        [](void* p){++*static_cast<unsigned*>(p);})==0);
    {
      Role role;Observer observer;Bank bank;Outputs outputs;
      REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,
          construction.graph.primitives,construction.graph.seeds,construction.graph.maximum_rank)==0);
      REQUIRE(mlx_quantize_fixed(&outputs.values[0],&outputs.values[1],&outputs.values[2],
          {&input},32,4,"mxfp4",{&stream})==0);
      CHECK(mlx_original_buffer_budget_occupied(budget.value)==capacity);
      bank.reset();Operation operation;
      operation.append(mlx_array_get_(outputs.values[0]));operation.append(mlx_array_get_(outputs.values[1]));
      CHECK(eval_traversal_tests::submit(operation,stream,{2,74,67,200,67,1,16})==static_cast<unsigned>(ScopedEvaluation::failed));
      settle(role);REQUIRE(role.error.get()->borrow());
      try {std::rethrow_exception(role.error.get()->borrow()->exception);FAIL("expected quantizer capacity refusal");}
      catch(const allocator::OriginalBufferError& error){CHECK(error.cause()==allocator::OriginalBufferCause::capacity);}
      for(size_t i=0;i<2;++i) {
        mlx_original_buffer_info info{};
        REQUIRE(mlx_original_buffer_array_info(&info,outputs.values[i],budget.value)==0);CHECK_FALSE(info.known);
      }
      CHECK(role.scope->query_records().pending==0);
      operation.reset();outputs.reset();
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==1);
  }
}

#include "mlx/quantize_construction.h"

TEST_CASE("CPU MXFP4 construction validates geometry and six independent seed requests"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  for(auto dtype:{float16,bfloat16,float32})for(size_t rank:{2u,3u,4u,11u,21u}) {
    const auto scalar=dtype==float16?MLX_FLOAT16:dtype==bfloat16?MLX_BFLOAT16:MLX_FLOAT32;
    submission::CpuMxFp4QuantizeConstructionLayout native{};
    mlx_cpu_mxfp4_quantize_construction_layout raw{};
    REQUIRE(submission::cpu_mxfp4_quantize_construction_layout(dtype,rank,2,64,native));
    REQUIRE(mlx_operation_event_cpu_mxfp4_quantize_construction_layout(&raw,scalar,rank,2,64));
    CHECK(raw.graph.allocation_extents==native.graph.allocation_extents);
    CHECK(raw.graph.maximum_rank==std::max(rank,size_t(3)));
    CHECK(raw.graph.seeds==6);CHECK(raw.named_control_bytes>=native.named_control_bytes);
    const size_t expected[]={dtype.size(),dtype.size(),dtype.size(),4,64,4};
    for(size_t i=0;i<6;++i){CHECK(raw.seed_request_bytes[i]==expected[i]);CHECK(native.seed_request_bytes[i]==expected[i]);}
    const auto prior=raw;
    for(size_t bad:{size_t(0),size_t(1),SIZE_MAX})
      CHECK_FALSE(mlx_operation_event_cpu_mxfp4_quantize_construction_layout(&raw,scalar,bad,2,64));
    for(size_t bad:{size_t(0),size_t(31),size_t(33),SIZE_MAX})
      CHECK_FALSE(mlx_operation_event_cpu_mxfp4_quantize_construction_layout(&raw,scalar,rank,2,bad));
    CHECK_FALSE(mlx_operation_event_cpu_mxfp4_quantize_construction_layout(&raw,scalar,rank,0,64));
    CHECK_FALSE(mlx_operation_event_cpu_mxfp4_quantize_construction_layout(&raw,scalar,rank,SIZE_MAX,64));
    CHECK_FALSE(mlx_operation_event_cpu_mxfp4_quantize_construction_layout(&raw,MLX_FLOAT64,rank,2,64));
    CHECK_FALSE(mlx_operation_event_cpu_mxfp4_quantize_construction_layout(&raw,MLX_INT32,rank,2,64));
    CHECK(std::memcmp(&raw,&prior,sizeof(raw))==0);
  }
}

TEST_CASE("CPU MXFP4 construction uses its shared bank and exact constant backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  for(auto dtype:{float16,bfloat16,float32})for(int rank:{2,3,4,11,21})for(bool strided:{false,true}) {
    CAPTURE(dtype);CAPTURE(rank);CAPTURE(strided);
    auto input=affine_converter_tests::input(stream,dtype,rank,32,4,strided);
    const auto scalar=dtype==float16?MLX_FLOAT16:dtype==bfloat16?MLX_BFLOAT16:MLX_FLOAT32;
    mlx_cpu_mxfp4_quantize_construction_layout layout{};
    REQUIRE(mlx_operation_event_cpu_mxfp4_quantize_construction_layout(&layout,scalar,rank,2,64));
    size_t capacity=0;
    for(size_t bytes:layout.seed_request_bytes) {
      mlx_original_buffer_population_layout physical{};
      REQUIRE(mlx_original_buffer_request_layout_for(&physical,runtime,bytes)==0);capacity+=physical.capacity;
    }
    unsigned retired=0;
    struct Budget {mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}} budget;
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,capacity,&retired,
        [](void* p){++*static_cast<unsigned*>(p);})==0);
    affine_construction_tests::Outputs escaped;
    {
      Role role;affine_construction_tests::replace_graph(role,layout.graph.allocation_extents);
      Observer observer;Bank bank;
      REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,
          layout.graph.primitives,layout.graph.seeds,layout.graph.maximum_rank)==0);
      struct Context {array* input;Stream stream;affine_construction_tests::Outputs* output;int status{-1};} context{&input,stream,&escaped};
      const auto attempts=native_recovery_without_allocations([](void* p){
        auto& c=*static_cast<Context*>(p);
        c.status=mlx_quantize_fixed(&c.output->values[0],&c.output->values[1],&c.output->values[2],
            {c.input},32,4,"mxfp4",{&c.stream});
      },&context);
      CHECK(attempts==0);REQUIRE(context.status==0);CHECK(escaped.values[2].ctx==nullptr);
      for(size_t i=0;i<2;++i) {
        REQUIRE(escaped.values[i].ctx);REQUIRE(escaped.values[i].prepared_owner);
        const auto& output=mlx_array_get_(escaped.values[i]);
        CHECK(output.ndim()==rank);CHECK(output.shape(0)==2);CHECK(output.shape(-1)==(i?2:8));
        CHECK(output.dtype()==(i?uint8:uint32));CHECK(output.status()==array::Status::unscheduled);
      }
      bank.reset();CHECK(role.graph->occupied_bytes()>0);
      CHECK(mlx_original_buffer_budget_occupied(budget.value)==capacity);
    }
    CHECK(retired==0);mlx_original_buffer_budget_release(budget.value);budget.value={};
    CHECK(retired==0);escaped.reset();CHECK(retired==1);
  }
}

TEST_CASE("CPU MXFP4 construction retires every refused constant prefix"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  for(auto dtype:{float16,bfloat16,float32})for(size_t failed=0;failed<6;++failed) {
    CAPTURE(dtype);CAPTURE(failed);
    auto input=affine_converter_tests::input(stream,dtype,2,32,4,false);
    mlx_cpu_mxfp4_quantize_construction_layout layout{};
    const auto scalar=dtype==float16?MLX_FLOAT16:dtype==bfloat16?MLX_BFLOAT16:MLX_FLOAT32;
    REQUIRE(mlx_operation_event_cpu_mxfp4_quantize_construction_layout(&layout,scalar,2,2,64));
    size_t capacity=0;
    for(size_t i=0;i<=failed;++i) {
      mlx_original_buffer_population_layout physical{};
      REQUIRE(mlx_original_buffer_request_layout_for(&physical,runtime,layout.seed_request_bytes[i])==0);
      capacity+=physical.capacity;
    }
    unsigned retired=0;
    struct Budget {mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}} budget;
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,capacity-1,&retired,
        [](void* p){++*static_cast<unsigned*>(p);})==0);
    {
      Role role;Observer observer;Bank bank;affine_construction_tests::Outputs outputs;
      REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,
          layout.graph.primitives,layout.graph.seeds,layout.graph.maximum_rank)==0);
      CHECK(mlx_quantize_fixed(&outputs.values[0],&outputs.values[1],&outputs.values[2],
          {&input},32,4,"mxfp4",{&stream})==1);
      for(const auto& output:outputs.values)CHECK(output.ctx==nullptr);
      REQUIRE(role.error.get()->borrow());
      try {std::rethrow_exception(role.error.get()->borrow()->exception);FAIL("expected constant capacity refusal");}
      catch(const allocator::OriginalBufferError& error){CHECK(error.cause()==allocator::OriginalBufferCause::capacity);}
      CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
      bank.reset();CHECK(role.graph->occupied_bytes()==0);CHECK(role.scope->query_records().pending==0);
    }
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==1);
  }
}

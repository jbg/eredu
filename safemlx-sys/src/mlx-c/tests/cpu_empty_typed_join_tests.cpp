// Included by graph_construction_tests.cpp; actual source queries and the same
// original Graph/Record/physical-owner fixtures used by the nonempty workers.
TEST_CASE("CPU empty Broadcast and Full preserve typed zero-data source and original completion"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(auto dtype:{float32,float16,bfloat16,int32})for(const Shape shape:{Shape{0,1},Shape{2,0,3}}) {
    array seed(7,dtype);
    auto ordinary=full(shape,seed,dtype,stream);
    cpu::CopyEvalStorage cold,actual;
    REQUIRE(cpu::alias_eval_layout(cpu::AliasOperation::EmptyBroadcast,0,shape.size(),false,cold));
    REQUIRE(cpu::alias_eval_storage(ordinary.inputs()[0],actual));
    CHECK(actual.allocation_extents==cold.allocation_extents);
    CHECK(actual.named_control_bytes==cold.named_control_bytes);
    CHECK(actual.backing_births==0);CHECK(actual.request_counts[3]==1);
    CHECK(actual.request_counts[6]==0);CHECK(actual.request_counts[7]==0);
    eval(ordinary.inputs());
    REQUIRE(cpu::scalar_full_eval_layout(dtype,shape.size(),0,false,cold));
    REQUIRE(cpu::selection_eval_storage(ordinary,actual));
    CHECK(actual.allocation_extents==cold.allocation_extents);
    CHECK(actual.named_control_bytes==cold.named_control_bytes);
    CHECK(actual.backing_births==0);CHECK(actual.request_counts[3]==3);
    CHECK(actual.request_counts[6]==1);CHECK(actual.request_counts[7]==0);
    const auto prior=actual;
    CHECK_FALSE(cpu::scalar_full_eval_layout(dtype,0,0,false,actual));
    CHECK_FALSE(cpu::scalar_full_eval_layout(float64,shape.size(),0,false,actual));
    array inherited(shape,dtype,nullptr,{});inherited.copy_shared_buffer(seed);
    auto false_empty=array(shape,dtype,std::make_shared<Full>(stream),{inherited});
    CHECK_FALSE(cpu::selection_eval_storage(false_empty,actual));
    CHECK(std::memcmp(&prior,&actual,sizeof(actual))==0);
    eval(ordinary);CHECK(ordinary.size()==0);CHECK(ordinary.data_size()==0);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,2,0,3)==0);
      auto output=full(shape,seed,dtype,stream);bank.reset();Operation operation;operation.append(output);
      const mlx_operation_eval_traversal_limits limits{1,4,3,3,3,1,8};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      CHECK(output.shape()==shape);CHECK(output.dtype()==dtype);CHECK(output.size()==0);CHECK(output.data_size()==0);
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
      CHECK_FALSE(info.known);escaped.emplace(output);
    }
    mlx_original_buffer_budget_release(budget.value);budget.value={};
    CHECK(escaped->shape()==shape);CHECK(escaped->size()==0);
    escaped.reset();CHECK(retired==1);
  }
}

namespace empty_typed_join_tests {
template<class T>void check(Dtype dtype,Stream stream,const std::vector<int>& counts) {
  using namespace pointwise_graph_tests;
  ArrayVector inputs;std::vector<T> expected;
  for(size_t i=0;i<counts.size();++i) {
    std::vector<T> values(size_t(counts[i])*3);
    for(size_t j=0;j<values.size();++j)values[j]=T(int(i)*10+int(j)-7);
    inputs.emplace_back(values.data(),Shape{counts[i],3},dtype);
    expected.insert(expected.end(),values.begin(),values.end());
  }
  auto ordinary=concatenate(inputs,0,stream);
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::concatenate_many_eval_layout(dtype,2,inputs.size(),expected.size(),false,cold));
  REQUIRE(cpu::concatenate_eval_storage(ordinary,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.named_control_bytes==cold.named_control_bytes);
  CHECK(actual.backing_births==size_t(!expected.empty()));
  CHECK(actual.request_counts[3]==2*inputs.size()+1);
  CHECK(actual.request_counts[6]==inputs.size());
  CHECK(actual.request_counts[7]==size_t(!expected.empty()));
  // The Rust quote crosses this C ABI before reaching the same native source.
  // Retain coverage of its exact scalar type and empty/populated geometry.
  const mlx_dtype scalar=dtype==int32?MLX_INT32:MLX_FLOAT32;
  mlx_cpu_copy_eval_layout bridge{};
  REQUIRE(mlx_operation_event_cpu_concatenate_many_eval_layout(
      &bridge,scalar,2,inputs.size(),expected.size(),false));
  CHECK(bridge.graph_extents==cold.allocation_extents);
  CHECK(bridge.worker_graph_extents==cold.worker_graph_extents);
  CHECK(bridge.backing_births==cold.backing_births);
  CHECK(bridge.named_control_bytes>cold.named_control_bytes);
  REQUIRE(cpu::concatenate_eval_layout(dtype,2,0,expected.size(),false,cold));
  REQUIRE(mlx_operation_event_cpu_concatenate_eval_layout(
      &bridge,scalar,2,0,expected.size(),false));
  CHECK(bridge.graph_extents==cold.allocation_extents);
  CHECK(bridge.backing_births==cold.backing_births);
  const auto prior_bridge=bridge;
  for(auto unsupported:{MLX_INT64,MLX_FLOAT64,MLX_BOOL,static_cast<mlx_dtype>(99)}) {
    CHECK_FALSE(mlx_operation_event_cpu_concatenate_many_eval_layout(
        &bridge,unsupported,2,inputs.size(),expected.size(),false));
    CHECK_FALSE(mlx_operation_event_cpu_concatenate_eval_layout(
        &bridge,unsupported,2,0,expected.size(),false));
    CHECK(std::memcmp(&prior_bridge,&bridge,sizeof(bridge))==0);
  }
  const auto prior=actual;
  CHECK_FALSE(cpu::concatenate_many_eval_layout(int64,2,inputs.size(),expected.size(),false,actual));
  CHECK_FALSE(cpu::concatenate_many_eval_layout(dtype,2,size_t(INT_MAX)+1,expected.size(),false,actual));
  array seed(1,dtype),inherited(Shape{0,3},dtype,nullptr,{});
  inherited.copy_shared_buffer(seed);
  auto false_empty=array(Shape{0,3},dtype,std::make_shared<Concatenate>(stream,0),{inherited,inherited});
  CHECK_FALSE(cpu::concatenate_eval_storage(false_empty,actual));
  CHECK(std::memcmp(&prior,&actual,sizeof(actual))==0);
  eval(ordinary);
  if(!expected.empty())CHECK(std::vector<T>(ordinary.data<T>(),ordinary.data<T>()+expected.size())==expected);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,inputs.size(),0,inputs.size())==0);
    auto output=concatenate(inputs,0,stream);bank.reset();Operation operation;operation.append(output);
    const mlx_operation_eval_traversal_limits limits{1,inputs.size()+2,2,inputs.size()+1,2,1,std::max(size_t{8},1+2*inputs.size())};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
    eval_traversal_tests::complete(role,operation,output);
    CHECK(output.shape()==ordinary.shape());CHECK(output.dtype()==dtype);CHECK(output.size()==expected.size());
    if(!expected.empty())CHECK(std::vector<T>(output.data<T>(),output.data<T>()+expected.size())==expected);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
    CHECK(info.known==!expected.empty());escaped.emplace(output);
  }
  mlx_original_buffer_budget_release(budget.value);budget.value={};
  if(!expected.empty()){CHECK(retired==0);CHECK(std::vector<T>(escaped->data<T>(),escaped->data<T>()+expected.size())==expected);}
  escaped.reset();CHECK(retired==1);
}
}
TEST_CASE("CPU typed concatenate prices every empty and integer job with exact retained backing"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(const std::vector<int> counts: {std::vector<int>{0,0},std::vector<int>{0,3,0},std::vector<int>{2,0,3}}) {
    empty_typed_join_tests::check<int32_t>(int32,stream,counts);
    empty_typed_join_tests::check<float>(float32,stream,counts);
  }
}

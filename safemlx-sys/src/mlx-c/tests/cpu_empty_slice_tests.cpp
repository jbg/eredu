#include "empty_buffer_fixture.h"
// The common Slice worker has a separate allocate_data(0) branch. Its zero
// Data owner is real metadata, but it neither reads nor aliases source bytes.

TEST_CASE("CPU empty Slice source proves exact zero Data constructor geometry"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  const bool zero_backing=empty_buffer_tests::has_backing();
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(auto dtype:{int32,uint32,uint8,bool_,float32,float16,bfloat16}) {
    CAPTURE(dtype);
    for(const auto& shape:std::vector<Shape>{{3},{1,1},{2,3,4},{2,3,4,5}}) {
      CAPTURE(shape.size());
      auto input=full(shape,array(1,dtype),dtype,stream);eval(input);input.eval();
      Shape starts(shape.size(),0),ends(shape),steps(shape.size(),1);ends[0]=0;
      auto output=slice(input,starts,ends,steps,stream);
      cpu::CopyEvalStorage cold,actual,positive;
      REQUIRE(cpu::slice_eval_layout(shape.size(),true,false,cold));
      REQUIRE(cpu::slice_eval_layout(shape.size(),false,false,positive));
      REQUIRE(cpu::slice_eval_storage(output,actual));
      CHECK(actual.named_control_bytes==cold.named_control_bytes);
      CHECK(actual.allocation_extents==cold.allocation_extents);
      CHECK(actual.allocation_extents>positive.allocation_extents);
      CHECK(actual.request_counts[3]==1);CHECK(actual.request_counts[6]==0);
      CHECK(actual.backing_births==size_t(zero_backing));CHECK(actual.worker_graph_extents==0);
      mlx_cpu_copy_eval_layout raw{};
      REQUIRE(mlx_operation_event_cpu_slice_eval_layout(&raw,shape.size(),true,false));
      CHECK(raw.graph_extents==actual.allocation_extents);CHECK(raw.backing_births==size_t(zero_backing));
      eval(output);output.eval();
      CHECK(output.size()==0);CHECK(output.data_size()==0);CHECK(output.dtype()==dtype);
      const auto saved=actual;
      auto wrong=array(output.shape(),dtype,std::make_shared<Slice>(stream,starts,shape,steps),{input});
      CHECK_FALSE(cpu::slice_eval_storage(wrong,actual));
      auto zero_steps=steps;zero_steps[0]=0;
      wrong=array(output.shape(),dtype,std::make_shared<Slice>(stream,starts,ends,zero_steps),{input});
      CHECK_FALSE(cpu::slice_eval_storage(wrong,actual));
      auto out_of_range=starts;out_of_range[0]=shape[0]+1;
      auto bad_ends=ends;bad_ends[0]=out_of_range[0];
      wrong=array(output.shape(),dtype,std::make_shared<Slice>(stream,out_of_range,bad_ends,steps),{input});
      CHECK_FALSE(cpu::slice_eval_storage(wrong,actual));
      CHECK(std::memcmp(&actual,&saved,sizeof(actual))==0);
    }
  }
}

TEST_CASE("CPU original empty Slice retires source backing while its zero Data survives"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  const bool zero_backing=empty_buffer_tests::has_backing();
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const int values[]={7,-11,23,13,17,-19};
  array base(values,Shape{2,3},int32);
  auto strided=slice(base,{0,0},{2,3},{1,2},stream);eval(strided);strided.eval();
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  unsigned retired=0;
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}} budget;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void* p){++*static_cast<unsigned*>(p);})==0);
  std::optional<array> source,escaped;
  mlx_original_buffer_population_layout zero{};
  REQUIRE(mlx_original_buffer_request_layout_for(&zero,runtime,0)==0);
  uint64_t capacity=0;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,2)==0);
    auto value=contiguous(strided,false,stream);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
    eval_traversal_tests::complete(role,operation,value);
    REQUIRE(value.data<int>()[0]==7);REQUIRE(value.data<int>()[1]==23);
    capacity=mlx_original_buffer_budget_occupied(budget.value);REQUIRE(capacity>0);
    source.emplace(value);
  }
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;
    REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,2)==0);
    auto value=slice(*source,{0,0},{0,2},{1,1},stream);bank.reset();Operation operation;operation.append(value);
    const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
    eval_traversal_tests::complete(role,operation,value);
    CHECK(value.shape()==Shape{0,2});CHECK(value.size()==0);CHECK(value.data_size()==0);
    CHECK(mlx_original_buffer_budget_occupied(budget.value)==capacity+zero.capacity);escaped.emplace(value);
  }
  source.reset();CHECK(mlx_original_buffer_budget_occupied(budget.value)==zero.capacity);
  REQUIRE(escaped.has_value());CHECK(escaped->size()==0);
  mlx_original_buffer_budget_release(budget.value);budget.value={};
  CHECK(retired==unsigned(!zero_backing));escaped.reset();CHECK(retired==1);
}

TEST_CASE("CPU empty Slice admits exact allocator backing and refuses one byte short"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  const bool backing=empty_buffer_tests::has_backing();
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  mlx_original_buffer_population_layout physical{};
  REQUIRE(mlx_original_buffer_request_layout_for(&physical,runtime,0)==0);
  for(bool short_budget:{false,true}) {
    if(short_budget&&!backing)continue;
    auto stream=new_stream(Device::cpu);prepare(stream,stream);
    const int values[]={7,-11,23};array source(values,Shape{3},int32);
    unsigned retired=0;
    struct Budget {mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}} budget;
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,
        physical.capacity-size_t(short_budget),&retired,[](void* p){++*static_cast<unsigned*>(p);})==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,1)==0);
      auto output=slice(source,{0},{0},{1},stream);bank.reset();
      Operation operation;operation.append(output);
      const auto status=eval_traversal_tests::submit(operation,stream,{1,3,2,2,2,1,8});
      if(short_budget) {
        CHECK(status==static_cast<unsigned>(ScopedEvaluation::failed));settle(role);
        CHECK(role.scope->query_records().pending==0);REQUIRE(role.error.get()->borrow());
        try {std::rethrow_exception(role.error.get()->borrow()->exception);FAIL("expected physical capacity refusal");}
        catch(const allocator::OriginalBufferError& error){CHECK(error.cause()==allocator::OriginalBufferCause::capacity);}
        CHECK(mlx_original_buffer_budget_occupied(budget.value)==0);
      } else {
        REQUIRE(status==0);eval_traversal_tests::complete(role,operation,output);
        CHECK(output.size()==0);CHECK(output.data_size()==0);REQUIRE(output.data_shared_ptr());
        mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
        CHECK(info.known==backing);CHECK(info.charged_bytes==physical.capacity);
        escaped.emplace(output);
      }
      CHECK(source.data<int>()[0]==7);CHECK(source.data<int>()[1]==-11);CHECK(source.data<int>()[2]==23);
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)==(short_budget?0:physical.capacity));
    mlx_original_buffer_budget_release(budget.value);budget.value={};
    CHECK(retired==unsigned(short_budget||!backing));escaped.reset();CHECK(retired==1);
  }
}

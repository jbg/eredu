#include "empty_buffer_fixture.h"
// Shared ordinary row movement, with exact native facts and real original
// Graph/Record completion. Included after the canonical graph fixtures.
namespace row_movement_tests {
struct Budget {
  mlx_original_buffer_budget value{};
  ~Budget(){mlx_original_buffer_budget_release(value);}
  void prepare(unsigned& retired) {
    mlx_prepared_input_runtime runtime{};
    REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    REQUIRE(mlx_original_buffer_budget_new_retaining(&value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);})==0);
  }
  void release(){mlx_original_buffer_budget_release(value);value={};}
};
void same_layout(const cpu::CopyEvalStorage& cold,const cpu::CopyEvalStorage& actual) {
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.worker_graph_extents==cold.worker_graph_extents);
  CHECK(actual.named_control_bytes==cold.named_control_bytes);
  CHECK(actual.backing_births==cold.backing_births);
}
void values(const array& output,const std::vector<float>& expected) {
  REQUIRE(output.size()==expected.size());
  if(!expected.empty())CHECK(std::vector<float>(output.data<float>(),output.data<float>()+expected.size())==expected);
}
}
TEST_CASE("CPU row Gather retains empty and typed index workers with exact original custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  const bool zero_backing=empty_buffer_tests::has_backing();
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(bool signed_index:{false,true})for(int rows:{0,4})for(int count:{0,1,7}) {
    if(rows==0&&count!=0)continue;
    CAPTURE(signed_index);CAPTURE(rows);CAPTURE(count);
    constexpr int width=3;
    std::vector<float> data(size_t(rows)*width),expected;
    for(size_t i=0;i<data.size();++i)data[i]=float(int(i)*3-11);
    std::vector<int32_t> signed_ids(count);std::vector<uint32_t> unsigned_ids(count);
    for(int i=0;i<count;++i) {
      unsigned_ids[i]=uint32_t((i*3+1)%rows);
      signed_ids[i]=int32_t(unsigned_ids[i])-(i%2?rows:0);
      for(int column=0;column<width;++column)expected.push_back(data[unsigned_ids[i]*width+column]);
    }
    array source(data.data(),Shape{rows,width},float32);
    auto index=signed_index?array(signed_ids.data(),Shape{count},int32):array(unsigned_ids.data(),Shape{count},uint32);
    auto ordinary=gather(source,index,0,Shape{1,width},stream);
    cpu::CopyEvalStorage cold,actual;
    REQUIRE(cpu::gather_eval_layout(float32,index.dtype(),2,1,data.size(),count,width,false,cold));
    REQUIRE(cpu::gather_eval_storage(ordinary,actual));
    row_movement_tests::same_layout(cold,actual);
    CHECK(actual.request_counts[3]==4);CHECK(actual.request_counts[6]==1);
    CHECK(actual.request_counts[7]==size_t(count!=0||zero_backing));CHECK(actual.request_counts[9]==1);
    mlx_cpu_copy_eval_layout raw{};
    REQUIRE(mlx_operation_event_cpu_gather_eval_layout(&raw,MLX_FLOAT32,signed_index?MLX_INT32:MLX_UINT32,
        2,1,data.size(),count,width,false));
    CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.backing_births==cold.backing_births);
    const auto prior=actual;
    CHECK_FALSE(cpu::gather_eval_layout(float32,int64,2,1,data.size(),count,width,false,actual));
    CHECK_FALSE(cpu::gather_eval_layout(float32,index.dtype(),2,1,0,1,width,false,actual));
    CHECK_FALSE(cpu::gather_eval_layout(float32,index.dtype(),2,1,data.size(),SIZE_MAX,width,false,actual));
    auto invalid=array(Shape{count,1,width+1},float32,std::make_shared<Gather>(stream,std::vector<int>{0},Shape{1,width}),{source,index});
    CHECK_FALSE(cpu::gather_eval_storage(invalid,actual));
    CHECK(std::memcmp(&prior,&actual,sizeof(actual))==0);
    eval(ordinary);row_movement_tests::values(ordinary,expected);
    unsigned retired=0;row_movement_tests::Budget budget;budget.prepare(retired);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,2)==0);
      auto output=gather(source,index,0,Shape{1,width},stream);
      bank.reset();Operation operation;operation.append(output);
      const mlx_operation_eval_traversal_limits limits{1,4,2,3,2,1,8};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      row_movement_tests::values(output,expected);CHECK(output.dtype()==float32);
      CHECK(output.shape()==Shape{count,1,width});CHECK(role.records->occupied_bytes()==0);
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
      CHECK(info.known==(count!=0||zero_backing));escaped.emplace(output);
    }
    budget.release();CHECK(retired==unsigned(count==0&&!zero_backing));
    row_movement_tests::values(*escaped,expected);escaped.reset();CHECK(retired==1);
    row_movement_tests::values(source,data);
  }
}
TEST_CASE("CPU row ScatterAxis sum preserves duplicate arithmetic zero updates and original custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(bool signed_index:{false,true})for(int width:{1,3})for(int count:{0,1,7}) {
    CAPTURE(signed_index);CAPTURE(width);CAPTURE(count);
    constexpr int rows=4;
    std::vector<float> data(rows*width),changes(size_t(count)*width);
    for(size_t i=0;i<data.size();++i)data[i]=float(int(i)*2-9);
    for(size_t i=0;i<changes.size();++i)changes[i]=float(int(i)%7-3);
    auto expected=data;
    std::vector<int32_t> signed_ids(count);std::vector<uint32_t> unsigned_ids(count);
    for(int i=0;i<count;++i) {
      unsigned_ids[i]=uint32_t(i%3==0?0:2); // Repeated rows retain ordinary addition order.
      signed_ids[i]=int32_t(unsigned_ids[i])-(i%2?rows:0);
      for(int column=0;column<width;++column)expected[unsigned_ids[i]*width+column]+=changes[i*width+column];
    }
    array source(data.data(),Shape{rows,width},float32),updates(changes.data(),Shape{count,width},float32);
    auto index=signed_index?array(signed_ids.data(),Shape{count,1},int32):array(unsigned_ids.data(),Shape{count,1},uint32);
    auto ordinary=scatter_add_axis(source,index,updates,0,stream);
    eval(ordinary.inputs());
    cpu::CopyEvalStorage cold,actual;
    REQUIRE(cpu::scatter_add_rows_eval_layout(index.dtype(),data.size(),changes.size(),false,cold));
    REQUIRE(cpu::scatter_axis_eval_storage(ordinary,actual));
    row_movement_tests::same_layout(cold,actual);
    CHECK(actual.backing_births==1);CHECK(actual.request_counts[3]==6);
    CHECK(actual.request_counts[6]==2);CHECK(actual.request_counts[7]==1);
    mlx_cpu_copy_eval_layout raw{};
    REQUIRE(mlx_operation_event_cpu_scatter_add_rows_eval_layout(&raw,signed_index?MLX_INT32:MLX_UINT32,
        data.size(),changes.size(),false));
    CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.backing_births==1);
    const auto prior=actual;
    CHECK_FALSE(cpu::scatter_add_rows_eval_layout(int64,data.size(),changes.size(),false,actual));
    CHECK_FALSE(cpu::scatter_add_rows_eval_layout(index.dtype(),0,changes.size(),false,actual));
    CHECK_FALSE(cpu::scatter_add_rows_eval_layout(index.dtype(),data.size(),SIZE_MAX,false,actual));
    auto invalid=array(source.shape(),float32,std::make_shared<ScatterAxis>(stream,ScatterAxis::Sum,1),ordinary.inputs());
    CHECK_FALSE(cpu::scatter_axis_eval_storage(invalid,actual));
    CHECK(std::memcmp(&prior,&actual,sizeof(actual))==0);
    eval(ordinary);row_movement_tests::values(ordinary,expected);
    unsigned retired=0;row_movement_tests::Budget budget;budget.prepare(retired);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;
      // Seven actual frontend candidates own six ArrayVector growth requests,
      // including identity casts/broadcasts. Only width>1 adds a Broadcast Eval.
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,7,0,2)==0);
      auto output=scatter_add_axis(source,index,updates,0,stream);
      bank.reset();Operation operation;operation.append(output);
      const size_t broadcasts=width>1?1:0;
      const mlx_operation_eval_traversal_limits limits{1,5+broadcasts,2+broadcasts,4+broadcasts,2+broadcasts,1,8};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      row_movement_tests::values(output,expected);CHECK(role.records->occupied_bytes()==0);
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
      CHECK(info.known);escaped.emplace(output);
    }
    budget.release();CHECK(retired==0);row_movement_tests::values(*escaped,expected);
    escaped.reset();CHECK(retired==1);row_movement_tests::values(source,data);
  }
}

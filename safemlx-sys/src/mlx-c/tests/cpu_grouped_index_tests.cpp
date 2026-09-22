// Included by graph_construction_tests.cpp; reuses its actual original role,
// completion and source-budget fixtures without a separate execution engine.
namespace grouped_index_tests {
std::vector<uint32_t> integer_order(const std::vector<int32_t>& values,size_t columns) {
  std::vector<uint32_t> expected(values.size());
  for(size_t row=0;row<values.size()/columns;++row) {
    auto start=expected.begin()+row*columns;
    std::iota(start,start+columns,0);
    std::stable_sort(start,start+columns,[&](uint32_t a,uint32_t b) {
      return values[row*columns+a]<values[row*columns+b];
    });
  }
  return expected;
}
}
TEST_CASE("CPU grouped I32 ArgSort authenticates rows and preserves stable original ordering"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(const Shape shape:{Shape{19},Shape{3,19},Shape{2,3,19}}) {
    size_t count=1;for(int dimension:shape)count*=dimension;
    std::vector<int32_t> values(count);
    for(size_t i=0;i<count;++i)values[i]=int32_t(i%7)-3;
    values[0]=INT32_MAX;values[1]=INT32_MIN;values[2]=INT32_MIN;
    array input(values.data(),shape,int32);
    const auto expected=grouped_index_tests::integer_order(values,19);
    auto ordinary=argsort(input,-1,stream);
    cpu::CopyEvalStorage cold,actual;
    REQUIRE(cpu::argsort_eval_layout(int32,shape.size(),19,count/19,false,cold));
    REQUIRE(cpu::argsort_eval_storage(ordinary,actual));
    CHECK(actual.allocation_extents==cold.allocation_extents);
    CHECK(actual.named_control_bytes==cold.named_control_bytes);
    CHECK(actual.worker_graph_extents==0);CHECK(actual.backing_births==1);
    const auto prior=actual;
    CHECK_FALSE(cpu::argsort_eval_layout(int64,shape.size(),19,count/19,false,actual));
    CHECK_FALSE(cpu::argsort_eval_layout(int32,shape.size(),SIZE_MAX,count/19,false,actual));
    CHECK_FALSE(cpu::argsort_eval_layout(int32,1,19,2,false,actual));
    CHECK_FALSE(cpu::argsort_eval_layout(float32,2,19,3,false,actual));
    if(shape.size()>1)CHECK_FALSE(cpu::argsort_eval_storage(argsort(input,0,stream),actual));
    array wrong_dtype(shape,uint32,std::make_shared<ArgSort>(stream,-1),
        {array(Shape(shape),uint64,nullptr,{})});
    CHECK_FALSE(cpu::argsort_eval_storage(wrong_dtype,actual));
    CHECK(std::memcmp(&prior,&actual,sizeof(actual))==0);
    mlx_cpu_copy_eval_layout raw{};
    REQUIRE(mlx_operation_event_cpu_argsort_eval_layout(&raw,MLX_INT32,shape.size(),19,count/19,false));
    CHECK(raw.graph_extents==cold.allocation_extents);
    eval(ordinary);
    REQUIRE(std::vector<uint32_t>(ordinary.data<uint32_t>(),ordinary.data<uint32_t>()+count)==expected);
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
    unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
        [](void*p){++*static_cast<unsigned*>(p);}, nullptr)==0);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,1)==0);
      auto output=argsort(input,-1,stream);bank.reset();Operation operation;operation.append(output);
      // One source, result and selected completion Synchronizer; the record's
      // actual initial capture capacity is eight.
      const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,8};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,output);
      CHECK(std::vector<uint32_t>(output.data<uint32_t>(),output.data<uint32_t>()+count)==expected);
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
      CHECK(info.known);escaped.emplace(output);
    }
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    escaped.reset();CHECK(retired==1);
  }
  const int32_t strided_values[]={4,99,1,99,1,99};
  array base(strided_values,Shape{6},int32);
  auto view=slice(base,Shape{0},Shape{6},Shape{2},stream);eval(view);
  cpu::CopyEvalStorage ignored;
  CHECK_FALSE(cpu::argsort_eval_storage(argsort(view,stream),ignored));
}

namespace grouped_index_tests {
template<class T>
void scatter_rows(Dtype dtype,const Shape& shape,Stream stream) {
  using namespace pointwise_graph_tests;
  const size_t width=shape.size()==1?1:size_t(shape[1]),count=5*width;
  std::vector<T> data(count),changes(4*width);
  for(size_t i=0;i<data.size();++i)data[i]=T(int(i)-9);
  for(size_t i=0;i<changes.size();++i)changes[i]=T(int(i)+31);
  const int32_t picks[]={-1,2,2,0};
  Shape update_shape=shape.size()==1?Shape{4,1}:Shape{4,1,int(width)};
  array source(data.data(),shape,dtype),index(picks,Shape{4},int32),updates(changes.data(),update_shape,dtype);
  auto ordinary=scatter(source,index,updates,0,stream);
  cpu::CopyEvalStorage cold,actual;
  REQUIRE(cpu::scatter_eval_layout(dtype,int32,shape.size(),count,changes.size(),false,cold));
  REQUIRE(cpu::flat_scatter_eval_storage(ordinary,actual));
  CHECK(actual.allocation_extents==cold.allocation_extents);
  CHECK(actual.named_control_bytes==cold.named_control_bytes);
  CHECK(actual.worker_graph_extents==cold.worker_graph_extents);
  CHECK(actual.rank==shape.size()+1);CHECK(actual.backing_births==1);
  CHECK(actual.request_counts[6]==2);CHECK(actual.request_counts[8]==1);
  const auto prior=actual;
  CHECK_FALSE(cpu::scatter_eval_layout(float64,int32,shape.size(),count,changes.size(),false,actual));
  CHECK_FALSE(cpu::scatter_eval_layout(dtype,uint32,shape.size(),count,changes.size(),false,actual));
  CHECK_FALSE(cpu::scatter_eval_layout(dtype,int32,3,count,changes.size(),false,actual));
  auto sum=array(shape,dtype,std::make_shared<Scatter>(stream,Scatter::Sum,std::vector<int>{0}),{source,index,updates});
  CHECK_FALSE(cpu::flat_scatter_eval_storage(sum,actual));
  auto wrong_axis=array(shape,dtype,std::make_shared<Scatter>(stream,Scatter::None,std::vector<int>{1}),{source,index,updates});
  CHECK_FALSE(cpu::flat_scatter_eval_storage(wrong_axis,actual));
  CHECK(std::memcmp(&prior,&actual,sizeof(actual))==0);
  mlx_cpu_copy_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_scatter_eval_layout(&raw,dtype==int32?MLX_INT32:MLX_FLOAT32,
      MLX_INT32,shape.size(),count,changes.size(),false));
  CHECK(raw.graph_extents==cold.allocation_extents);
  eval(ordinary);
  auto expected=data;
  for(size_t pick=0;pick<4;++pick)for(size_t column=0;column<width;++column) {
    const size_t row=picks[pick]<0?size_t(picks[pick]+5):size_t(picks[pick]);
    expected[row*width+column]=changes[pick*width+column];
  }
  CHECK(std::vector<T>(ordinary.data<T>(),ordinary.data<T>()+count)==expected);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
  unsigned retired=0;REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,1<<20,&retired,
      [](void*p){++*static_cast<unsigned*>(p);}, nullptr)==0);
  std::optional<array> escaped;
  {
    Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,3,0,2)==0);
    auto output=scatter(source,index,updates,0,stream);bank.reset();Operation operation;operation.append(output);
    const mlx_operation_eval_traversal_limits limits{1,6,3,5,3,1,16};
    REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
    eval_traversal_tests::complete(role,operation,output);
    CHECK(std::vector<T>(output.data<T>(),output.data<T>()+count)==expected);
    CHECK(std::vector<T>(source.data<T>(),source.data<T>()+count)==data);
    mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
    CHECK(info.known);escaped.emplace(output);
  }
  mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
  CHECK(std::vector<T>(escaped->data<T>(),escaped->data<T>()+count)==expected);
  escaped.reset();CHECK(retired==1);
}
}
TEST_CASE("CPU grouped Scatter preserves row overwrite duplicates negatives and original custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(const Shape shape:{Shape{5},Shape{5,7}}) {
    grouped_index_tests::scatter_rows<float>(float32,shape,stream);
    grouped_index_tests::scatter_rows<int32_t>(int32,shape,stream);
  }
}

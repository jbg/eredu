// Exact existing GatherAxis and floating log-add-exp workers used by routing.
TEST_CASE("CPU GatherAxis selects unequal widths through actual strided signed and unsigned sources"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(bool strided:{false,true})for(bool signed_index:{false,true}) {
    CAPTURE(strided);CAPTURE(signed_index);
    const int width=strided?9:4;
    std::vector<float> data(2*width),expected;
    for(size_t i=0;i<data.size();++i)data[i]=float(int(i)*3-13)*.25f;
    array base(data.data(),{2,width},float32);
    auto source=strided?slice(base,{0,1},{2,8},{1,2},stream):base;
    eval(source);source.eval();
    const uint32_t unsigned_ids[]={3,0,2,1,1,0};
    const int32_t signed_ids[]={-1,0,-2,1,-3,0};
    auto index=signed_index?array(signed_ids,{2,3},int32):array(unsigned_ids,{2,3},uint32);
    for(int row=0;row<2;++row)for(int column=0;column<3;++column)
      expected.push_back(data[row*width+(strided?1+2*unsigned_ids[row*3+column]:unsigned_ids[row*3+column])]);
    auto ordinary=take_along_axis(source,index,1,stream);
    cpu::CopyEvalStorage cold,actual;
    REQUIRE(cpu::gather_axis_row_eval_layout(2,6,false,cold));
    REQUIRE(cpu::gather_axis_row_eval_storage(ordinary,actual));
    row_movement_tests::same_layout(cold,actual);
    CHECK(actual.backing_births==1);CHECK(actual.request_counts[3]==4);CHECK(actual.request_counts[6]==1);
    mlx_cpu_copy_eval_layout raw{};
    REQUIRE(mlx_operation_event_cpu_gather_axis_row_eval_layout(&raw,2,6,false));
    CHECK(raw.graph_extents==cold.allocation_extents);
    const auto prior=actual;
    auto wrong_rows=array({1,3},float32,std::make_shared<GatherAxis>(stream,1),{source,index});
    auto wrong_axis=array({2,3},float32,std::make_shared<GatherAxis>(stream,2),{source,index});
    array too_short(source.shape(),float32,nullptr,{});too_short.copy_shared_buffer(array(1.f));
    auto bad_span=take_along_axis(too_short,index,1,stream);
    for(const auto* invalid:{&wrong_rows,&wrong_axis,&bad_span})CHECK_FALSE(cpu::gather_axis_row_eval_storage(*invalid,actual));
    CHECK(std::memcmp(&prior,&actual,sizeof(actual))==0);
    eval(ordinary);row_movement_tests::values(ordinary,expected);
    row_movement_tests::Budget budget;unsigned retired=0;budget.prepare(retired);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,2)==0);
      auto value=take_along_axis(source,index,1,stream);
      bank.reset();Operation operation;operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1,4,2,3,2,1,8};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,value);
      row_movement_tests::values(value,expected);CHECK(role.records->occupied_bytes()==0);
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
      REQUIRE(info.known);escaped.emplace(value);
    }
    budget.release();CHECK(retired==0);row_movement_tests::values(*escaped,expected);
    escaped.reset();CHECK(retired==1);
  }
}
TEST_CASE("CPU LogAddExp uses the exact floating task with stable numerics and original custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(const Shape shape:{Shape{},Shape{7},Shape{2,7},Shape{1,2,1,7}}) {
    size_t count=1;for(int dimension:shape)count*=dimension;
    std::vector<float> data(count),expected(count);
    for(size_t i=0;i<count;++i) {
      data[i]=float(int(i%7)*40-120);
      expected[i]=float(std::max(0.0,double(data[i]))+std::log1p(std::exp(-std::abs(double(data[i])))));
    }
    array input(data.data(),shape,float32),zero(0.f);
    auto ordinary=logaddexp(zero,input,stream);
    eval(ordinary.inputs());
    cpu::BinaryEvalStorage cold,actual;
    REQUIRE(cpu::binary_eval_layout(cpu::BinaryEvalKind::log_add_exp,float32,shape.size(),count,false,cold));
    REQUIRE(cpu::binary_eval_storage(ordinary,actual));
    CHECK(actual.allocation_extents==cold.allocation_extents);
    CHECK(actual.worker_graph_extents==cold.worker_graph_extents);
    CHECK(actual.named_control_bytes==cold.named_control_bytes);
    CHECK(actual.backing_births==1);CHECK(actual.request_counts[3]==4);CHECK(actual.request_counts[6]==1);
    mlx_cpu_binary_eval_layout raw{};
    REQUIRE(mlx_operation_event_cpu_binary_eval_layout(&raw,uint32_t(cpu::BinaryEvalKind::log_add_exp),MLX_FLOAT32,shape.size(),count,false));
    CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.backing_births==1);
    const auto prior=actual;
    CHECK_FALSE(cpu::binary_eval_layout(cpu::BinaryEvalKind::log_add_exp,int32,shape.size(),count,false,actual));
    CHECK_FALSE(cpu::binary_eval_layout(cpu::BinaryEvalKind::log_add_exp,float16,shape.size(),count,false,actual));
    CHECK_FALSE(cpu::binary_eval_layout(cpu::BinaryEvalKind::log_add_exp,float32,shape.size(),size_t(INT_MAX)+1,false,actual));
    CHECK(std::memcmp(&prior,&actual,sizeof(actual))==0);
    eval(ordinary);
    for(size_t i=0;i<count;++i)CHECK(ordinary.data<float>()[i]==doctest::Approx(expected[i]).epsilon(2e-6));
    row_movement_tests::Budget budget;unsigned retired=0;budget.prepare(retired);
    std::optional<array> escaped;
    {
      Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      Observer observer;Bank bank;
      const size_t broadcasts=shape.empty()?0:1;
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1+broadcasts,0,shape.size())==0);
      auto value=logaddexp(zero,input,stream);
      bank.reset();Operation operation;operation.append(value);
      const mlx_operation_eval_traversal_limits limits{1,4+broadcasts,2+broadcasts,3+broadcasts,2+broadcasts,1,8};
      REQUIRE(eval_traversal_tests::submit(operation,stream,limits)==0);
      eval_traversal_tests::complete(role,operation,value);
      for(size_t i=0;i<count;++i) {
        CHECK(value.data<float>()[i]==ordinary.data<float>()[i]);
        CHECK(value.data<float>()[i]==doctest::Approx(expected[i]).epsilon(2e-6));
      }
      CHECK(role.records->occupied_bytes()==0);
      mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&value},budget.value)==0);
      REQUIRE(info.known);escaped.emplace(value);
    }
    budget.release();CHECK(retired==0);
    for(size_t i=0;i<count;++i)CHECK(escaped->data<float>()[i]==ordinary.data<float>()[i]);
    escaped.reset();CHECK(retired==1);row_movement_tests::values(input,data);
  }
}

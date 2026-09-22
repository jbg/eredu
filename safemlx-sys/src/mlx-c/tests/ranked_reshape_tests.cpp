TEST_CASE("CPU ranked reshape planner preserves independent collapse decisions") {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  std::array<float,192> data{};for(size_t i=0;i<data.size();++i)data[i]=float(i);
  for(size_t rank:{4u,5u,11u,21u}) {
    CAPTURE(rank);
    Shape source_shape(rank,1);source_shape[0]=2;source_shape[1]=4;
    source_shape[rank-2]=3;source_shape[rank-1]=8;
    std::vector<int> axes(rank);std::iota(axes.begin(),axes.end(),0);std::swap(axes[1],axes[rank-2]);
    auto input=transpose(array(data.data(),source_shape,float32),axes,stream);eval(input);input.eval();
    for(const Shape& shape:std::vector<Shape>{{2,3,32},{2,3,4,8,1},{2,3,4,2,4},{6,32},{192},input.shape()}) {
      auto collapsed=collapse_contiguous_dims(input.shape(),input.strides(),int64_t(INT_MAX));
      Strides expected;bool expected_copy=false;size_t group=0;
      for(int dimension:shape) {
        if(group<collapsed.first.size()&&collapsed.first[group]%dimension==0) {
          collapsed.first[group]/=dimension;
          expected.push_back(collapsed.first[group]*collapsed.second[group]);
          group+=size_t(collapsed.first[group]==1);
        } else if(dimension==1)expected.push_back(expected.back());
        else {expected_copy=true;break;}
      }
      Strides actual(shape.size(),0);bool copy=!expected_copy;
      REQUIRE(borrowed_reshape_plan(input.shape().data(),input.strides().data(),rank,false,
          shape.data(),shape.size(),actual.data(),copy));
      CHECK(copy==expected_copy);if(!copy)CHECK(actual==expected);
      bool cold_copy=!expected_copy;
      REQUIRE(borrowed_reshape_plan(input.shape().data(),input.strides().data(),rank,false,
          shape.data(),shape.size(),nullptr,cold_copy));CHECK(cold_copy==copy);
      cpu::CopyEvalStorage layout{};
      REQUIRE(cpu::reshape_eval_layout(input.shape().data(),input.strides().data(),rank,
          shape.data(),shape.size(),false,layout));CHECK(layout.backing_births==size_t(copy));
    }
  }
  // Pure geometry cases retain many non-singleton groups beyond inline rank.
  // The existing general collapse implementation is the reference, not the
  // borrowed planner's own cold path.
  for(size_t rank=1;rank<=21;++rank)for(size_t variant=0;variant<8;++variant) {
    Shape shape(rank,2);Strides strides(rank,1);
    for(size_t axis=rank;axis--;) {
      if(axis+1<rank)strides[axis]=strides[axis+1]*2;
    }
    if(variant&1)std::reverse(strides.begin(),strides.end());
    if(variant&2)for(size_t axis=0;axis<rank;axis+=2)strides[axis]=0;
    if(variant&4)for(size_t axis=0;axis<rank;axis+=3)shape[axis]=1;
    size_t elements=1;for(int dimension:shape)elements*=dimension;
    std::vector<Shape> destinations{shape,Shape{int(elements)}};
    auto expanded=shape;expanded.insert(expanded.begin(),1);expanded.push_back(1);destinations.push_back(expanded);
    for(const auto& destination:destinations) {
      auto collapsed=collapse_contiguous_dims(shape,strides,int64_t(INT_MAX));
      Strides expected;bool expected_copy=false;size_t group=0;
      for(int dimension:destination) {
        if(group<collapsed.first.size()&&collapsed.first[group]%dimension==0) {
          collapsed.first[group]/=dimension;expected.push_back(collapsed.first[group]*collapsed.second[group]);
          group+=size_t(collapsed.first[group]==1);
        } else if(dimension==1)expected.push_back(expected.back());
        else{expected_copy=true;break;}
      }
      Strides result(destination.size(),0);
      struct Query {const Shape* shape;const Strides* strides;const Shape* destination;Strides* result;bool copy,valid{false};}
          query{&shape,&strides,&destination,&result,!expected_copy};
      const auto attempts=native_recovery_without_allocations([](void* raw){
        auto& q=*static_cast<Query*>(raw);
        q.valid=borrowed_reshape_plan(q.shape->data(),q.strides->data(),q.shape->size(),false,
            q.destination->data(),q.destination->size(),q.result->data(),q.copy);
      },&query);
      CHECK(attempts==0);REQUIRE(query.valid);CHECK(query.copy==expected_copy);
      if(!query.copy)CHECK(result==expected);
    }
  }
  bool copy=true;const int scalar[]={1},bad[]={0},huge[]={INT_MAX,2};const int64_t strides[]={1,1},negative[]={-1};
  for(bool prior:{false,true}) {
    copy=prior;
    CHECK_FALSE(borrowed_reshape_plan(scalar,strides,SIZE_MAX,false,scalar,1,nullptr,copy));
    CHECK_FALSE(borrowed_reshape_plan(scalar,strides,1,false,scalar,SIZE_MAX,nullptr,copy));
    CHECK_FALSE(borrowed_reshape_plan(bad,strides,1,false,scalar,1,nullptr,copy));
    CHECK_FALSE(borrowed_reshape_plan(huge,strides,2,false,scalar,1,nullptr,copy));
    CHECK_FALSE(borrowed_reshape_plan(scalar,negative,1,false,scalar,1,nullptr,copy));
    CHECK(copy==prior);
  }
}

namespace mxfp4_composed_tests {
void exercise(mlx::core::Dtype dtype,int rank,size_t rows,size_t columns,int variant) {
  using namespace affine_construction_tests;
  const float codebook[]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
  const size_t elements=rows*columns;
  {
    CAPTURE(dtype);CAPTURE(rank);CAPTURE(variant);CAPTURE(rows);CAPTURE(columns);
    const bool strided=variant==1;
    auto stream=new_stream(Device::cpu);prepare(stream,stream);
    Shape shape(rank,1);shape[0]=int(rows);shape.back()=int(columns)*(strided?2:1);
    std::vector<float> data(rows*shape.back(),99.f);
    for(size_t i=0;i<elements;++i)data[i*(strided?2:1)]=std::ldexp(codebook[i%16],int((i/32)%4)-1);
    auto input=astype(array(data.begin(),shape),dtype,stream);
    if(strided){Shape starts(rank,0),steps(rank,1);steps.back()=2;input=slice(input,starts,shape,steps,stream);}
    if(variant==2) {
      std::vector<float> transposed(elements);
      for(size_t i=0;i<elements;++i)transposed[(i%columns)*rows+i/columns]=std::ldexp(codebook[i%16],int((i/32)%4)-1);
      auto base=astype(array(transposed.begin(),Shape{int(columns),int(rows)}),dtype,stream);
      input=reshape(transpose(base,stream),shape,stream);
    }
    eval(input);input.eval();
    mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    unsigned retired=0;
    struct Budget {mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}} budget;
    mlx_cpu_mxfp4_quantize_payload_layout payload{};
    const auto input_scalar=dtype==float16?MLX_FLOAT16:dtype==bfloat16?MLX_BFLOAT16:MLX_FLOAT32;
    REQUIRE(mlx_operation_event_cpu_mxfp4_quantize_payload_layout(&payload,input_scalar,rows,columns));
    size_t capacity=0;
    for(size_t i=0;i<payload.request_count;++i) {
      mlx_original_buffer_population_layout physical{};
      REQUIRE(mlx_original_buffer_request_layout_for(&physical,runtime,payload.request_bytes[i])==0);
      capacity+=physical.capacity;
    }
    // The physical budget comes from the cold per-request inventory. The
    // fixture's Graph/Record arenas still isolate evaluation/custody coverage.
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,capacity,&retired,[](void* p){++*static_cast<unsigned*>(p);}, nullptr)==0);
    std::array<std::optional<array>,2> escaped;
    {
      Role role;Observer observer;Bank bank;Outputs outputs;
      REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      mlx_cpu_mxfp4_quantize_construction_layout layout{};
      const auto scalar=dtype==float16?MLX_FLOAT16:dtype==bfloat16?MLX_BFLOAT16:MLX_FLOAT32;
      REQUIRE(mlx_operation_event_cpu_mxfp4_quantize_construction_layout(&layout,scalar,rank,rows,columns));
      REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,layout.graph.primitives,layout.graph.seeds,layout.graph.maximum_rank)==0);
      REQUIRE(mlx_quantize_fixed(&outputs.values[0],&outputs.values[1],&outputs.values[2],{&input},32,4,"mxfp4",{&stream})==0);
      bank.reset();Operation operation;
      operation.append(mlx_array_get_(outputs.values[0]));operation.append(mlx_array_get_(outputs.values[1]));
      struct Submission {Operation* operation;Stream stream;unsigned status{999};} submission{&operation,stream};
      const auto attempts=native_recovery_without_allocations([](void* raw){
        auto& s=*static_cast<Submission*>(raw);
        s.status=eval_traversal_tests::submit(*s.operation,s.stream,{2,74,67,200,67,1,16});
      },&submission);
      CHECK(attempts==0);const auto status=submission.status;
      if(status) {
        settle(role);
        if(role.error.get()->borrow())try{std::rethrow_exception(role.error.get()->borrow()->exception);}catch(const std::exception& error){FAIL_CHECK(std::string(error.what()));}
      }
      REQUIRE(status==0);
      eval_traversal_tests::complete(role,operation,mlx_array_get_(outputs.values[0]));
      for(size_t i=0;i<2;++i) {
        auto& value=mlx_array_get_(outputs.values[i]);
        REQUIRE(validate_scoped_array(value,*role.scope)==ScopedEvaluation::complete);
        mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,outputs.values[i],budget.value)==0);CHECK(info.known);escaped[i].emplace(value);
      }
    }
    for(size_t i=0;i<elements;++i) {
      const unsigned expected=i%16==8?0:i%16;
      CHECK(((escaped[0]->data<uint32_t>()[i/8]>>((i%8)*4))&15u)==expected);
    }
    for(size_t i=0;i<elements/32;++i)CHECK(escaped[1]->data<uint8_t>()[i]==126+i%4);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);escaped={};CHECK(retired==1);
  }
}
} // namespace mxfp4_composed_tests

TEST_CASE("CPU MXFP4 composed constructor and Eval preserve codes at high rank") {
  for(auto dtype:{mlx::core::float16,mlx::core::bfloat16,mlx::core::float32})
    for(int rank:{2,3,4,11,21})for(int variant:{0,1,2})
      mxfp4_composed_tests::exercise(dtype,rank,2,64,variant);
}

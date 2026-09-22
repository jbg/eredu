// Shared SliceUpdate copy tasks with exact typed sources and positive steps.
TEST_CASE("CPU typed positive-step static update admits exact backing and preserves custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  mlx_prepared_input_runtime runtime{};REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  for(auto dtype:{float32,float16,bfloat16,int32}) {
    CAPTURE(dtype);
    for(Shape shape:{Shape{7},Shape{3,7},Shape{2,3,7},Shape{2,2,3,7}}) {
      CAPTURE(shape.size());
      for(int step:{1,2}) {
        CAPTURE(step);
        Shape starts(shape.size(),0),ends(shape),strides(shape.size(),1),update_shape(shape);
        starts.back()=1;ends.back()=6;strides.back()=step;update_shape.back()=1+4/step;
        if(shape.size()>1){starts[shape.size()-2]=1;update_shape[shape.size()-2]-=1;}
        const size_t count=std::accumulate(shape.begin(),shape.end(),size_t{1},std::multiplies<size_t>());
        const size_t update_count=std::accumulate(update_shape.begin(),update_shape.end(),size_t{1},std::multiplies<size_t>());
        std::vector<float> values(count),replacements(update_count);
        for(size_t i=0;i<count;++i)values[i]=float(int(i%19)-9);
        for(size_t i=0;i<update_count;++i)replacements[i]=-float(i+1);
        if(dtype!=int32){replacements[0]=-0.0f;replacements[1]=-INFINITY;}
        auto expected=values;
        for(size_t i=0;i<count;++i) {
          size_t offset=i,update_index=0,multiplier=1;bool selected=true;
          for(size_t axis=shape.size();axis-->0;) {
            const int coordinate=offset%shape[axis];offset/=shape[axis];
            const int delta=coordinate-starts[axis];
            if(delta<0||coordinate>=ends[axis]||delta%strides[axis]!=0)selected=false;
            if(delta>=0)update_index+=size_t(delta/strides[axis])*multiplier;
            multiplier*=update_shape[axis];
          }
          if(selected)expected[i]=replacements[update_index];
        }
        auto source=astype(array(values.data(),shape,float32),dtype,stream);
        auto update=astype(array(replacements.data(),update_shape,float32),dtype,stream);
        auto oracle=astype(array(expected.data(),shape,float32),dtype,stream);
        eval(source,update,oracle);source.eval();update.eval();oracle.eval();
        std::vector<unsigned char> source_bits(count*dtype.size());
        std::memcpy(source_bits.data(),source.data<void>(),source_bits.size());
        auto probe=slice_update(source,update,starts,ends,strides,stream);
        cpu::CopyEvalStorage cold,actual;
        REQUIRE(cpu::static_update_eval_layout(shape.size(),count,update_count,false,cold));
        REQUIRE(cpu::copy_eval_storage(probe,actual));
        CHECK(actual.allocation_extents==cold.allocation_extents);
        CHECK(actual.worker_graph_extents==cold.worker_graph_extents);
        CHECK(actual.backing_births==1);
        const auto saved=actual;
        auto wrong_width=array(shape,Dtype(dtype.val(),1),
            std::make_shared<SliceUpdate>(stream,SliceUpdate::None,starts,ends,strides),{source,update});
        CHECK_FALSE(cpu::copy_eval_storage(wrong_width,actual));
        CHECK(std::memcmp(&actual,&saved,sizeof(actual))==0);
        auto invalid_steps=strides;invalid_steps.back()=0;
        auto invalid=array(shape,dtype,std::make_shared<SliceUpdate>(stream,SliceUpdate::None,starts,ends,invalid_steps),{source,update});
        CHECK_FALSE(cpu::copy_eval_storage(invalid,actual));
        CHECK(std::memcmp(&actual,&saved,sizeof(actual))==0);
        mlx_original_buffer_population_layout physical{};
        REQUIRE(mlx_original_buffer_request_layout_for(&physical,runtime,count*dtype.size())==0);
        for(bool short_budget:{false,true}) {
          CAPTURE(short_budget);
          unsigned retired=0;
          struct Budget{mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}}budget;
          REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,physical.capacity-size_t(short_budget),&retired,
              [](void*p){++*static_cast<unsigned*>(p);}, nullptr)==0);
          std::optional<array> escaped;
          {
            Role role;REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
            Observer observer;Bank bank;REQUIRE(mlx_operation_event_prepare_resident_graph(&bank.value,observer.value,1,0,1)==0);
            auto output=slice_update(source,update,starts,ends,strides,stream);
            bank.reset();Operation operation;operation.append(output);
            const auto status=eval_traversal_tests::submit(operation,stream,{1,3,2,3,2,1,8});
            if(short_budget) {
              CHECK(status==static_cast<unsigned>(ScopedEvaluation::failed));settle(role);
              CHECK(role.scope->query_records().pending==0);REQUIRE(role.error.get()->borrow());
              try {std::rethrow_exception(role.error.get()->borrow()->exception);FAIL("expected capacity refusal");}
              catch(const allocator::OriginalBufferError& error){CHECK(error.cause()==allocator::OriginalBufferCause::capacity);}
            } else {
              REQUIRE(status==0);eval_traversal_tests::complete(role,operation,output);
              CHECK(output.shape()==shape);CHECK(output.dtype()==dtype);
              CHECK(std::memcmp(output.data<void>(),oracle.data<void>(),count*dtype.size())==0);
              mlx_original_buffer_info info{};REQUIRE(mlx_original_buffer_array_info(&info,{&output},budget.value)==0);
              CHECK(info.known);CHECK(info.charged_bytes==physical.capacity);escaped.emplace(output);
            }
            CHECK(std::memcmp(source.data<void>(),source_bits.data(),source_bits.size())==0);
          }
          CHECK(mlx_original_buffer_budget_occupied(budget.value)==(short_budget?0:physical.capacity));
          mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==unsigned(short_budget));
          if(escaped)CHECK(std::memcmp(escaped->data<void>(),oracle.data<void>(),count*dtype.size())==0);
          escaped.reset();CHECK(retired==1);
        }
      }
    }
  }
}

namespace quantization_half_tests {
using namespace pointwise_graph_tests;
float at(const array& input, size_t i) {
  return input.dtype()==float16 ? float(input.data<float16_t>()[i]) :
      input.dtype()==bfloat16 ? float(input.data<mlx::core::bfloat16_t>()[i]) : input.data<float>()[i];
}
} // namespace quantization_half_tests

TEST_CASE("CPU quantization half reduction sources authenticate dtype shape and bounds"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for (auto dtype : {float16,bfloat16,float32}) {
    for (int rank : {1,2,3,4}) {
      Shape shape(rank,1);shape.back()=7;if(rank>1)shape[0]=3;
      const size_t rows=rank==1?1:3;
      std::vector<float> data(rows*7,2.5f);
      auto input=astype(array(data.begin(),shape),dtype,stream);eval(input);input.eval();
      auto value=argmin(input,-1,true,stream);
      cpu::CopyEvalStorage cold{},actual{};
      REQUIRE(cpu::typed_arg_reduce_eval_layout(dtype,rank,7,rows,false,cold));
      REQUIRE(cpu::greedy_eval_storage(value,actual));
      CHECK(actual.allocation_extents==cold.allocation_extents);
      CHECK(actual.named_control_bytes==cold.named_control_bytes);
      CHECK(actual.worker_graph_extents==0);CHECK(actual.backing_births==1);
      if(dtype==float32) {
        cpu::CopyEvalStorage legacy{};
        REQUIRE(cpu::arg_reduce_eval_layout(rank,7,rows,false,legacy));
        CHECK(legacy.named_control_bytes==cold.named_control_bytes);
      }
      const auto saved=actual;
      auto output_shape=shape;output_shape.back()=1;
      auto bad_axis=array(output_shape,uint32,std::make_shared<ArgReduce>(stream,ArgReduce::ArgMin,rank),{input});
      auto bad_dtype=array(output_shape,int32,std::make_shared<ArgReduce>(stream,ArgReduce::ArgMin,rank-1),{input});
      auto bad_shape=array(shape,uint32,std::make_shared<ArgReduce>(stream,ArgReduce::ArgMin,rank-1),{input});
      for(const array* bad:{&bad_axis,&bad_dtype,&bad_shape}) {
        CHECK_FALSE(cpu::greedy_eval_storage(*bad,actual));
        CHECK(std::memcmp(&saved,&actual,sizeof(actual))==0);
      }
      for(size_t width:{size_t(0),size_t(INT_MAX)+1,SIZE_MAX})
        CHECK_FALSE(cpu::typed_arg_reduce_eval_layout(dtype,rank,width,rows,false,actual));
      for(size_t count:{size_t(0),size_t(INT_MAX)+2,SIZE_MAX})
        CHECK_FALSE(cpu::typed_arg_reduce_eval_layout(dtype,rank,7,count,false,actual));
      CHECK(std::memcmp(&saved,&actual,sizeof(actual))==0);
      mlx_cpu_copy_eval_layout raw{};
      const auto c_dtype=dtype==float16?MLX_FLOAT16:dtype==bfloat16?MLX_BFLOAT16:MLX_FLOAT32;
      REQUIRE(mlx_operation_event_cpu_typed_arg_reduce_eval_layout(&raw,c_dtype,rank,7,rows,false));
      CHECK(raw.graph_extents==cold.allocation_extents);CHECK(raw.named_control_bytes>=cold.named_control_bytes);
      const auto raw_saved=raw;
      CHECK_FALSE(mlx_operation_event_cpu_typed_arg_reduce_eval_layout(&raw,MLX_INT32,rank,7,rows,false));
      CHECK_FALSE(mlx_operation_event_cpu_typed_arg_reduce_eval_layout(&raw,c_dtype,5,7,rows,false));
      CHECK(std::memcmp(&raw_saved,&raw,sizeof(raw))==0);
      if(dtype==float32)continue;
      const auto kind=dtype==float16?cpu::ReductionEvalKind::Float16MaximumRows:cpu::ReductionEvalKind::Bfloat16MaximumRows;
      auto maximum=max(input,std::vector<int>{rank-1},true,stream);
      REQUIRE(cpu::reduction_eval_layout(kind,rank,7,rows,false,cold));
      REQUIRE(cpu::reduction_eval_storage(maximum,actual));
      CHECK(actual.allocation_extents==cold.allocation_extents);
      CHECK(actual.named_control_bytes==cold.named_control_bytes);
      CHECK(actual.worker_graph_extents==0);CHECK(actual.backing_births==1);
      const auto max_saved=actual;
      auto bad_max_axis=array(output_shape,dtype,std::make_shared<Reduce>(stream,Reduce::Max,std::vector<int>{rank}),{input});
      auto bad_max_dtype=array(output_shape,float32,std::make_shared<Reduce>(stream,Reduce::Max,std::vector<int>{rank-1}),{input});
      Shape starts(rank,0),steps(rank,1);steps.back()=2;
      auto sparse=slice(input,starts,shape,steps,stream);eval(sparse);sparse.eval();
      auto sparse_max=max(sparse,std::vector<int>{rank-1},true,stream);
      for(const array* bad:{&bad_max_axis,&bad_max_dtype,&sparse_max}) {
        CHECK_FALSE(cpu::reduction_eval_storage(*bad,actual));
        CHECK(std::memcmp(&max_saved,&actual,sizeof(actual))==0);
      }
      for(size_t width:{size_t(0),size_t(1),SIZE_MAX})
        CHECK_FALSE(cpu::reduction_eval_layout(kind,rank,width,rows,false,actual));
      REQUIRE(mlx_operation_event_cpu_reduction_eval_layout(&raw,unsigned(kind),rank,7,rows,false));
      CHECK(raw.graph_extents==cold.allocation_extents);
    }
  }
  cpu::CopyEvalStorage cold{};
  for(auto dtype:{bool_,uint32,int32,float64})
    CHECK_FALSE(cpu::typed_arg_reduce_eval_layout(dtype,2,7,3,false,cold));
  REQUIRE(cpu::typed_arg_reduce_eval_layout(float16,3,1,size_t(INT_MAX)+1,false,cold));
  CHECK_FALSE(cpu::arg_reduce_eval_layout(3,1,size_t(INT_MAX)+2,false,cold));
}

TEST_CASE("CPU quantization half maximum preserves SIMD NaNs tails and escaped custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float nan=std::numeric_limits<float>::quiet_NaN(),inf=std::numeric_limits<float>::infinity();
  for(auto dtype:{float16,bfloat16}) {
    for(int rank:{1,2,3,4}) {
      for(int width:{2,7,8,17,1031}) {
        for(int variant:{0,1,2,3}) {
          CAPTURE(dtype);CAPTURE(rank);CAPTURE(width);CAPTURE(variant);
          const int rows=rank==1?1:3;
          Shape shape(rank,1);shape[0]=rows;shape.back()=width;
          std::vector<float> data(size_t(rows)*width);
          std::vector<float> expected(rows,-inf);
          for(int row=0;row<rows;++row) {
            for(int i=0;i<width;++i) {
              float x=variant==3?-inf:float((i*7+row)%29)/4.0f-5.0f;
              if(variant==1&&i==0)x=nan;
              if(variant==2&&i==width-1)x=row==1?inf:nan;
              data[size_t(row)*width+i]=x;
              expected[row]=std::isnan(x)||std::isnan(expected[row])?nan:std::max(expected[row],x);
            }
          }
          auto input=astype(array(data.begin(),shape),dtype,stream);eval(input);input.eval();
          auto make=[&] {return max(input,std::vector<int>{rank-1},true,stream);};
          quantization_source_tests::exercise(stream,size_t(rows)*dtype.size(),rank,make,
              [&](const array& result) {
                CHECK(result.dtype()==dtype);CHECK(result.size()==size_t(rows));
                for(int row=0;row<rows;++row) {
                  const auto actual=quantization_half_tests::at(result,row);
                  if(std::isnan(expected[row]))CHECK(std::isnan(actual));else CHECK(actual==expected[row]);
                }
              });
          if(rank==2&&width==8&&variant==0)
            quantization_source_tests::exercise(stream,size_t(rows)*dtype.size(),rank,make,[](const array&){},true);
        }
      }
    }
  }
}

TEST_CASE("CPU quantization typed arg reductions preserve first ties and strided custody"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace pointwise_graph_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  const float nan=std::numeric_limits<float>::quiet_NaN(),inf=std::numeric_limits<float>::infinity();
  for(auto dtype:{float16,bfloat16,float32}) {
    for(bool minimum:{true,false}) {
      for(int rank:{1,2,3,4}) {
        for(bool strided:{false,true}) {
          for(int variant:{0,1,2,3,4}) {
            CAPTURE(dtype);CAPTURE(minimum);CAPTURE(rank);CAPTURE(strided);CAPTURE(variant);
            const int width=17,rows=rank==1?1:3,step=strided?2:1;
            Shape shape(rank,1);shape[0]=rows;shape.back()=width*step;
            std::vector<float> data(size_t(rows)*width*step,99.0f);
            std::vector<uint32_t> expected(rows,0);
            for(int row=0;row<rows;++row) {
              float best=0;
              for(int i=0;i<width;++i) {
                float x=float((i+row)%9)-4.0f;
                if(variant==0&&(i==3||i==7))x=minimum?-20.0f:20.0f;
                if(variant==1&&i==0)x=nan;
                if(variant==2&&i==8)x=nan;
                if(variant==3)x=i%2?-0.0f:0.0f;
                if(variant==4)x=i==4?-inf:i==12?inf:0.0f;
                data[(size_t(row)*width+i)*step]=x;
                if(i==0)best=x;
                else if(minimum?x<best:x>best){best=x;expected[row]=uint32_t(i);}
              }
            }
            auto input=astype(array(data.begin(),shape),dtype,stream);
            if(strided) {
              Shape starts(rank,0),steps(rank,1);steps.back()=step;
              input=slice(input,starts,shape,steps,stream);
            }
            eval(input);input.eval();
            auto make=[&] {return minimum?argmin(input,-1,true,stream):argmax(input,-1,true,stream);};
            quantization_source_tests::exercise(stream,size_t(rows)*4,rank,make,
                [&](const array& result) {
                  CHECK(result.dtype()==uint32);CHECK(result.size()==size_t(rows));
                  for(int row=0;row<rows;++row)CHECK(result.data<uint32_t>()[row]==expected[row]);
                });
            if(rank==2&&strided&&variant==0)
              quantization_source_tests::exercise(stream,size_t(rows)*4,rank,make,[](const array&){},true);
          }
        }
      }
    }
  }
}

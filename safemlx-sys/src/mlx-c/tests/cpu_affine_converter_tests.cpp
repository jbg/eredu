#include "mlx/backend/cpu/quantize_storage.h"
#include "mlx/fast_primitives.h"

namespace affine_converter_tests {
using namespace pointwise_graph_tests;
float scalar(const array& value, size_t i) {
  return value.dtype()==float16 ? float(value.data<float16_t>()[i]) :
      value.dtype()==bfloat16 ? float(value.data<mlx::core::bfloat16_t>()[i]) : value.data<float>()[i];
}
unsigned code(size_t group, int j, int bits) {
  const unsigned bins=(1u<<bits)-1;
  const unsigned q=j==0?0:j==1?bins:unsigned((j*7+group*3)%(bins+1));
  return bins-q;
}
array input(Stream stream,Dtype dtype,int rank,int group,int bits,bool strided) {
  Shape shape(rank,1);shape[0]=2;shape.back()=group*2*(strided?2:1);
  std::vector<float> data(2*shape.back(),99.f);
  const unsigned bins=(1u<<bits)-1;
  for(size_t g=0;g<4;++g)for(int j=0;j<group;++j)
    data[(g*group+j)*(strided?2:1)]=(g%2?-1.f:1.f)*float(bins-code(g,j,bits))*0.25f;
  auto result=astype(array(data.begin(),shape),dtype,stream);
  if(strided) {
    Shape starts(rank,0),steps(rank,1);steps.back()=2;
    result=slice(result,starts,shape,steps,stream);
  }
  eval(result);result.eval();return result;
}
void numerical(const std::array<std::optional<array>,3>& values,int group,int bits) {
  for(const auto& value:values)REQUIRE(value.has_value());
  const auto* packed=values[0]->data<uint8_t>();
  const unsigned bins=(1u<<bits)-1;
  for(size_t g=0;g<4;++g) {
    CHECK(scalar(*values[1],g)==(g%2?0.25f:-0.25f));
    CHECK(scalar(*values[2],g)==(g%2?-1.f:1.f)*float(bins)*0.25f);
    for(int j=0;j<group;++j) {
      unsigned observed=0;
      for(int b=0;b<bits;++b) {
        const size_t offset=(g*group+j)*bits+b;
        observed|=unsigned((packed[offset/8]>>(offset%8))&1)<<b;
      }
      CHECK(observed==code(g,j,bits));
    }
  }
}
void exercise(Stream stream,Dtype dtype,int rank,int group,int bits,bool copy,int root,int fail=-1) {
  CAPTURE(dtype);CAPTURE(rank);CAPTURE(group);CAPTURE(bits);CAPTURE(copy);CAPTURE(root);CAPTURE(fail);
  auto source=input(stream,dtype,rank,group,bits,copy);
  // This fixture isolates the Eval source. Public quantize construction occurs
  // before admission and is not a claim of complete producer construction funding.
  auto outputs=quantize(source,group,bits,"affine",std::nullopt,stream);
  cpu::QuantizeEvalStorage cold{},actual{};
  REQUIRE(cpu::affine_quantize_eval_layout(dtype,rank,2,group*2,group,bits,copy,false,cold));
  for(const auto& output:outputs) {
    REQUIRE(cpu::affine_quantize_eval_storage(output,actual));
    CHECK(actual.allocation_extents==cold.allocation_extents);
    CHECK(actual.named_control_bytes==cold.named_control_bytes);
    CHECK(actual.backing_births==size_t(3+copy));
  }
  mlx_prepared_input_runtime runtime{};
  REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
  std::array<size_t,3> capacities{};
  size_t capacity=0;
  mlx_original_buffer_population_layout physical{};
  if(copy) {
    REQUIRE(mlx_original_buffer_request_layout_for(&physical,runtime,source.nbytes())==0);
    capacity=physical.capacity;
  }
  for(size_t i=0;i<3;++i) {
    REQUIRE(mlx_original_buffer_request_layout_for(&physical,runtime,outputs[i].nbytes())==0);
    capacities[i]=physical.capacity;
    if(fail<0||i<=size_t(fail))capacity+=physical.capacity;
  }
  if(fail>=0)--capacity;
  unsigned retired=0;
  struct Budget {mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}} budget;
  REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,capacity,&retired,
      [](void* p){++*static_cast<unsigned*>(p);})==0);
  std::array<std::optional<array>,3> escaped;
  std::array<uint64_t,3> identities{};
  {
    Role role;
    REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
    Operation operation;operation.append(outputs[root]);
    // Input, three siblings and synchronizer; Quantize and synchronizer on tape.
    // Seven Data captures, plus three for optional copy (two weak, one owned).
    const mlx_operation_eval_traversal_limits limits{1,5,2,4,4,1,copy?10u:8u};
    const auto submitted=eval_traversal_tests::submit(operation,stream,limits);
    if(fail>=0) {
      REQUIRE(submitted==static_cast<unsigned>(ScopedEvaluation::failed));
      settle(role);
      REQUIRE(role.scope->query_records().pending==0);
      REQUIRE(role.error.get()->borrow());
      try {std::rethrow_exception(role.error.get()->borrow()->exception);FAIL("physical refusal must retain its cause");}
      catch(const allocator::OriginalBufferError& error){CHECK(error.cause()==allocator::OriginalBufferCause::capacity);}
      // A copy accepted before the refusal remains encoder-owned. Retire that
      // temporary only after its record establishes completion.
      auto temporaries=cpu::get_command_encoder(stream).take_temporaries();
    } else {
      if(submitted!=0) {
        settle(role);
        if(role.error.get()->borrow())try {std::rethrow_exception(role.error.get()->borrow()->exception);}
        catch(const std::exception& error){FAIL_CHECK(std::string(error.what()));}
      }
      REQUIRE(submitted==0);
      eval_traversal_tests::complete(role,operation,outputs[root]);
      for(const auto& output:outputs)REQUIRE(validate_scoped_array(output,*role.scope)==ScopedEvaluation::complete);
    }
    for(size_t i=0;i<3;++i) {
      mlx_original_buffer_info info{};
      REQUIRE(mlx_original_buffer_array_info(&info,{&outputs[i]},budget.value)==0);
      CAPTURE(i);
      if(fail<0) {
        REQUIRE(info.known);CHECK(info.identity!=0);CHECK(info.charged_bytes==capacities[i]);
        identities[i]=info.identity;escaped[i].emplace(outputs[i]);
      } else {
        // Failed outputs are not published as ready, even when their storage
        // was accepted before a later allocation failed.
        CHECK_FALSE(info.known);
        if(i<size_t(fail)) {
          REQUIRE(outputs[i].data_shared_ptr());
          const auto& buffer=outputs[i].data_shared_ptr()->buffer;
          REQUIRE(buffer.original_buffer_budget());
          CHECK(buffer.original_allocation_capacity()==capacities[i]);
          identities[i]=buffer.original_allocation_generation();
          CHECK(identities[i]!=0);escaped[i].emplace(outputs[i]);
        }
      }
    }
    operation.reset();outputs.clear();
  }
  CHECK(retired==0);
  for(size_t i=0;i<3;++i)if(escaped[i]) {
    mlx_original_buffer_info info{};
    REQUIRE(mlx_original_buffer_array_info(&info,{&*escaped[i]},budget.value)==0);
    if(fail<0)CHECK(info.identity==identities[i]);
    else {
      CHECK_FALSE(info.known);
      CHECK(escaped[i]->data_shared_ptr()->buffer.original_allocation_generation()==identities[i]);
    }
  }
  if(fail<0)numerical(escaped,group,bits);
  size_t remaining=0;
  for(size_t i=0;i<3;++i)if(escaped[i])remaining+=capacities[i];
  CHECK(mlx_original_buffer_budget_occupied(budget.value)==remaining);
  // Rotate destruction to cover all three outputs as the final retained owner.
  // Drop the budget handle before that last output; the backing owns its account.
  for(int offset=1;offset<=3;++offset) {
    const size_t index=(root+offset)%3;
    if(offset==3&&fail<0) {
      CHECK(retired==0);
      mlx_original_buffer_budget_release(budget.value);budget.value={};
      CHECK(retired==0);
    }
    if(escaped[index]){escaped[index].reset();remaining-=capacities[index];}
    if(budget.value.ctx)CHECK(mlx_original_buffer_budget_occupied(budget.value)==remaining);
  }
  mlx_original_buffer_budget_release(budget.value);budget.value={};
  CHECK(retired==1);
}
} // namespace affine_converter_tests

TEST_CASE("CPU affine converter packs all formats with three escaped outputs"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace affine_converter_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(auto dtype:{float16,bfloat16,float32})for(int group:{32,64,128})
    for(int bits:{2,3,4,5,6,8})for(int rank:{2,4,11})for(bool copy:{false,true})
      for(int root:{0,1,2})exercise(stream,dtype,rank,group,bits,copy,root);
}

TEST_CASE("CPU affine converter retains partial allocations through failure"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace affine_converter_tests;
  for(auto dtype:{float16,bfloat16,float32})for(bool copy:{false,true})
    for(int fail:{0,1,2}) {
      auto stream=new_stream(Device::cpu);prepare(stream,stream);
      exercise(stream,dtype,2,32,4,copy,fail,fail);
    }
}

TEST_CASE("CPU affine converter authenticates siblings and rejects invalid modes"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace affine_converter_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  auto source=input(stream,float32,2,32,4,false);
  auto outputs=quantize(source,32,4,"affine",std::nullopt,stream);
  cpu::QuantizeEvalStorage value{};
  REQUIRE(cpu::affine_quantize_eval_storage(outputs[0],value));
  const auto saved=value;
  mlx_cpu_copy_eval_layout raw{};
  REQUIRE(mlx_operation_event_cpu_affine_quantize_eval_layout(&raw,MLX_FLOAT32,2,2,64,32,4,false,false));
  CHECK(raw.graph_extents==value.allocation_extents);
  CHECK(raw.named_control_bytes>=value.named_control_bytes);
  const auto raw_saved=raw;
  CHECK_FALSE(mlx_operation_event_cpu_affine_quantize_eval_layout(&raw,MLX_INT32,2,2,64,32,4,false,false));
  CHECK(std::memcmp(&raw_saved,&raw,sizeof(raw))==0);
  for(int variant=0;variant<7;++variant) {
    const auto mode=variant==0?QuantizationMode::Mxfp4:QuantizationMode::Affine;
    const bool dequantize=variant==1;
    auto primitive=std::make_shared<fast::Quantize>(stream,
        std::function<ArrayVector(ArrayVector)>{},32,4,mode,dequantize);
    std::vector<Shape> shapes{{2,8},{2,2},{2,2}};
    std::vector<Dtype> dtypes{uint32,float32,float32};
    if(variant==2)shapes[0].back()=9;
    if(variant==3)shapes[1][0]=1;
    if(variant==4)dtypes[0]=float32;
    if(variant==5)dtypes[2]=float16;
    if(variant==6){shapes.pop_back();dtypes.pop_back();}
    auto malformed=array::make_arrays(shapes,dtypes,primitive,{source});
    for(const auto& output:malformed) {
      CHECK_FALSE(cpu::affine_quantize_eval_storage(output,value));
      CHECK(std::memcmp(&saved,&value,sizeof(value))==0);
    }
  }
  const auto siblings=outputs[0].siblings();
  for(int position:{0,1,2,3}) {
    outputs[0].set_siblings(ArrayVector{outputs[1],outputs[1]},position);
    CHECK_FALSE(cpu::affine_quantize_eval_storage(outputs[0],value));
    CHECK(std::memcmp(&saved,&value,sizeof(value))==0);
  }
  outputs[0].set_siblings(siblings,0);
  REQUIRE(cpu::affine_quantize_eval_storage(outputs[0],value));
  for(int group:{0,16,31,256})CHECK_FALSE(cpu::affine_quantize_eval_layout(float32,2,2,64,group,4,false,false,value));
  for(int bits:{0,1,7,9,32})CHECK_FALSE(cpu::affine_quantize_eval_layout(float32,2,2,64,32,bits,false,false,value));
  for(auto dtype:{int32,uint32,float64})CHECK_FALSE(cpu::affine_quantize_eval_layout(dtype,2,2,64,32,4,false,false,value));
  for(size_t rows:{size_t(0),size_t(INT_MAX),SIZE_MAX})
    CHECK_FALSE(cpu::affine_quantize_eval_layout(float32,2,rows,64,32,4,false,false,value));
  CHECK(std::memcmp(&saved,&value,sizeof(value))==0);
}

#include "mlx/quantize_construction.h"

namespace affine_construction_tests {
using namespace pointwise_graph_tests;
struct Outputs {
  mlx_array values[3]{};
  void reset() noexcept {for(auto& value:values){mlx_array_free(value);value={};}}
  ~Outputs(){reset();}
};
void replace_graph(Role& role,size_t capacity) {
  role.scope.reset();
  role.error=wait_record_facts::failure();
  role.graph.reset(submission::GraphQuota::create(capacity,nullptr,nullptr));
  REQUIRE(role.graph);
  role.scope.reset(new submission::Scope(nullptr,nullptr,role.records.get(),role.graph.get()));
  REQUIRE(role.scope->enable_scoped_observation());
  REQUIRE(role.scope->require_original_controls()==submission::NativeControlFailure::none);
  REQUIRE(role.scope->bind_failure(role.error));
  REQUIRE(role.scope->enable_original_controls()==submission::NativeControlFailure::none);
}
}

TEST_CASE("CPU affine construction authenticates rank and reservation bounds"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace affine_construction_tests;
  for(size_t rank:{2u,4u,11u,21u}) {
    submission::AffineQuantizeConstructionLayout native{};
    mlx_affine_quantize_construction_layout raw{};
    REQUIRE(submission::affine_quantize_construction_layout(rank,native));
    REQUIRE(mlx_operation_event_affine_quantize_construction_layout(&raw,rank));
    CHECK(raw.graph_extents==native.allocation_extents);
    CHECK(raw.named_control_bytes>=native.named_control_bytes);
    // The public bound includes per-block alignment headroom. The fresh-arena
    // allocator also exposes an exact minimum for a deterministic refusal test.
    size_t minimum=0,extent=0;
    REQUIRE(submission::GraphQuota::minimum_allocation_extent(native.header_bytes,native.header_alignment,extent));minimum+=extent;
    REQUIRE(submission::GraphQuota::minimum_allocation_extent(native.slots_bytes,native.slots_alignment,extent));minimum+=extent;
    for(size_t i=0;i<submission::GraphConstructionLayout::classes;++i)if(native.request_counts[i]) {
      REQUIRE(submission::GraphQuota::minimum_allocation_extent(native.request_bytes[i],native.reserved_alignment,extent));
      minimum+=extent*native.request_counts[i];
    }
    CHECK(minimum<=raw.graph_extents);
    for(bool short_budget:{false,true}) {
      Role role;replace_graph(role,minimum-size_t(short_budget));
      Observer observer;Bank bank;
      const auto status=mlx_operation_event_prepare_affine_quantize_graph(&bank.value,observer.value,rank);
      if(short_budget){CHECK(status==2);CHECK(bank.value==nullptr);}
      else {REQUIRE(status==0);CHECK(role.graph->occupied_bytes()==minimum);}
      bank.reset();CHECK(role.graph->occupied_bytes()==0);
    }
    const auto saved=raw;
    for(size_t bad:{size_t(0),size_t(1),SIZE_MAX}) {
      CHECK_FALSE(mlx_operation_event_affine_quantize_construction_layout(&raw,bad));
      CHECK(std::memcmp(&saved,&raw,sizeof(raw))==0);
    }
  }
}

TEST_CASE("CPU affine construction uses reserved storage for fixed C outputs"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace affine_construction_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(auto dtype:{float16,bfloat16,float32})for(int group:{32,64,128})
    for(int bits:{2,3,4,5,6,8})for(int rank:{2,4,11,21}) {
      CAPTURE(dtype);CAPTURE(group);CAPTURE(bits);CAPTURE(rank);
      auto input=affine_converter_tests::input(stream,dtype,rank,group,bits,false);
      mlx_affine_quantize_construction_layout layout{};
      REQUIRE(mlx_operation_event_affine_quantize_construction_layout(&layout,rank));
      Role role;replace_graph(role,layout.graph_extents);
      Observer observer;Bank bank;
      REQUIRE(mlx_operation_event_prepare_affine_quantize_graph(&bank.value,observer.value,rank)==0);
      struct Context {const array* input;Stream stream;int group,bits;Outputs output;int status{-1};} context{&input,stream,group,bits};
      const auto attempts=native_recovery_without_allocations([](void* p){
        auto& c=*static_cast<Context*>(p);
        c.status=mlx_quantize_fixed(&c.output.values[0],&c.output.values[1],&c.output.values[2],
            {const_cast<array*>(c.input)},c.group,c.bits,"affine",{&c.stream});
      },&context);
      CHECK(attempts==0);REQUIRE(context.status==0);
      for(size_t i=0;i<3;++i) {
        REQUIRE(context.output.values[i].ctx);REQUIRE(context.output.values[i].prepared_owner);
        const auto& output=mlx_array_get_(context.output.values[i]);
        CHECK(output.sibling_position()==i);CHECK(output.siblings().size()==2);
        CHECK(output.inputs()[0].id()==input.id());CHECK(output.ndim()==rank);
        CHECK(output.dtype()==(i?dtype:uint32));
        CHECK(output.shape(-1)==(i?2:group*2*bits/32));
      }
      bank.reset();CHECK(role.graph->occupied_bytes()>0);
      context.output.reset();CHECK(role.graph->occupied_bytes()==0);
    }
}

TEST_CASE("CPU affine constructed outputs retain original physical custody after Eval"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace affine_construction_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  for(int rank:{2,11,21})for(auto dtype:{float16,bfloat16,float32}) {
    auto input=affine_converter_tests::input(stream,dtype,rank,32,4,false);
    mlx_prepared_input_runtime runtime{};
    REQUIRE(mlx_prepared_input_runtime_prepare(&runtime)==0);
    size_t capacity=0;
    for(size_t bytes:{size_t(64),size_t(4*dtype.size()),size_t(4*dtype.size())}) {
      mlx_original_buffer_population_layout physical{};
      REQUIRE(mlx_original_buffer_request_layout_for(&physical,runtime,bytes)==0);
      capacity+=physical.capacity;
    }
    unsigned retired=0;
    struct Budget {mlx_original_buffer_budget value{};~Budget(){mlx_original_buffer_budget_release(value);}} budget;
    REQUIRE(mlx_original_buffer_budget_new_retaining(&budget.value,runtime,capacity,&retired,
        [](void* p){++*static_cast<unsigned*>(p);})==0);
    std::array<std::optional<array>,3> escaped;
    {
      Role role;Observer observer;Bank bank;Outputs outputs;
      REQUIRE(mlx_original_buffer_budget_bind({role.scope.get()},budget.value)==0);
      REQUIRE(mlx_operation_event_prepare_affine_quantize_graph(&bank.value,observer.value,rank)==0);
      REQUIRE(mlx_quantize_fixed(&outputs.values[0],&outputs.values[1],&outputs.values[2],
          {&input},32,4,"affine",{&stream})==0);
      bank.reset();Operation operation;
      auto& first=mlx_array_get_(outputs.values[0]);operation.append(first);
      REQUIRE(eval_traversal_tests::submit(operation,stream,{1,5,2,4,4,1,8})==0);
      eval_traversal_tests::complete(role,operation,first);
      for(size_t i=0;i<3;++i) {
        auto& value=mlx_array_get_(outputs.values[i]);
        REQUIRE(validate_scoped_array(value,*role.scope)==ScopedEvaluation::complete);
        mlx_original_buffer_info info{};
        REQUIRE(mlx_original_buffer_array_info(&info,outputs.values[i],budget.value)==0);
        CHECK(info.known);CHECK(info.identity!=0);escaped[i].emplace(value);
      }
    }
    CHECK(mlx_original_buffer_budget_occupied(budget.value)==capacity);
    affine_converter_tests::numerical(escaped,32,4);
    mlx_original_buffer_budget_release(budget.value);budget.value={};CHECK(retired==0);
    escaped={};CHECK(retired==1);
  }
}

TEST_CASE("CPU affine construction retains and releases a failed C publication prefix"
    * doctest::skip(!wait_record_facts::layout_qualified)) {
  using namespace affine_construction_tests;
  auto stream=new_stream(Device::cpu);prepare(stream,stream);
  auto input=affine_converter_tests::input(stream,float32,2,32,4,false);
  Role role;Observer observer;Bank bank;Outputs output;
  REQUIRE(mlx_operation_event_prepare_affine_quantize_graph(&bank.value,observer.value,2)==0);
  // Consume one shell-sized request. The complete core graph can still be
  // built, but the fixed third C destination must refuse publication.
  void* held=role.graph->allocate(sizeof(PreparedInputArray),alignof(PreparedInputArray));
  REQUIRE(mlx_quantize_fixed(&output.values[0],&output.values[1],&output.values[2],
      {&input},32,4,"affine",{&stream})==1);
  CHECK(output.values[0].ctx!=nullptr);CHECK(output.values[1].ctx!=nullptr);
  CHECK(output.values[2].ctx==nullptr);
  REQUIRE(role.error.get()->borrow());
  try {std::rethrow_exception(role.error.get()->borrow()->exception);FAIL("publication must retain its refusal");}
  catch(const submission::GraphQuotaError& error) {CHECK(error.cause()==submission::GraphFailure::construction_mismatch);}
  role.graph->deallocate(held,sizeof(PreparedInputArray),alignof(PreparedInputArray));
  bank.reset();CHECK(role.graph->occupied_bytes()>0);
  output.reset();CHECK(role.graph->occupied_bytes()==0);
}

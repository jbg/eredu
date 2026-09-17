// Run as two actual localhost Ring processes; see run_local_ring.py.
// Graph/Record/backing capacities come from the actual retained operation source.
// Runtime, source-host and role authorities are explicit fixture owners.
#include "mlx/c/distributed_group.h"
#include "mlx/c/event.h"
#include "mlx/c/original_buffer.h"
#include "mlx/c/private/array.h"
#include "mlx/mlx.h"
#include "mlx/distributed/ops.h"
#include "mlx/backend/cpu/distributed_storage.h"
#include "mlx/record_quota.h"
#include "mlx/scoped_evaluation.h"
#include "mlx/submission.h"
#include "mlx/failure.h"
#include "mlx/transforms_impl.h"
#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>


#include "distributed_original_ring_support.h"

namespace {
using namespace mlx::core;
namespace dist=mlx::core::distributed;
using namespace original_ring_test_support;
constexpr std::array<int,6> base{7,-11,23,31,-5,17};
array input_for(int rank,bool strided,Stream stream){
  std::array<int,6> values;
  for(size_t i=0;i<values.size();++i)values[i]=base[i]+100*rank;
  array input(values.begin(),Shape{2,3},int32);
  if(strided){
    auto transposed=transpose(input,{1,0},stream);eval(transposed);
    // eval waits its synchronizer; settle this actual array's ordinary event
    // before lending a completed source to a new original failure domain.
    transposed.wait();
    require(transposed.status()==array::Status::available && !transposed.event().valid() &&
        !transposed.has_primitive() && transposed.inputs().empty(),"settled transpose source");
    return transposed;
  }
  return input;
}
std::vector<int> expected(bool gather,bool strided){
  std::vector<int> result;result.reserve(gather?12:6);
  for(int rank=0;rank<(gather?2:1);++rank)
    for(size_t i=0;i<6;++i){
      const size_t index=strided?(i%2)*3+i/2:i;
      result.push_back(gather?base[index]+100*rank:2*base[index]+100);
    }
  return result;
}
void compare(const array& result,const std::vector<int>& oracle){
  require(result.size()==oracle.size(),"output population");
  const auto* values=result.data<int>();
  for(size_t i=0;i<oracle.size();++i)require(values[i]==oracle[i],"nonzero signed Ring oracle mismatch");
}
void run_case(dist::Group& group,int rank,Stream& stream,mlx_prepared_input_runtime runtime,
              bool gather,bool strided,bool fence){
  const auto oracle=expected(gather,strided);
  {
    auto input=input_for(rank,strided,stream);
    require(input.flags().row_contiguous!=strided,"ordinary input contiguity");
    auto output=gather?dist::all_gather(input,group,stream):dist::all_sum(input,group,stream);
    eval(output);compare(output,oracle);
  }
  auto input=input_for(rank,strided,stream);
  require(input.flags().row_contiguous!=strided,"original input contiguity");
  const unsigned operation=gather?3:0;
  mlx_distributed_cpu_eval_storage cold{};
  require(mlx_distributed_query_cpu_source_storage(&cold,{&group},{&input},operation,0),"cold actual operation source");
  require(cold.operation==operation && !cold.tracer && cold.allocation_extents>0,
      "cold operation population");
  mlx_distributed_cpu_eval_storage layout_source{};
  mlx_distributed_constructor_storage layout_constructor{};
  require(mlx_distributed_query_cpu_layout_storage(&layout_source,&layout_constructor,{&group},
      input.shape().data(),input.ndim(),MLX_INT32,operation,0),"layout source from actual Group");
  require(layout_source.operation==cold.operation && layout_source.allocation_extents==cold.allocation_extents &&
      layout_source.logical_backing_bytes==cold.logical_backing_bytes &&
      layout_source.copy_worker_graph_extent==cold.copy_worker_graph_extent &&
      layout_source.communication_worker_graph_extent==cold.communication_worker_graph_extent &&
      layout_constructor.primitives==1 && layout_constructor.input_edges==1,"layout and actual shared worker census");
  for(size_t i=0;i<10;++i)require(layout_source.request_bytes[i]==cold.request_bytes[i] &&
      layout_source.request_alignments[i]==cold.request_alignments[i] &&
      layout_source.request_counts[i]==cold.request_counts[i],"layout and actual allocation class differ");
  {
    const int bad_shape[]={-1,2};
    mlx_distributed_cpu_eval_storage refused{};refused.blocks=99;
    mlx_distributed_constructor_storage untouched{};untouched.blocks=87;
    require(!mlx_distributed_query_cpu_layout_storage(&refused,&untouched,{&group},bad_shape,2,MLX_INT32,operation,0) &&
        refused.blocks==99 && untouched.blocks==87,"layout refusal changed either output");
  }
  mlx_original_buffer_population_layout copy_backing{},output_backing{};
  require(mlx_original_buffer_request_layout_for(&copy_backing,runtime,cold.copy_backing_bytes)==0 &&
      mlx_original_buffer_request_layout_for(&output_backing,runtime,cold.output_backing_bytes)==0,
      "actual allocator per-birth layout");
  require(copy_backing.capacity<=SIZE_MAX-output_backing.capacity,"physical source overflow");
  const size_t backing_capacity=gather?copy_backing.capacity+output_backing.capacity:output_backing.capacity;
  mlx_distributed_cpu_completion_storage completed{};
  require(mlx_distributed_query_cpu_completion_storage(&completed,{&group},{&input},operation,0),
      "actual complete operation source");
  require(completed.traversal.limits.root_count==1 && completed.traversal.limits.array_nodes==3 &&
      completed.traversal.limits.tape_entries==2 && completed.traversal.limits.stream_count==1 &&
      completed.graph_capacity>completed.graph_allocation_extents && completed.record_capacity>0 &&
      completed.synchronizer_graph_extent>0 && completed.signal_graph_extent>0,"completed source census");
  {
    auto lazy=add(input,array(1,int32),stream);
    mlx_distributed_cpu_eval_storage lazy_source{};
    require(mlx_distributed_query_cpu_source_storage(&lazy_source,{&group},{&lazy},operation,0) &&
        lazy_source.allocation_extents==layout_source.allocation_extents &&
        lazy.status()==array::Status::unscheduled,"layout binding evaluated the lazy partial or changed its source");
    mlx_distributed_cpu_completion_storage refused{};refused.graph_capacity=99;
    require(!mlx_distributed_query_cpu_completion_storage(&refused,{&group},{&lazy},operation,0) &&
        refused.graph_capacity==99,"lazy predecessor accepted or refusal output mutated");
  }
  auto retired=std::make_shared<std::atomic<unsigned>>(0);
  {
    BufferBudget buffer(runtime,backing_capacity,retired);
    Role role(completed);
    require(mlx_original_buffer_budget_bind({role.scope.get()},buffer.value)==0,"bind actual buffer budget");
    {
      Event event(stream);
      require(mlx_operation_event_validate_traversal_leaf(event.observer,{&input})==0,"actual input leaf");
      mlx_distributed_constructor_storage construction{};
      require(mlx_distributed_group_constructor_storage(&construction,{&group},{&input},operation,0),"constructor source");
      require(construction.primitives==1 && construction.input_edges==1 && construction.blocks>0,"constructor population");
      Output output;
      require(mlx_distributed_construct_original(&output.value,event.observer,{&group},{&input},operation,0,{&stream})==0,"original constructor");
      require(output.value.prepared_owner!=nullptr,"original output owner");
      mlx_distributed_cpu_eval_storage source{};
      require(mlx_distributed_query_cpu_eval_storage(&source,output.value),"actual CPU Eval source");
      require(source.operation==operation && source.inputs==1 && source.possible_copy &&
          source.backing_births==(gather?2u:1u) && source.communication.socket_attempts>0 &&
          source.communication_worker_graph_extent>0 && source.allocation_extents>0,"Eval/worker populations");
      require(cold.input_rank==source.input_rank && cold.output_rank==source.output_rank &&
          cold.inputs==source.inputs && cold.blocks==source.blocks &&
          cold.allocation_extents==source.allocation_extents &&
          cold.logical_backing_bytes==source.logical_backing_bytes &&
          cold.copy_worker_graph_extent==source.copy_worker_graph_extent &&
          cold.communication_worker_graph_extent==source.communication_worker_graph_extent,
          "cold and accepted shared operation census differ");
      for(size_t i=0;i<10;++i)require(cold.request_bytes[i]==source.request_bytes[i] &&
          cold.request_alignments[i]==source.request_alignments[i] &&
          cold.request_counts[i]==source.request_counts[i],"cold and accepted allocation class differ");
      if(fence){
        group.mark_terminal_submission();
        mlx_distributed_cpu_eval_storage refused{};refused.blocks=99;
        require(!mlx_distributed_query_cpu_source_storage(&refused,{&group},{&input},operation,0) && refused.blocks==99,
            "fenced fresh source was admitted or changed refusal output");
        mlx_distributed_cpu_eval_storage accepted{};
        require(mlx_distributed_query_cpu_eval_storage(&accepted,output.value),"accepted Eval survives source fence");
        require(accepted.allocation_extents==source.allocation_extents,"fence changed accepted recipe");
      }
      require(mlx_operation_event_append(event.value,output.value)==0,"retain actual output root");
      // Actual graph: input leaf, collective, selected Synchronizer; two edges.
      require(mlx_operation_event_submit_on_stream_prepared(event.value,{&stream},&completed.traversal.limits)==0,"prepared Ring Eval submission");
      require(mlx_operation_event_wait(event.value)==0,"prepared Ring completion");
      auto& native=mlx_array_get_(output.value);
      require(validate_scoped_array(native,*role.scope)==ScopedEvaluation::complete,"same scope output completion");
      role.settle();compare(native,oracle);
      mlx_original_buffer_info backing{};
      require(mlx_original_buffer_array_info(&backing,output.value,buffer.value)==0 && backing.known && backing.charged_bytes==output_backing.capacity,
          "completed actual output backing custody/physical source");
    }
    role.settle();
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    while(role.graph->occupied_bytes() && std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
    require(role.graph->occupied_bytes()==0,"Graph retained after all actual roots/tasks retired");
    require(role.records->occupied_bytes()==0,"Record retained after completion");
  }
  require(retired->load()==1,"original buffer source did not retire exactly once");
  std::cout<<"RING_CASE rank="<<rank<<" gather="<<gather<<" strided="<<strided<<" values=";
  for(int value:oracle)std::cout<<value<<',';
  std::cout<<" original=passed ordinary=passed"<<std::endl;
}
}
int main(){
  try{
    set_default_device(Device::cpu);
    auto group=dist::init(true,"ring");
    const int rank=group.rank();require(group.size()==2 && (rank==0||rank==1),"requires actual two-rank group");
    dist::GroupStorageInventory inventory;require(group.storage_inventory(inventory),"retained inventory");
    require(inventory.kind==dist::GroupStorageKind::ring && inventory.socket_handles>0 &&
        inventory.pool_workers>0 && inventory.socket_workers>0,"actual Ring initialization");
    mlx_distributed_persistent_storage persistent{};
    require(mlx_distributed_group_persistent_storage(&persistent,{&group}),"actual persistent C++ source");
    require(persistent.kind==unsigned(dist::GroupStorageKind::ring) &&
        persistent.shared_owner_bytes>=inventory.implementation_bytes &&
        persistent.socket_nodes==inventory.socket_workers && persistent.socket_node_bytes>0 &&
        persistent.socket_bucket_bytes>0 && persistent.communication_buffer_bytes==inventory.buffer_bytes &&
        persistent.thread_count==inventory.pool_workers+inventory.socket_workers &&
        persistent.thread_cpp_runtime_bytes>0 && persistent.retained_cpp_bytes>inventory.buffer_bytes,
        "persistent actual owner population");
    require((persistent.unresolved&(dist::group_shared_control|dist::group_associative_storage|dist::group_submission_storage))==0,
        "known native C++ owners still unresolved");
    require(persistent.unresolved==0,"actual retained managed Ring owner is unqualified");
    auto alias=group;
    mlx_distributed_persistent_storage aliased{};
    require(mlx_distributed_group_persistent_storage(&aliased,{&alias}) &&
        mlx_distributed_group_same_implementation({&group},{&alias}) &&
        aliased.retained_cpp_bytes==persistent.retained_cpp_bytes,"persistent source identity");
    auto stream=new_stream(Device::cpu);
    auto warm=add(array({2,3,5}),array({7,11,13}),stream);eval(warm);
    mlx_submission_runtime_baseline baseline{};
    require(mlx_submission_prepare_runtime(&baseline,{&stream},{&stream})==0,"prepared native runtime");
    mlx_prepared_input_runtime runtime{};require(mlx_prepared_input_runtime_prepare(&runtime)==0,"input runtime");
    mlx_original_buffer_population_layout empty{},refused{};refused.capacity=99;
    require(mlx_original_buffer_request_layout_for(&empty,runtime,0)==0,"actual zero-byte strategy");
    require(mlx_original_buffer_request_layout_for(&refused,runtime,SIZE_MAX)!=0 && refused.capacity==99,
        "overflowing physical query changed its refused destination");
    for(bool gather:{false,true})for(bool strided:{false,true})
      run_case(group,rank,stream,runtime,gather,strided,gather&&strided);
    require(group.terminal_submission(),"last accepted graph did not fence source");
    clear_streams();
    std::cout<<"RING_CONFORMANCE_PASS rank="<<rank<<std::endl;
    return 0;
  }catch(const std::exception& error){std::cerr<<"RING_CONFORMANCE_FAIL "<<error.what()<<std::endl;return 1;}
}

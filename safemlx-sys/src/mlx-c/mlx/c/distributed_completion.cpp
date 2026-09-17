#include "mlx/c/distributed_group.h"
#include "mlx/c/private/mlx.h"
#include "mlx/backend/cpu/distributed_storage.h"
#include "mlx/backend/cpu/copy_storage.h"
#include "mlx/event.h"
#include "mlx/graph_quota.h"
#include "mlx/graph_construction.h"
#include "mlx/record_quota.h"
#include "mlx/submission.h"
#include <algorithm>
#include <limits>
#include <optional>

namespace {
using namespace mlx::core;
bool add(size_t& value,size_t bytes) noexcept {
  if(bytes>SIZE_MAX-value)return false;value+=bytes;return true;
}
// The same selected completion always executes its Synchronizer on one CPU
// stream. These are its existing cleanup, outputs() and final Event::signal
// owners; there is no second evaluator or fabricated primitive source.
bool completion_tail(size_t inputs,cpu::CopyEvalStorage& cleanup,
    EventDispatchControlLayout& dispatch) noexcept {
  return cpu::copy_eval_layout(0,inputs,false,false,cleanup) &&
      event_dispatch_control_layout(dispatch);
}
size_t frames() noexcept {
  return sizeof(mlx_distributed_cpu_completion_storage)*2+
      sizeof(mlx_distributed_cpu_eval_storage)+sizeof(mlx_distributed_constructor_storage)+
      sizeof(mlx_operation_eval_record_layout)+sizeof(mlx_operation_root_storage_layout)+
      sizeof(mlx_operation_eval_traversal_limits)+sizeof(mlx_operation_eval_traversal_layout)+
      sizeof(cpu::CopyEvalStorage)+sizeof(EventDispatchControlLayout)+
      sizeof(mlx_array)+sizeof(mlx_distributed_group)+sizeof(unsigned)+sizeof(int)+
      sizeof(void*)*5+sizeof(size_t)*13+sizeof(bool)*4+
      // The actual second CPU eval invocation and its fixed cleanup transport.
      sizeof(ArrayVector)*2+sizeof(Stream)+sizeof(array)+sizeof(submission::GraphConstruction*)+
      sizeof(std::optional<submission::FixedEvalPrologueGuard>)+sizeof(submission::GraphFailure);
}
// Both one-operation completion and the actual send/receive pair use this
// exact root/Synchronizer/Event/Record producer. Source operations contribute
// only their own constructor/Eval/worker extents before this shared tail.
bool finish_storage(mlx_distributed_cpu_completion_storage& value,
    const mlx_operation_eval_record_layout& eval,
    const mlx_operation_root_storage_layout& roots,
    const cpu::CopyEvalStorage& cleanup,
    const EventDispatchControlLayout& dispatch) noexcept {
  value.synchronizer_graph_extent=cleanup.allocation_extents;
  value.signal_graph_extent=dispatch.cpu_signal_extent;
  value.platform_events=eval.host_graph_platform_events;
  const size_t extents[]={roots.graph_request_extent,eval.host_graph_allocation_extents,
      value.synchronizer_graph_extent,value.signal_graph_extent};
  for(size_t extent:extents)if(!add(value.graph_allocation_extents,extent))return false;
  for(size_t i=0;i<11;++i) {
    if(!value.traversal.request_bytes[i])continue;
    size_t extent=0;
    if(!submission::RecordQuota::allocation_extent(value.traversal.request_bytes[i],
          value.traversal.request_alignments[i],extent) || !add(value.record_allocation_extents,extent))return false;
  }
  if(!submission::GraphQuota::fresh_capacity_for_extents(value.graph_allocation_extents,value.graph_capacity) ||
      !submission::RecordQuota::fresh_capacity_for_extents(value.record_allocation_extents,value.record_capacity))return false;
  return add(value.named_control_bytes,eval.named_control_bytes) &&
      add(value.named_control_bytes,value.traversal.named_control_bytes) &&
      add(value.named_control_bytes,cleanup.named_control_bytes) &&
      add(value.named_control_bytes,dispatch.controls);
}
size_t finish_frames() noexcept {
  return sizeof(mlx_distributed_cpu_completion_storage*)+sizeof(mlx_operation_eval_record_layout*)+
      sizeof(mlx_operation_root_storage_layout*)+sizeof(cpu::CopyEvalStorage*)+
      sizeof(EventDispatchControlLayout*)+sizeof(size_t)*7+sizeof(bool)*3;
}
size_t exchange_frames() noexcept {
  return frames()+finish_frames()+sizeof(mlx_distributed_cpu_eval_storage)*2+
      sizeof(mlx_distributed_constructor_storage)*2+sizeof(unsigned)*2+sizeof(int)*3+
      sizeof(size_t)*9+sizeof(bool)*4+
      // The same pair may borrow a distinct receive prototype. It contributes
      // no receive DAG edge, but its query/validation frames are real.
      sizeof(mlx_array)*4+sizeof(const array*)*2+sizeof(const int*)*4+
      sizeof(size_t)*4+sizeof(mlx_dtype)*2;
}
}
namespace {
size_t single_source_frames() noexcept {
  return sizeof(mlx_distributed_cpu_completion_storage*)+
      sizeof(mlx_distributed_cpu_eval_storage*)+sizeof(mlx_distributed_constructor_storage*);
}
bool finish_single_source(mlx_distributed_cpu_completion_storage* out,
    const mlx_distributed_cpu_eval_storage& cpu,
    const mlx_distributed_constructor_storage& constructor) noexcept {
  mlx_operation_eval_record_layout eval{};
  mlx_operation_root_storage_layout roots{};
  cpu::CopyEvalStorage cleanup{};
  EventDispatchControlLayout dispatch{};
  if(constructor.primitives!=1 || cpu.inputs>1 || constructor.input_edges!=cpu.inputs ||
      !mlx_operation_event_eval_record_layout(&eval,2,1,2) ||
      !mlx_operation_event_root_storage_layout(&roots,1) ||
      !completion_tail(1,cleanup,dispatch))return false;
  mlx_distributed_cpu_completion_storage value{};
  // Receive has no input edge; all other selected constructors retain one
  // completed leaf. Copy helpers stay inside the ordinary CPU primitive.
  const mlx_operation_eval_traversal_limits limits{1,2+cpu.inputs,2,1+cpu.inputs,2,1,
      std::max(eval.capture_slots,cpu.data_captures)};
  if(!mlx_operation_event_eval_traversal_layout(&value.traversal,&limits))return false;
  const size_t extents[]={constructor.allocation_extents,cpu.allocation_extents,
      cpu.copy_worker_graph_extent,cpu.communication_worker_graph_extent};
  for(size_t extent:extents)if(!add(value.graph_allocation_extents,extent))return false;
  value.named_control_bytes=frames()+finish_frames()+single_source_frames();
  if(!add(value.named_control_bytes,cpu.named_control_bytes) ||
      !add(value.named_control_bytes,constructor.named_control_bytes) ||
      !finish_storage(value,eval,roots,cleanup,dispatch))return false;
  *out=value;return true;
}
}
extern "C" size_t mlx_distributed_cpu_completion_storage_controls(mlx_distributed_group group) {
  mlx_operation_eval_record_layout eval{};
  mlx_operation_eval_traversal_layout traversal{};
  mlx::core::cpu::CopyEvalStorage cleanup{};
  mlx::core::EventDispatchControlLayout dispatch{};
  if(!mlx_operation_event_eval_record_layout(&eval,2,1,2) ||
      !completion_tail(1,cleanup,dispatch))return SIZE_MAX;
  const mlx_operation_eval_traversal_limits limits{1,3,2,2,2,1,eval.capture_slots};
  if(!mlx_operation_event_eval_traversal_layout(&traversal,&limits))return SIZE_MAX;
  size_t total=frames()+finish_frames()+single_source_frames();
  if(!add(total,mlx_distributed_cpu_source_storage_controls(group)) ||
      !add(total,mlx_distributed_group_constructor_storage_controls()) ||
      !add(total,eval.named_control_bytes) || !add(total,traversal.named_control_bytes) ||
      !add(total,cleanup.named_control_bytes) || !add(total,dispatch.controls))return SIZE_MAX;
  return total;
}
extern "C" bool mlx_distributed_query_cpu_completion_storage(
    mlx_distributed_cpu_completion_storage* out,mlx_distributed_group group,
    mlx_array input,unsigned operation,int peer) {
  using namespace mlx::core;
  if(!out || !group.ctx || !input.ctx || operation>5)return false;
  const auto& leaf=*static_cast<const array*>(input.ctx);
  // The pre-admission source is read-only. Ordinary pending events must be
  // settled by their owner before this loan; quotation never waits or detaches.
  if(leaf.status()!=array::Status::available || leaf.has_primitive() ||
      !leaf.inputs().empty() || leaf.event().valid() || leaf.is_tracer())return false;
  mlx_distributed_cpu_eval_storage cpu{};
  mlx_distributed_constructor_storage constructor{};
  if(!mlx_distributed_query_cpu_source_storage(&cpu,group,input,operation,peer) ||
      !mlx_distributed_group_constructor_storage(&constructor,group,input,operation,peer))return false;
  return finish_single_source(out,cpu,constructor);
}

extern "C" size_t mlx_distributed_cpu_completion_layout_storage_controls(mlx_distributed_group group) {
  // Reuse the same complete tail census, replacing only the input query source.
  size_t total=mlx_distributed_cpu_completion_storage_controls(group);
  const size_t query=mlx_distributed_cpu_layout_storage_controls(group);
  const size_t own=sizeof(mlx_distributed_cpu_completion_storage*)+sizeof(mlx_distributed_group)+
      sizeof(const int*)+sizeof(size_t)+sizeof(mlx_dtype)+sizeof(unsigned)+sizeof(int)+
      sizeof(mlx_distributed_cpu_eval_storage)+sizeof(mlx_distributed_constructor_storage)+sizeof(bool)*2;
  if(total==SIZE_MAX || query==SIZE_MAX || !add(total,query) || !add(total,own))return SIZE_MAX;
  return total;
}
extern "C" bool mlx_distributed_query_cpu_completion_layout_storage(
    mlx_distributed_cpu_completion_storage* out,mlx_distributed_group group,
    const int* shape,size_t rank,mlx_dtype dtype,unsigned operation,int peer) {
  if(!out || !group.ctx || (rank && !shape) || operation>5)return false;
  mlx_distributed_cpu_eval_storage cpu{};
  mlx_distributed_constructor_storage constructor{};
  if(!mlx_distributed_query_cpu_layout_storage(&cpu,&constructor,group,
      shape,rank,dtype,operation,peer))return false;
  return finish_single_source(out,cpu,constructor);
}



namespace {
bool finish_exchange_sources(mlx_distributed_cpu_completion_storage* out,
    const mlx_distributed_cpu_eval_storage (&cpu)[2],
    const mlx_distributed_constructor_storage (&constructor)[2]) noexcept {
  mlx_operation_eval_record_layout eval{};
  mlx_operation_root_storage_layout roots{};
  cpu::CopyEvalStorage cleanup{};
  EventDispatchControlLayout dispatch{};
  mlx_distributed_cpu_completion_storage value{};
  value.named_control_bytes=exchange_frames()+sizeof(mlx_distributed_cpu_eval_storage*)+
      sizeof(mlx_distributed_constructor_storage*)+sizeof(mlx_distributed_cpu_completion_storage*);
  size_t captures=0;
  for(size_t i=0;i<2;++i) {
    if(constructor[i].primitives!=1 || cpu[i].inputs!=(i==0?1:0) ||
        constructor[i].input_edges!=cpu[i].inputs)return false;
    const size_t extents[]={constructor[i].allocation_extents,cpu[i].allocation_extents,
        cpu[i].copy_worker_graph_extent,cpu[i].communication_worker_graph_extent};
    for(size_t extent:extents)if(!add(value.graph_allocation_extents,extent))return false;
    captures=std::max(captures,cpu[i].data_captures);
    if(!add(value.named_control_bytes,cpu[i].named_control_bytes) ||
        !add(value.named_control_bytes,constructor[i].named_control_bytes))return false;
  }
  if(!mlx_operation_event_eval_record_layout(&eval,3,1,3) ||
      !mlx_operation_event_root_storage_layout(&roots,2) ||
      !completion_tail(2,cleanup,dispatch))return false;
  // One shared settled leaf, two actual outputs and one Synchronizer. The
  // receive has no input edge; the send and two root edges retain their order.
  const mlx_operation_eval_traversal_limits limits{2,4,3,3,3,1,std::max(captures,eval.capture_slots)};
  if(!mlx_operation_event_eval_traversal_layout(&value.traversal,&limits) ||
      !finish_storage(value,eval,roots,cleanup,dispatch))return false;
  *out=value;return true;
}
}

namespace {
size_t exchange_controls(mlx_distributed_group group,bool layout) noexcept {
  mlx_operation_eval_record_layout eval{};
  mlx_operation_eval_traversal_layout traversal{};
  mlx::core::cpu::CopyEvalStorage cleanup{};
  mlx::core::EventDispatchControlLayout dispatch{};
  if(!mlx_operation_event_eval_record_layout(&eval,3,1,3) ||
      !completion_tail(2,cleanup,dispatch))return SIZE_MAX;
  const mlx_operation_eval_traversal_limits limits{2,4,3,3,3,1,eval.capture_slots};
  if(!mlx_operation_event_eval_traversal_layout(&traversal,&limits))return SIZE_MAX;
  size_t total=exchange_frames()+sizeof(mlx_distributed_cpu_eval_storage*)+
      sizeof(mlx_distributed_constructor_storage*)+sizeof(mlx_distributed_cpu_completion_storage*);
  for(size_t i=0;i<2;++i) {
    const size_t query=layout ? mlx_distributed_cpu_layout_storage_controls(group)
        : mlx_distributed_cpu_source_storage_controls(group);
    if(query==SIZE_MAX || !add(total,query) ||
        (!layout && !add(total,mlx_distributed_group_constructor_storage_controls())))return SIZE_MAX;
  }
  if(!add(total,eval.named_control_bytes) || !add(total,traversal.named_control_bytes) ||
      !add(total,cleanup.named_control_bytes) || !add(total,dispatch.controls))return SIZE_MAX;
  return add(total,sizeof(bool)+sizeof(size_t)*2) ? total : SIZE_MAX;
}
}
extern "C" size_t mlx_distributed_cpu_exchange_storage_controls(mlx_distributed_group group) {
  const size_t bytes=exchange_controls(group,false);
  return bytes>SIZE_MAX-sizeof(mlx_distributed_group) ? SIZE_MAX : bytes+sizeof(mlx_distributed_group);
}
extern "C" bool mlx_distributed_query_cpu_exchange_sources_storage(
    mlx_distributed_cpu_completion_storage* out,mlx_distributed_group group,
    mlx_array input,mlx_array receive_like,int destination,int source) {
  using namespace mlx::core;
  if(!out || !group.ctx || !input.ctx || !receive_like.ctx)return false;
  const mlx_array inputs[2]={input,receive_like};
  for(const auto& value:inputs) {
    const auto& leaf=*static_cast<const array*>(value.ctx);
    if(leaf.status()!=array::Status::available || leaf.has_primitive() ||
        !leaf.inputs().empty() || leaf.event().valid() || leaf.is_tracer())return false;
  }
  if(static_cast<const array*>(input.ctx)->dtype()!=
      static_cast<const array*>(receive_like.ctx)->dtype())return false;
  mlx_distributed_cpu_eval_storage cpu[2]{};
  mlx_distributed_constructor_storage constructor[2]{};
  const unsigned operations[2]={4,5};
  const int peers[2]={destination,source};
  for(size_t i=0;i<2;++i) {
    if(!mlx_distributed_query_cpu_source_storage(&cpu[i],group,inputs[i],operations[i],peers[i]) ||
        !mlx_distributed_group_constructor_storage(&constructor[i],group,inputs[i],operations[i],peers[i]))return false;
  }
  return finish_exchange_sources(out,cpu,constructor);
}
extern "C" bool mlx_distributed_query_cpu_exchange_storage(
    mlx_distributed_cpu_completion_storage* out,mlx_distributed_group group,
    mlx_array input,int destination,int source) {
  return mlx_distributed_query_cpu_exchange_sources_storage(out,group,input,input,destination,source);
}

extern "C" size_t mlx_distributed_cpu_exchange_layout_storage_controls(mlx_distributed_group group) {
  size_t total=exchange_controls(group,true);
  if(total==SIZE_MAX)return SIZE_MAX;
  const size_t own=sizeof(mlx_distributed_cpu_completion_storage*)+sizeof(mlx_distributed_group)+
      sizeof(const int*)+sizeof(size_t)+sizeof(mlx_dtype)+sizeof(int)*4+sizeof(unsigned)*2+
      sizeof(mlx_distributed_cpu_eval_storage)*2+sizeof(mlx_distributed_constructor_storage)*2+
      sizeof(size_t)+sizeof(bool)*2;
  if(!add(total,own))return SIZE_MAX;
  return total;
}
extern "C" bool mlx_distributed_query_cpu_exchange_layouts_storage(
    mlx_distributed_cpu_completion_storage* out,mlx_distributed_group group,
    const int* send_shape,size_t send_rank,const int* receive_shape,size_t receive_rank,
    mlx_dtype dtype,int destination,int source) {
  if(!out || !group.ctx || (send_rank && !send_shape) || (receive_rank && !receive_shape))return false;
  mlx_distributed_cpu_eval_storage cpu[2]{};
  mlx_distributed_constructor_storage constructor[2]{};
  const unsigned operations[2]={4,5};
  const int peers[2]={destination,source};
  const int* shapes[2]={send_shape,receive_shape};
  const size_t ranks[2]={send_rank,receive_rank};
  for(size_t i=0;i<2;++i) {
    if(!mlx_distributed_query_cpu_layout_storage(&cpu[i],&constructor[i],group,
        shapes[i],ranks[i],dtype,operations[i],peers[i]))return false;
  }
  return finish_exchange_sources(out,cpu,constructor);
}
extern "C" bool mlx_distributed_query_cpu_exchange_layout_storage(
    mlx_distributed_cpu_completion_storage* out,mlx_distributed_group group,
    const int* shape,size_t rank,mlx_dtype dtype,int destination,int source) {
  return mlx_distributed_query_cpu_exchange_layouts_storage(out,group,
      shape,rank,shape,rank,dtype,destination,source);
}

extern "C" size_t mlx_distributed_variable_completion_storage_controls(mlx_distributed_group group) {
  size_t total=mlx_distributed_cpu_completion_storage_controls(group);
  const size_t query=mlx_distributed_variable_storage_controls(group);
  const size_t own=sizeof(mlx_distributed_cpu_completion_storage*)+sizeof(mlx_distributed_group)+
      sizeof(const int*)+sizeof(size_t)*3+sizeof(mlx_dtype)+sizeof(const size_t*)+sizeof(bool)*3+
      sizeof(mlx_distributed_cpu_eval_storage)+sizeof(mlx_distributed_constructor_storage)+sizeof(mlx_array);
  if(total==SIZE_MAX || query==SIZE_MAX || !add(total,query) || !add(total,own))return SIZE_MAX;
  return total;
}
extern "C" bool mlx_distributed_query_variable_completion_layout_storage(
    mlx_distributed_cpu_completion_storage* out,mlx_distributed_group group,
    const int* shape,size_t rank,mlx_dtype dtype,const size_t* matrix,size_t entries,bool transposed) {
  if(!out)return false;
  mlx_distributed_cpu_eval_storage cpu{};
  mlx_distributed_constructor_storage constructor{};
  return mlx_distributed_query_variable_layout_storage(&cpu,&constructor,group,
      shape,rank,dtype,matrix,entries,transposed) && finish_single_source(out,cpu,constructor);
}
extern "C" bool mlx_distributed_query_variable_completion_storage(
    mlx_distributed_cpu_completion_storage* out,mlx_distributed_group group,
    mlx_array input,const size_t* matrix,size_t entries,bool transposed) {
  using namespace mlx::core;
  if(!out || !input.ctx)return false;
  const auto& leaf=*static_cast<const array*>(input.ctx);
  if(leaf.status()!=array::Status::available || leaf.has_primitive() ||
      !leaf.inputs().empty() || leaf.event().valid() || leaf.is_tracer())return false;
  return mlx_distributed_query_variable_completion_layout_storage(out,group,
      leaf.shape().data(),leaf.ndim(),mlx_dtype_to_c(leaf.dtype()),matrix,entries,transposed);
}

extern "C" size_t mlx_distributed_variable_envelope_storage_controls(mlx_distributed_group group) {
  size_t total=mlx_distributed_variable_completion_storage_controls(group);
  const size_t own=mlx::core::distributed::VariableAllToAllEnvelope::inspection_controls()*3+
      sizeof(mlx_distributed_cpu_eval_storage)*2+sizeof(mlx_distributed_constructor_storage)*2+
      sizeof(mlx_distributed_cpu_completion_storage)*2+sizeof(void*)*10+sizeof(size_t)*8+
      sizeof(mlx_dtype)+sizeof(mlx_distributed_group)+sizeof(bool)*4;
  if(total==SIZE_MAX || !add(total,own))return SIZE_MAX;
  return total;
}
extern "C" bool mlx_distributed_query_variable_envelope_storage(
    mlx_distributed_cpu_eval_storage* out,mlx_distributed_constructor_storage* constructor_out,
    mlx_distributed_cpu_completion_storage* completion_out,mlx_distributed_group group,
    const int* shape,size_t rank,mlx_dtype dtype,size_t maximum_receive_rows) {
  if(!out || !constructor_out || !completion_out)return false;
  mlx_distributed_cpu_eval_storage cpu{};
  mlx_distributed_constructor_storage constructor{};
  mlx_distributed_cpu_completion_storage completion{};
  if(!mlx_distributed_query_variable_envelope_layout_storage(&cpu,&constructor,group,
        shape,rank,dtype,maximum_receive_rows) || !finish_single_source(&completion,cpu,constructor))return false;
  *out=cpu;*constructor_out=constructor;*completion_out=completion;return true;
}

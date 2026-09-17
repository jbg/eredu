#include "mlx/c/distributed_group.h"
#include "mlx/c/private/distributed_storage.h"
#include "mlx/c/private/mlx.h"
#include "mlx/backend/cpu/distributed_storage.h"
#include <algorithm>
#include <limits>

namespace {
bool publish(mlx_distributed_cpu_eval_storage* out,
    const mlx::core::distributed::CpuDistributedStorage& native,size_t controls) noexcept {
  mlx_distributed_cpu_eval_storage value{};
  value.operation=static_cast<unsigned>(native.operation);value.peer=native.peer;
  value.input_rank=native.input_rank;value.output_rank=native.output_rank;value.inputs=native.inputs;
  value.tracer=native.tracer;value.possible_copy=native.possible_copy;
  value.backing_births=native.backing_births;value.data_captures=native.data_captures;
  value.temporary_batches=native.temporary_batches;value.logical_backing_bytes=native.logical_backing_bytes;
  value.copy_backing_bytes=native.copy_backing_bytes;value.output_backing_bytes=native.output_backing_bytes;
  value.copy_worker_graph_extent=native.copy_worker_graph_extent;
  value.communication_worker_graph_extent=native.communication_worker_graph_extent;
  value.communication={native.communication.pool_jobs,native.communication.socket_attempts,
      native.communication.destination_arrays,native.communication.task_graph_extent,
      native.communication.destination_graph_extent,native.communication.controls};
  value.blocks=native.blocks;value.header_bytes=native.header_bytes;value.header_alignment=native.header_alignment;
  value.slots_bytes=native.slots_bytes;value.slots_alignment=native.slots_alignment;
  value.reserved_alignment=native.reserved_alignment;value.requested_bytes=native.requested_bytes;
  value.allocation_extents=native.allocation_extents;
  if(controls>SIZE_MAX-native.named_control_bytes)return false;
  value.named_control_bytes=native.named_control_bytes+controls;
  std::copy(std::begin(native.request_bytes),std::end(native.request_bytes),value.request_bytes);
  std::copy(std::begin(native.request_alignments),std::end(native.request_alignments),value.request_alignments);
  std::copy(std::begin(native.request_counts),std::end(native.request_counts),value.request_counts);
  *out=value;return true;
}
}

extern "C" size_t mlx_distributed_cpu_eval_storage_controls(mlx_array output) {
  const size_t c=sizeof(mlx_distributed_cpu_eval_storage)*2+sizeof(mlx_array)+
      sizeof(mlx::core::distributed::CpuDistributedStorage)+sizeof(void*)*3+sizeof(bool)+sizeof(size_t)*2;
  if(!output.ctx)return c;
  const size_t native=mlx::core::distributed::cpu_distributed_storage_inspection_controls(
      *static_cast<const mlx::core::array*>(output.ctx));
  return native>SIZE_MAX-c?SIZE_MAX:native+c;
}
extern "C" bool mlx_distributed_query_cpu_eval_storage(mlx_distributed_cpu_eval_storage* out,mlx_array output) {
  if(!out || !output.ctx)return false;
  mlx::core::distributed::CpuDistributedStorage native;
  if(!mlx::core::distributed::cpu_distributed_storage(*static_cast<const mlx::core::array*>(output.ctx),native))return false;
  return publish(out,native,mlx_distributed_cpu_eval_storage_controls(output));
}

extern "C" size_t mlx_distributed_cpu_source_storage_controls(mlx_distributed_group group) {
  const size_t c=sizeof(mlx_distributed_cpu_eval_storage)*2+sizeof(mlx_array)+sizeof(mlx_distributed_group)+
      sizeof(mlx::core::distributed::CpuDistributedStorage)+sizeof(void*)*3+sizeof(bool)*2+
      sizeof(size_t)*3+sizeof(unsigned)+sizeof(int);
  if(!group.ctx)return c;
  const size_t native=mlx::core::distributed::cpu_distributed_source_inspection_controls(
      *static_cast<const mlx::core::distributed::Group*>(group.ctx));
  return native>SIZE_MAX-c?SIZE_MAX:native+c;
}
extern "C" bool mlx_distributed_query_cpu_source_storage(mlx_distributed_cpu_eval_storage* out,
    mlx_distributed_group group,mlx_array input,unsigned operation,int peer) {
  using namespace mlx::core::distributed;
  if(!out || !group.ctx || !input.ctx || operation>5)return false;
  CpuDistributedStorage native;
  if(!cpu_distributed_source_storage(*static_cast<const Group*>(group.ctx),
      *static_cast<const mlx::core::array*>(input.ctx),static_cast<GroupWorkerOperation>(operation),peer,native))return false;
  return publish(out,native,mlx_distributed_cpu_source_storage_controls(group));
}

extern "C" size_t mlx_distributed_cpu_layout_storage_controls(mlx_distributed_group group) {
  using namespace mlx::core::distributed;
  const size_t frames=sizeof(GroupTensorLayout)+sizeof(std::optional<GroupTensorLayout>)+
      sizeof(std::span<const int>)+sizeof(mlx::core::Dtype)+sizeof(mlx_dtype)+sizeof(unsigned)+sizeof(int)+
      sizeof(void*)*5+sizeof(size_t)*4+sizeof(CpuDistributedStorage)+sizeof(GroupConstructorStorage)+
      sizeof(mlx_distributed_cpu_eval_storage)+sizeof(mlx_distributed_constructor_storage);
  const size_t cpu=mlx_distributed_cpu_source_storage_controls(group);
  const size_t constructor=mlx_distributed_group_constructor_storage_controls();
  if(cpu>SIZE_MAX-frames || constructor>SIZE_MAX-frames-cpu)return SIZE_MAX;
  return frames+cpu+constructor;
}
extern "C" bool mlx_distributed_query_cpu_layout_storage(mlx_distributed_cpu_eval_storage* out,
    mlx_distributed_constructor_storage* constructor_out,mlx_distributed_group group,
    const int* shape,size_t rank,mlx_dtype dtype,unsigned operation,int peer) {
  using namespace mlx::core::distributed;
  if(!out || !constructor_out || !group.ctx || (rank&&!shape) || operation>5 ||
      dtype<MLX_BOOL || dtype>MLX_COMPLEX64)return false;
  const auto layout=GroupTensorLayout::checked(std::span<const int>(shape,rank),mlx_dtype_to_cpp(dtype));
  if(!layout)return false;
  const auto& native_group=*static_cast<const Group*>(group.ctx);
  CpuDistributedStorage native;
  GroupConstructorStorage constructor;
  if(!cpu_distributed_source_storage(native_group,*layout,static_cast<GroupWorkerOperation>(operation),peer,native) ||
      !constructor_storage(native_group,*layout,static_cast<GroupWorkerOperation>(operation),peer,constructor))return false;
  mlx_distributed_cpu_eval_storage value{};
  mlx_distributed_constructor_storage constructed{};
  if(!publish(&value,native,mlx_distributed_cpu_layout_storage_controls(group)) ||
      !mlx_distributed_publish_constructor_storage(&constructed,constructor))return false;
  *out=value;*constructor_out=constructed;return true;
}

extern "C" size_t mlx_distributed_variable_storage_controls(mlx_distributed_group group) {
  const size_t base=mlx_distributed_cpu_layout_storage_controls(group);
  const size_t frames=sizeof(std::span<const size_t>)*3+sizeof(size_t)*4+sizeof(bool)*2+
      sizeof(mlx_array)*2+sizeof(mlx_stream)+sizeof(mlx_submission_observer)+sizeof(void*)*7+
      mlx::core::distributed::VariableAllToAllGeometry::inspection_controls();
  return base>SIZE_MAX-frames?SIZE_MAX:base+frames;
}
extern "C" bool mlx_distributed_query_variable_layout_storage(mlx_distributed_cpu_eval_storage* out,
    mlx_distributed_constructor_storage* constructor_out,mlx_distributed_group group,
    const int* shape,size_t rank,mlx_dtype dtype,const size_t* matrix,
    size_t matrix_entries,bool transposed) {
  using namespace mlx::core::distributed;
  if(!out || !constructor_out || !group.ctx || !shape || !rank || !matrix || !matrix_entries ||
      dtype<MLX_BOOL || dtype>MLX_COMPLEX64)return false;
  const auto layout=GroupTensorLayout::checked(std::span<const int>(shape,rank),mlx_dtype_to_cpp(dtype));
  if(!layout)return false;
  const auto& native_group=*static_cast<const Group*>(group.ctx);
  const auto counts=std::span<const size_t>(matrix,matrix_entries);
  CpuDistributedStorage native;
  GroupConstructorStorage constructor;
  if(!cpu_distributed_variable_source_storage(native_group,*layout,counts,transposed,native) ||
      !variable_constructor_storage(native_group,*layout,counts,transposed,constructor))return false;
  mlx_distributed_cpu_eval_storage value{};
  mlx_distributed_constructor_storage constructed{};
  const size_t controls=mlx_distributed_variable_storage_controls(group);
  if(!publish(&value,native,controls) ||
      !mlx_distributed_publish_constructor_storage(&constructed,constructor) ||
      controls>SIZE_MAX-constructed.named_control_bytes)return false;
  constructed.named_control_bytes+=controls;
  *out=value;*constructor_out=constructed;return true;
}

extern "C" bool mlx_distributed_query_variable_envelope_layout_storage(
    mlx_distributed_cpu_eval_storage* out,mlx_distributed_constructor_storage* constructor_out,
    mlx_distributed_group group,const int* shape,size_t rank,mlx_dtype dtype,
    size_t maximum_receive_rows) {
  using namespace mlx::core::distributed;
  if(!out || !constructor_out || !group.ctx || !shape || !rank ||
      dtype<MLX_BOOL || dtype>MLX_COMPLEX64)return false;
  const auto layout=GroupTensorLayout::checked(std::span<const int>(shape,rank),mlx_dtype_to_cpp(dtype));
  if(!layout)return false;
  const auto& native_group=*static_cast<const Group*>(group.ctx);
  CpuDistributedStorage native;
  GroupConstructorStorage constructor;
  if(!cpu_distributed_variable_envelope_storage(native_group,*layout,maximum_receive_rows,native) ||
      !variable_envelope_constructor_storage(native_group,*layout,maximum_receive_rows,constructor))return false;
  mlx_distributed_cpu_eval_storage value{};
  mlx_distributed_constructor_storage constructed{};
  const size_t controls=mlx_distributed_variable_envelope_storage_controls(group);
  if(controls==SIZE_MAX || !publish(&value,native,controls) ||
      !mlx_distributed_publish_constructor_storage(&constructed,constructor) ||
      controls>SIZE_MAX-constructed.named_control_bytes)return false;
  constructed.named_control_bytes+=controls;
  *out=value;*constructor_out=constructed;return true;
}

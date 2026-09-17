#include "mlx/c/distributed_group.h"
#include "mlx/distributed/distributed.h"
#include <cstdint>
extern "C" size_t mlx_distributed_group_persistent_storage_controls(mlx_distributed_group group) {
  const size_t c=sizeof(mlx_distributed_persistent_storage)*2+sizeof(mlx_distributed_group)+
      sizeof(mlx::core::distributed::GroupPersistentStorage)+sizeof(void*)*3+sizeof(size_t)*2+sizeof(bool);
  if(!group.ctx)return c;
  const size_t native=static_cast<const mlx::core::distributed::Group*>(group.ctx)->persistent_storage_controls();
  return native>SIZE_MAX-c?SIZE_MAX:native+c;
}
extern "C" bool mlx_distributed_group_persistent_storage(mlx_distributed_persistent_storage* out,mlx_distributed_group group) {
  using namespace mlx::core::distributed;
  if(!out||!group.ctx)return false;
  GroupPersistentStorage native;
  if(!static_cast<const Group*>(group.ctx)->persistent_storage(native))return false;
  mlx_distributed_persistent_storage value{};
  value.storage_identity=native.storage_identity;
  value.kind=static_cast<unsigned>(native.kind);value.unresolved=native.unresolved;
  value.wrapper_bytes=sizeof(Group);value.shared_owner_bytes=native.shared_owner_bytes;
  value.shared_owner_alignment=native.shared_owner_alignment;
  value.socket_vector_bytes=native.socket_vector_bytes;value.communication_buffer_bytes=native.communication_buffer_bytes;
  value.pool_vector_bytes=native.pool_vector_bytes;value.socket_nodes=native.socket_nodes;
  value.socket_node_bytes=native.socket_node_bytes;value.socket_bucket_bytes=native.socket_bucket_bytes;
  value.thread_count=native.thread_count;value.thread_cpp_runtime_bytes=native.thread_cpp_runtime_bytes;
  if(sizeof(Group)>SIZE_MAX-native.retained_cpp_bytes)return false;
  value.retained_cpp_bytes=native.retained_cpp_bytes+sizeof(Group);
  const size_t c=mlx_distributed_group_persistent_storage_controls(group);
  if(c==SIZE_MAX)return false;value.controls=c;
  *out=value;return true;
}

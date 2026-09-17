#pragma once
#include "mlx/c/distributed_group.h"
#include "mlx/distributed/constructor.h"
#include <algorithm>
#include <limits>
inline bool mlx_distributed_publish_constructor_storage(mlx_distributed_constructor_storage* out,
    const mlx::core::distributed::GroupConstructorStorage& native) noexcept {
  mlx_distributed_constructor_storage value{};
  value.output_rank=native.output_rank; value.output_elements=native.output_elements;
  value.primitives=native.primitives; value.input_edges=native.input_edges;
  value.blocks=native.blocks; value.header_bytes=native.header_bytes; value.header_alignment=native.header_alignment;
  value.slots_bytes=native.slots_bytes; value.slots_alignment=native.slots_alignment;
  value.reserved_alignment=native.reserved_alignment; value.requested_bytes=native.requested_bytes;
  value.allocation_extents=native.allocation_extents;
  const size_t controls=mlx_distributed_group_constructor_storage_controls();
  if(controls>SIZE_MAX-native.named_control_bytes)return false;
  value.named_control_bytes=native.named_control_bytes+controls;
  std::copy(std::begin(native.request_bytes),std::end(native.request_bytes),value.request_bytes);
  std::copy(std::begin(native.request_alignments),std::end(native.request_alignments),value.request_alignments);
  std::copy(std::begin(native.request_counts),std::end(native.request_counts),value.request_counts);
  *out=value;return true;
}
